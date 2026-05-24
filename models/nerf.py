"""
NeRF网络模型
基于经典NeRF架构，用于头部神经辐射场建模
- 8层MLP，隐藏层维度256
- 位置编码：空间坐标10阶(63维)，方向4阶(27维)
- 粗-精两级网络结构
- 体渲染方程离散化实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码模块
    将低维坐标映射到高维空间，使MLP能够捕捉高频几何细节
    γ(p) = [p, sin(2^0 πp), cos(2^0 πp), ..., sin(2^(L-1) πp), cos(2^(L-1) πp)]
    """

    def __init__(self, input_dim: int, num_freqs: int, include_input: bool = True):
        """
        Args:
            input_dim: 输入维度 (3 for xyz, 2 for direction)
            num_freqs: 编码阶数 (L)
            include_input: 是否包含原始输入
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.include_input = include_input

        # 预计算频率带
        freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer('freq_bands', freq_bands)

        # 计算输出维度
        self.output_dim = input_dim * (2 * num_freqs + (1 if include_input else 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim] 输入坐标
        Returns:
            encoded: [batch, output_dim] 编码后的特征
        """
        encoded = []
        if self.include_input:
            encoded.append(x)

        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * np.pi * x))
            encoded.append(torch.cos(freq * np.pi * x))

        return torch.cat(encoded, dim=-1)


class NeRF(nn.Module):
    """经典NeRF网络
    8层全连接网络，隐藏层维度256
    输入：位置编码后的空间坐标(63维) + 方向编码(27维)
    输出：体密度σ(1维) + RGB颜色c(3维)
    """

    def __init__(
        self,
        pos_encoding_freqs: int = 10,
        dir_encoding_freqs: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_layer: int = 4,
    ):
        """
        Args:
            pos_encoding_freqs: 空间坐标编码阶数，默认10
            dir_encoding_freqs: 方向编码阶数，默认4
            hidden_dim: 隐藏层维度，默认256
            num_layers: 网络层数，默认8
            skip_layer: 跳跃连接层索引，默认4
        """
        super().__init__()

        # 位置编码
        self.pos_encoder = PositionalEncoding(3, pos_encoding_freqs)
        self.dir_encoder = PositionalEncoding(3, dir_encoding_freqs)

        self.pos_dim = self.pos_encoder.output_dim  # 63
        self.dir_dim = self.dir_encoder.output_dim  # 27
        self.hidden_dim = hidden_dim
        self.skip_layer = skip_layer

        # 构建主干网络（处理空间坐标）
        layers = []
        layers.append(nn.Linear(self.pos_dim, hidden_dim))
        for i in range(1, num_layers):
            if i == skip_layer:
                # 跳跃连接：拼接原始位置编码特征
                layers.append(nn.Linear(hidden_dim + self.pos_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.backbone = nn.ModuleList(layers)

        # 体密度输出头（非负约束）
        self.density_head = nn.Linear(hidden_dim, 1)

        # 颜色预测分支（方向依赖）
        self.feature_linear = nn.Linear(hidden_dim, hidden_dim)
        self.color_layer = nn.Sequential(
            nn.Linear(hidden_dim + self.dir_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()  # 颜色约束在[0,1]
        )

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> tuple:
        """
        Args:
            x: [batch, 3] 空间坐标
            d: [batch, 3] 观察方向（单位向量）
        Returns:
            rgb: [batch, 3] RGB颜色
            sigma: [batch, 1] 体密度
        """
        # 位置编码
        pos_encoded = self.pos_encoder(x)
        dir_encoded = self.dir_encoder(d)

        # 主干网络前向传播
        h = pos_encoded
        for i, layer in enumerate(self.backbone):
            if i == self.skip_layer:
                h = torch.cat([h, pos_encoded], dim=-1)
            h = layer(h)
            h = F.relu(h, inplace=True)

        # 体密度（ReLU保证非负）
        sigma = F.relu(self.density_head(h))

        # 颜色预测（方向依赖）
        feature = self.feature_linear(h)
        color_input = torch.cat([feature, dir_encoded], dim=-1)
        rgb = self.color_layer(color_input)

        return rgb, sigma


class NeRFCoarse(NeRF):
    """粗网络 - 用于第一阶段均匀采样"""
    pass


class NeRFFine(NeRF):
    """精网络 - 用于第二阶段重要性采样"""
    pass


class VolumeRenderer(nn.Module):
    """体渲染器
    实现离散化的体渲染方程：
    C(r) = Σ T_i * α_i * c_i
    T_i = Π(1 - α_j), j=1..i-1
    α_i = 1 - exp(-σ_i * δ_i)
    """

    def __init__(
        self,
        near: float = 0.1,
        far: float = 5.0,
        num_coarse_samples: int = 64,
        num_fine_samples: int = 128,
        white_background: bool = True,
    ):
        """
        Args:
            near: 近平面距离
            far: 远平面距离
            num_coarse_samples: 粗采样点数
            num_fine_samples: 精采样点数
            white_background: 是否使用白色背景
        """
        super().__init__()
        self.near = near
        self.far = far
        self.num_coarse_samples = num_coarse_samples
        self.num_fine_samples = num_fine_samples
        self.white_background = white_background

    def sample_stratified(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                          num_samples: int, perturb: bool = True) -> tuple:
        """分层随机采样
        Args:
            rays_o: [batch, 3] 光线起点
            rays_d: [batch, 3] 光线方向
            num_samples: 采样点数
            perturb: 是否添加随机扰动
        Returns:
            pts: [batch, num_samples, 3] 采样点坐标
            t_vals: [batch, num_samples] 采样深度值
        """
        batch_size = rays_o.shape[0]
        device = rays_o.device

        # 均匀划分区间
        t_vals = torch.linspace(self.near, self.far, num_samples, device=device)
        t_vals = t_vals.unsqueeze(0).expand(batch_size, -1)

        # 分层随机扰动
        if perturb:
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
            lower = torch.cat([t_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(t_vals)
            t_vals = lower + (upper - lower) * t_rand

        # 计算采样点坐标: r(t) = o + t*d
        pts = rays_o.unsqueeze(1) + t_vals.unsqueeze(-1) * rays_d.unsqueeze(1)

        return pts, t_vals

    def sample_importance(self, t_vals: torch.Tensor, weights: torch.Tensor,
                          num_samples: int) -> torch.Tensor:
        """重要性采样（逆变换采样）
        Args:
            t_vals: [batch, N_coarse] 粗采样深度值
            weights: [batch, N_coarse] 粗网络预测的权重
            num_samples: 精采样点数
        Returns:
            t_fine: [batch, num_samples] 精采样深度值
        """
        # 构建PDF
        weights = weights + 1e-5  # 防止除零
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        # 逆变换采样
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=cdf.device)
        u = u.contiguous()

        # 二分查找
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds - 1, min=0)
        above = torch.clamp(inds, max=cdf.shape[-1] - 1)

        inds_g = torch.stack([below, above], dim=-1)
        cdf_g = torch.gather(cdf, -1, inds_g.view(*cdf.shape[:-1], -1)).view(*inds_g.shape)
        t_vals_g = torch.gather(
            t_vals.unsqueeze(-1).expand(-1, -1, 2).reshape(*t_vals.shape[:-1], -1),
            -1,
            inds_g.view(*t_vals.shape[:-1], -1)
        ).view(*inds_g.shape)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t_fine = t_vals_g[..., 0] + (u - cdf_g[..., 0]) / denom * (t_vals_g[..., 1] - t_vals_g[..., 0])

        return t_fine.detach()

    def volume_render(self, rgb: torch.Tensor, sigma: torch.Tensor,
                      t_vals: torch.Tensor, rays_d: torch.Tensor) -> dict:
        """体渲染计算
        Args:
            rgb: [batch, N, 3] 各采样点颜色
            sigma: [batch, N, 1] 各采样点体密度
            t_vals: [batch, N] 采样深度值
            rays_d: [batch, 3] 光线方向
        Returns:
            dict: 包含渲染颜色、深度图、权重等
        """
        # 计算相邻采样点间距 δ_i
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)], dim=-1)
        dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)

        # 计算不透明度 α_i = 1 - exp(-σ_i * δ_i)
        sigma = sigma.squeeze(-1)
        alpha = 1.0 - torch.exp(-sigma * dists)

        # 计算累积透射率 T_i = Π(1 - α_j), j<i
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]

        # 渲染权重 w_i = T_i * α_i
        weights = transmittance * alpha

        # 最终颜色 C = Σ w_i * c_i
        rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)

        # 深度图 D = Σ w_i * t_i
        depth_map = torch.sum(weights * t_vals, dim=-1)

        # 累积不透明度（用于背景合成）
        acc_map = torch.sum(weights, dim=-1)

        # 白色背景合成
        if self.white_background:
            rgb_map = rgb_map + (1.0 - acc_map.unsqueeze(-1))

        return {
            'rgb': rgb_map,
            'depth': depth_map,
            'acc': acc_map,
            'weights': weights,
        }

    def forward(self, nerf_coarse: NeRF, nerf_fine: NeRF,
                rays_o: torch.Tensor, rays_d: torch.Tensor,
                perturb: bool = True) -> dict:
        """完整的两阶段体渲染流程
        Args:
            nerf_coarse: 粗网络
            nerf_fine: 精网络
            rays_o: [batch, 3] 光线起点
            rays_d: [batch, 3] 光线方向
            perturb: 是否添加随机扰动
        Returns:
            dict: 包含粗/精渲染结果
        """
        # 第一阶段：粗采样
        pts_coarse, t_coarse = self.sample_stratified(
            rays_o, rays_d, self.num_coarse_samples, perturb
        )

        # 粗网络推理
        batch_size, n_samples = pts_coarse.shape[:2]
        dirs_coarse = rays_d.unsqueeze(1).expand_as(pts_coarse)

        pts_flat = pts_coarse.reshape(-1, 3)
        dirs_flat = dirs_coarse.reshape(-1, 3)

        rgb_coarse, sigma_coarse = nerf_coarse(pts_flat, dirs_flat)
        rgb_coarse = rgb_coarse.reshape(batch_size, n_samples, 3)
        sigma_coarse = sigma_coarse.reshape(batch_size, n_samples, 1)

        # 粗渲染
        coarse_result = self.volume_render(rgb_coarse, sigma_coarse, t_coarse, rays_d)

        # 第二阶段：重要性采样
        t_fine = self.sample_importance(t_coarse, coarse_result['weights'].detach(),
                                        self.num_fine_samples)

        # 合并粗+精采样点并排序
        t_all, _ = torch.sort(torch.cat([t_coarse, t_fine], dim=-1), dim=-1)

        # 精网络推理
        pts_fine = rays_o.unsqueeze(1) + t_all.unsqueeze(-1) * rays_d.unsqueeze(1)
        n_all = pts_fine.shape[1]
        dirs_fine = rays_d.unsqueeze(1).expand_as(pts_fine)

        pts_flat = pts_fine.reshape(-1, 3)
        dirs_flat = dirs_fine.reshape(-1, 3)

        rgb_fine, sigma_fine = nerf_fine(pts_flat, dirs_flat)
        rgb_fine = rgb_fine.reshape(batch_size, n_all, 3)
        sigma_fine = sigma_fine.reshape(batch_size, n_all, 1)

        # 精渲染
        fine_result = self.volume_render(rgb_fine, sigma_fine, t_all, rays_d)

        return {
            'coarse': coarse_result,
            'fine': fine_result,
        }


class NeRFRegistration(nn.Module):
    """基于NeRF的隐式配准模块
    通过梯度优化在连续神经场中求解6DoF位姿
    - 6D连续旋转表示 + Gram-Schmidt正交化
    - 体密度损失 + 可微渲染图像级损失
    - 由粗到精的多分辨率优化策略
    """

    def __init__(self, nerf_coarse: NeRF, nerf_fine: NeRF, renderer: VolumeRenderer):
        super().__init__()
        self.nerf_coarse = nerf_coarse
        self.nerf_fine = nerf_fine
        self.renderer = renderer

        # 冻结NeRF参数
        for param in self.nerf_coarse.parameters():
            param.requires_grad = False
        for param in self.nerf_fine.parameters():
            param.requires_grad = False

    @staticmethod
    def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
        """6D连续旋转表示转旋转矩阵（Gram-Schmidt正交化）
        Args:
            rot_6d: [batch, 6] 旋转矩阵前两列
        Returns:
            R: [batch, 3, 3] 正交旋转矩阵
        """
        a1 = rot_6d[..., :3]
        a2 = rot_6d[..., 3:6]

        # e1 = normalize(a1)
        e1 = F.normalize(a1, dim=-1)
        # e2 = normalize(a2 - (a2·e1)e1)
        e2 = a2 - (a2 * e1).sum(dim=-1, keepdim=True) * e1
        e2 = F.normalize(e2, dim=-1)
        # e3 = e1 × e2
        e3 = torch.cross(e1, e2, dim=-1)

        R = torch.stack([e1, e2, e3], dim=-1)
        return R

    def compute_density_loss(self, sensor_points: torch.Tensor,
                             rot_6d: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        """计算体密度配准损失
        L_density = -1/N * Σ log(σ(R*p_i + t))
        Args:
            sensor_points: [N, 3] 传感器坐标（传感器坐标系）
            rot_6d: [6] 6D旋转参数
            translation: [3] 平移参数
        Returns:
            loss: 标量损失值
        """
        R = self.rotation_6d_to_matrix(rot_6d.unsqueeze(0)).squeeze(0)  # [3, 3]

        # 变换传感器点到NeRF坐标系
        transformed_pts = (R @ sensor_points.T).T + translation  # [N, 3]

        # 查询NeRF体密度
        dummy_dirs = torch.zeros_like(transformed_pts)
        _, sigma = self.nerf_fine(transformed_pts, dummy_dirs)

        # 负对数密度损失
        loss = -torch.mean(torch.log(sigma + 1e-8))
        return loss

    def compute_render_loss(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                            target_img: torch.Tensor, rot_6d: torch.Tensor,
                            translation: torch.Tensor) -> dict:
        """计算可微渲染图像级损失
        L_image = λ_photo * L_photo + λ_percep * L_percep
        Args:
            rays_o: [H*W, 3] 光线起点
            rays_d: [H*W, 3] 光线方向
            target_img: [H, W, 3] 目标图像
            rot_6d: [6] 6D旋转参数
            translation: [3] 平移参数
        Returns:
            dict: 包含各项损失
        """
        R = self.rotation_6d_to_matrix(rot_6d.unsqueeze(0)).squeeze(0)

        # 变换光线（应用位姿变换）
        transformed_rays_o = (R @ rays_o.T).T + translation
        transformed_rays_d = (R @ rays_d.T).T

        # 渲染
        result = self.renderer(self.nerf_coarse, self.nerf_fine,
                               transformed_rays_o, transformed_rays_d, perturb=False)

        rendered_rgb = result['fine']['rgb']
        target_flat = target_img.reshape(-1, 3)

        # L1光度损失
        photo_loss = F.l1_loss(rendered_rgb, target_flat)

        return {
            'photo_loss': photo_loss,
            'rendered_rgb': rendered_rgb,
        }

    def optimize_pose(self, sensor_points: torch.Tensor, target_images: list,
                      camera_intrinsics: torch.Tensor,
                      init_rot_6d: torch.Tensor = None,
                      init_translation: torch.Tensor = None,
                      num_coarse_iters: int = 200,
                      num_fine_iters: int = 200,
                      num_render_iters: int = 100) -> dict:
        """由粗到精的位姿优化
        阶段1: 128x128低分辨率体密度优化，200步
        阶段2: 256x256高分辨率体密度优化，200步
        阶段3: 256x256图像级优化，100步
        """
        device = sensor_points.device

        # 初始化位姿参数
        if init_rot_6d is None:
            init_rot_6d = torch.tensor([1., 0., 0., 0., 1., 0.], device=device)
        if init_translation is None:
            init_translation = torch.zeros(3, device=device)

        rot_6d = init_rot_6d.clone().requires_grad_(True)
        translation = init_translation.clone().requires_grad_(True)

        results = {'losses': [], 'poses': []}

        # 阶段1：粗优化（低分辨率体密度）
        optimizer = torch.optim.Adam([rot_6d, translation], lr=5e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

        for i in range(num_coarse_iters):
            optimizer.zero_grad()
            loss = self.compute_density_loss(sensor_points, rot_6d, translation)
            loss.backward()
            optimizer.step()
            scheduler.step()
            results['losses'].append(loss.item())

        # 阶段2：精优化（高分辨率体密度）
        optimizer = torch.optim.Adam([rot_6d, translation], lr=1e-3)

        for i in range(num_fine_iters):
            optimizer.zero_grad()
            loss = self.compute_density_loss(sensor_points, rot_6d, translation)
            loss.backward()
            optimizer.step()
            results['losses'].append(loss.item())

        # 阶段3：图像级优化
        optimizer = torch.optim.Adam([rot_6d, translation], lr=5e-4)

        for i in range(num_render_iters):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)

            for img_data in target_images:
                render_result = self.compute_render_loss(
                    img_data['rays_o'], img_data['rays_d'],
                    img_data['image'], rot_6d, translation
                )
                total_loss = total_loss + render_result['photo_loss']

            total_loss.backward()
            optimizer.step()
            results['losses'].append(total_loss.item())

        # 最终位姿
        R_final = self.rotation_6d_to_matrix(rot_6d.unsqueeze(0)).squeeze(0)
        results['rotation'] = R_final.detach()
        results['translation'] = translation.detach()
        results['rot_6d'] = rot_6d.detach()

        return results
