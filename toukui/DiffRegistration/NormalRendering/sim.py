import pyvista as pv
import numpy as np
import cv2
import os
import math

def MorecoordinateTransformation(rvec, tvec, meshList, objPointList, reverse=False):
    # # rvec是旋转向量，可以通过Rodrigues函数转换为旋转矩阵
    # R, _ = cv2.Rodrigues(rvec)
    # 统一RT, 避免格式错误
    # print(rvec.shape, tvec.shape)
    if rvec.shape == (3, 3):
        R = rvec
    else:
        R, _ = cv2.Rodrigues(rvec)
    if tvec.shape == (3, ):
        tvec = np.reshape(tvec, (3, 1))
    # print(R, tvec)
    
    if reverse:
        R = R.T
        tvec = -R @ tvec

    # 使用numpy.concatenate进行拼接
    RT = np.concatenate((R, tvec), axis=1)
    # 将mesh也同步进行刚性变换
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = tvec[:,0]
    # ===================================================================
    # 对列表中的点云和实体施加刚性变换
    resMeshList = []
    resPointList = []

    for points in objPointList:
        resPoints = points.copy()
        homogeneous_coordinates = cv2.convertPointsToHomogeneous(resPoints)
        homogeneous_coordinates = homogeneous_coordinates[:, 0, :]
        
        result_at = RT @ homogeneous_coordinates.T
        result_at = result_at.T
        # # 解算错误排除
        # if np.mean(result_at[2, :]) < 0:
        #     return False, [], []
        resPointList.append(result_at)

    for mesh in meshList:
        Resmesh = mesh.copy()
        Resmesh.transform(transformation_matrix)
        resMeshList.append(Resmesh)

    return True, resMeshList, resPointList

class Simulation3D():
    def __init__(self, objHpath, objTKpath, texturePath, texturePathC, Rrange=0.1, Trange=1, w_s=(1024,768), BackgroundColor=(28,40,51)):
        self.obj_file_path = objHpath
        self.meshTK = pv.read(objTKpath)
        self.meshOri = pv.read(objHpath)
        self.BackgroundColor = BackgroundColor
        self.w_s = w_s
        self.Rrange = Rrange
        self.Trange = Trange
        self.d = 500
        self.CameraPosition = [0, self.d, 0]
        self.CameraView_up = [0, 0, 1]  # 设置视角向上的方向
        self.CameraFocal_point = [0, 0, 0]  # 设置摄像头焦点
        # fR = np.loadtxt('ProcessImg/' + 'mtxr.txt')
        # fR = (fR[0, 0] + fR[1, 1]) / 2
        # print(fR)
        # dpz = d + 0
        # self.view_angle = 2*math.degrees(math.atan(dpz/fR))  # 设置视场角
        self.view_angle = 50  # 设置视场角
        # print(dpz/math.tan(math.radians(self.view_angle / 2)))
        # print(2*math.degrees(math.atan(dpz/fR)))
        # self.scale = 5 # 设置焦距
        # 创建一个摄像头并设置畸变参数
        # self.camera = pv.Camera()
        # self.camera.position = [d, d, d]
        # # self.camera.focal_point = [0, 0, 0]  # 设置摄像头焦点
        # # self.camera.viewup = [1, 0, 0]  # 设置视角向上的方向
        # self.camera.clipping_range = [0.1, 1000]
        # self.camera.view_angle = 90

        self.texture1 = pv.read_texture(texturePath) # 加载棋盘贴图
        self.textureC = pv.read_texture(texturePathC) # 加载棋盘贴图
        # 创建一个平面网格
        self.plane1 = pv.Plane(center=(-165.9852+25, 24.23842, -20-25), direction=(0, 1, 0), i_size=70, j_size=50)
        self.plane2 = pv.Plane(center=(164.9852-25, 24.23842, -20-25), direction=(0, 1, 0), i_size=70, j_size=50)
        self.planeC = pv.Plane(center=(0, 24.23842, 0), direction=(0, 1, 0), i_size=150, j_size=250)
        r = (self.d-700) / 1000
        self.planeT = pv.Plane(center=(0, 24.23842, 0), direction=(0, 1, 0), i_size=150*(2+r), j_size=250*(2+r))


    def GenerateImages(self, PositionOffset, AngleOffset):

        # 产生随机的旋转和平移
        rotation_matrix = (np.random.random((3)) -0.5) * self.Rrange + AngleOffset
        translation_vector = (np.random.random((3)) -0.5) * self.Trange + PositionOffset
        # rotation_matrix = np.loadtxt('ProcessImg\head0s2cr.txt')
        # translation_vector = np.loadtxt('ProcessImg\head0s2ct.txt')

        _, [mesh, meshTK, plane1, plane2], [] = MorecoordinateTransformation(rotation_matrix, translation_vector, [self.meshOri, self.meshTK, self.plane1, self.plane2], [])
        # self.meshOri = mesh
        # self.plane1 = plane1
        # self.plane2 = plane2
        # # =====================================================
        # plotter = pv.Plotter()
        plotter = pv.Plotter(off_screen=True)
        # 将背景颜色设置为灰色
        plotter.background_color = self.BackgroundColor
        # 启用抗锯齿
        plotter.enable_anti_aliasing()
        plotter.add_mesh(mesh, color=(255,255,255), show_edges=False)
        plotter.add_mesh(meshTK, color=(162,66,66), show_edges=False)

        # # 创建一个场景并添加平面网格
        
        plotter.add_mesh(plane1, texture=self.texture1)
        plotter.add_mesh(plane2, texture=self.texture1)

        plotter.show_axes()

        # # 设置摄像头位置和方向
        # # 这里使用的是一个具有特定视角的方向矩阵，你可以根据需要进行调整
        plotter.camera_position = [
            self.CameraPosition,  # 摄像头位置
            self.CameraFocal_point,  # 摄像头焦点
            self.CameraView_up  # 视角向上的方向
        ]
        plotter.camera.view_angle = self.view_angle
        # plotter.enable_ambient_light()  # 启用环境光
        # 设置摄像头
        # plotter.set_scale(self.scale)
        # plotter.camera.SetFocalLength(5)
        # # 将摄像头锁定，以便在保存图像时保持相同的视角
        # plotter.enable_eye_dome_lighting()
        # plotter.enable_parallel_projection()
        savePath = self.obj_file_path.split('.')[0] + '.png'

        plotter.screenshot(savePath, window_size=self.w_s)
        plotter.close()
        # plotter.show()


        return savePath

    def GenerateImagesCalibration(self, savePathfile, CalibNum, TS=False):
        if not os.path.exists(savePathfile):
            os.makedirs(savePathfile)
        xpy = 120
        ypy = 70
            

        for index in range(CalibNum):
            r = 1
            planeC = self.planeC
            if index == 0:
                rotation_matrix = np.array([0, 0, 0], dtype=np.float32)
                translation_vector = np.array([0, 0, 0], dtype=np.float32)
                if TS:
                    planeC = self.planeT
            elif index % 5 == 1:
                rotation_matrix = (np.random.random((3)) -0.5) * 0.3 * r
                translation_vector = np.array([xpy, 0, ypy], dtype=np.float32) + (np.random.random((3)) -0.5) * 20 * r
            elif index % 5 == 2:
                rotation_matrix = (np.random.random((3)) -0.5) * 0.3 * r
                translation_vector = np.array([xpy*-1, 0, ypy], dtype=np.float32) + (np.random.random((3)) -0.5) * 20 * r
            # 产生随机的旋转和平移
            elif index % 5 == 3:
                rotation_matrix = (np.random.random((3)) -0.5) * 0.3 * r
                translation_vector = np.array([xpy, 0, ypy*-1], dtype=np.float32) + (np.random.random((3)) -0.5) * 20 * r
            elif index % 5 == 4:
                rotation_matrix = (np.random.random((3)) -0.5) * 0.3 * r
                translation_vector = np.array([xpy*-1, 0, ypy*-1], dtype=np.float32) + (np.random.random((3)) -0.5) * 20 * r
            else:
                rotation_matrix = (np.random.random((3)) -0.5) * 0.3 * r
                translation_vector = (np.random.random((3)) -0.5) * 200 * r

            _, [planeC], [] = MorecoordinateTransformation(rotation_matrix, translation_vector, [planeC], [])

            # # =====================================================

            plotter = pv.Plotter(off_screen=True)
            # 将背景颜色设置为灰色
            plotter.background_color = self.BackgroundColor
            # 启用抗锯齿
            plotter.enable_anti_aliasing()
            plotter.add_mesh(planeC, texture=self.textureC)
            # plotter.show_axes()

            # 设置摄像头位置和方向
            # 这里使用的是一个具有特定视角的方向矩阵，你可以根据需要进行调整
            plotter.camera_position = [
                self.CameraPosition,  # 摄像头位置
                self.CameraFocal_point,  # 摄像头焦点
                self.CameraView_up  # 视角向上的方向
            ]
            plotter.camera.view_angle = self.view_angle
            # plotter.enable_ambient_light()  # 启用环境光
            # plotter.set_scale(self.scale)
            # # 将摄像头锁定，以便在保存图像时保持相同的视角
            # plotter.enable_eye_dome_lighting()
            # plotter.enable_parallel_projection()
            savePath = os.path.join(savePathfile, str(index) + '.png')

            plotter.screenshot(savePath, window_size=self.w_s)
            plotter.close()

        # return savePath

    def getVideo(self, video_name, frameNum, offsetList_P, offsetList_A):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编码器
        
        # 使用 OpenCV VideoWriter 创建视频对象
        video = cv2.VideoWriter(video_name, fourcc, 30.0, self.w_s)
        for i in range(frameNum):
            print(i)
            if i % 10 == 0:
                video.write(cv2.imread(self.GenerateImages(offsetList_P[i], offsetList_A[i])))
            else:
                video.write(cv2.imread(self.obj_file_path.split('.')[0] + '.png'))


        # 释放视频对象
        video.release()
        return 



if __name__ == '__main__':
    # 0.1 * np.pi / 180
    simid = 'headGTY'
    s = Simulation3D('head/' + simid + '/headH.obj', 'helmet/stdTK.obj', 'element/qipan2.png', 'element/qipan25.png', 1 * np.pi / 180, 2)
    for mv in range(0, 1, 1):
        offsetList_P = []
        offsetList_A = []
        frameNum = 330
        # 静态偏移
        for i in range(frameNum):
            offsetList_P.append(np.array([0, 0, 0]))
            # offsetList_A.append(np.array([10 * np.pi / 180 * (i/frameNum), 0, 0]))  # 动态偏移
            offsetList_A.append(np.array([0, 0 * np.pi / 180, 0]))
        
        outname = r'videodata/videosim/' + simid + 'y_xzx_' + str(mv) + '.avi'
        s.getVideo(outname, frameNum, offsetList_P, offsetList_A)
    # s.GenerateImagesCalibration('St20240423', 60, TS=False)
    # s.GenerateImagesCalibration('St202404232', 1, TS=True)
    # print(s.GenerateImages())


