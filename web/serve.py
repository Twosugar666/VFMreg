"""
启动本地 Web 服务器以查看 VFMReg 项目主页

用法:
    python web/serve.py            # 默认 8000 端口
    python web/serve.py 8080       # 指定端口

访问:
    http://localhost:8000/
"""
import sys
import os
import http.server
import socketserver
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
ROOT = Path(__file__).parent.parent  # code/ 目录, 这样可以访问 demo_reports

os.chdir(ROOT)


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # 允许 CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, format, *args):
        sys.stdout.write(f'  {self.address_string()} - {format % args}\n')


def main():
    # 默认浏览器访问 web/index.html
    print('=' * 60)
    print(f'  🌐 VFMReg 项目主页本地服务')
    print('=' * 60)
    print(f'  📂 根目录: {ROOT}')
    print(f'  🔌 端口: {PORT}')
    print(f'  🔗 主页:    http://localhost:{PORT}/web/')
    print(f'  🔗 数据集:  http://localhost:{PORT}/web/pages/datasets.html')
    print(f'  🔗 实验结果:http://localhost:{PORT}/web/pages/results.html')
    print(f'  🔗 在线演示:http://localhost:{PORT}/web/pages/demo.html')
    print('=' * 60)
    print(f'  按 Ctrl+C 退出\n')

    try:
        with socketserver.TCPServer(('0.0.0.0', PORT), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n  👋 服务已停止')


if __name__ == '__main__':
    main()
