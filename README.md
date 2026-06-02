# Reflective Vest Detection

基于 YOLOv11 的反光衣检测系统，支持图像、视频和实时摄像头推理，提供 RESTful API 接口。

## 功能特性

- 单张/批量图像检测
- 视频文件处理与结果导出
- 实时摄像头推理
- RESTful API 服务
- ONNX 模型导出
- 合规性分析与结构化输出

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 图像检测
python scripts/run_image.py --source data/images/

# 视频处理
python scripts/run_video.py --source data/videos/

# 摄像头实时检测
python scripts/run_camera.py

# 启动 API 服务
python src/api/app.py
```

## 项目结构

```
├── configs/      # 配置文件
├── data/         # 输入数据
├── outputs/      # 推理输出
├── scripts/      # 入口脚本
├── src/          # 核心源码
│   ├── core/     # 检测核心
│   ├── services/ # 业务服务
│   ├── utils/    # 工具函数
│   ├── schemas/  # 数据结构定义
│   └── api/      # API 服务
├── tests/        # 测试
└── weights/      # 模型权重
```

## 配置

复制 `.env.example` 为 `.env` 并修改配置：

```bash
cp .env.example .env
```

## License

MIT
