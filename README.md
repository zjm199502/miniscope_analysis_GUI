# Miniscope 分析软件

一个轻量化的 miniscope 单光子钙成像分析软件，带中文图形界面，面向教学和日常实验室使用。
目标是在普通笔记本上跑完整流程，并能和动物行为视频做同步分析。

## 功能路线

| 步骤 | 内容 | 状态 |
| --- | --- | --- |
| 1. 导入数据 | 读取 UCLA Miniscope V4 文件夹（avi、metaData.json、timeStamps.csv），检查丢帧 | ✅ |
| 2. 预处理 | 截取时间段、框选区域、空间/时间降采样，分块写入硬盘 | ✅ |
| 3. 运动校正 | FFT 相位相关刚性配准，可选分块非刚性 | 开发中 |
| 4. 去除背景 | 中值滤波 + 形态学 tophat | 开发中 |
| 5. 细胞提取 | 局部相关图 × PNR 找种子，局部 NMF 精修，人工审核 | 开发中 |
| 6. 信号与 spike | ΔF/F，OASIS spike 推断，导出 | 开发中 |
| 7. 行为分析 | 手动导入行为视频，质心追踪，区域划分 | 开发中 |
| 8. 同步分析 | 手动对齐 / LED 闪光对齐，联合视图 | 开发中 |

## 安装

需要 Python 3.10 或更新版本。

```bash
git clone https://github.com/zjm199502/miniscope_analysis_GUI.git
cd miniscope_analysis_GUI
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
pip install -e .
```

## 使用

```bash
python -m miniscope_gui
```

1. 在「导入数据」里选择 `My_V4_Miniscope` 文件夹。
2. 选择一个**空的**输出文件夹。所有结果都写在这里，原始数据不会被修改。
   预处理后的视频可能有几 GB，建议放在本地硬盘，不要放在 OneDrive 等同步盘里。
3. 在「预处理」里设置参数，点「开始预处理」。

不打开界面也可以用命令行：

```bash
python -m miniscope_gui.cli info "D:\data\My_V4_Miniscope"
python -m miniscope_gui.cli preprocess "D:\data\My_V4_Miniscope" "D:\analysis\mouse1" --bin 2 --tbin 1
```

## 输出文件夹里有什么

| 文件 | 内容 |
| --- | --- |
| `project.json` | 原始数据路径、每一步的参数和完成时间 |
| `movie_pre.npy` | 预处理后的视频，形状 (帧, 高, 宽)，uint8 |
| `frame_times_pre.npy` | 每帧的时间（秒） |
| `mean_projection.npy` / `max_projection.npy` | 平均 / 最大投影 |

## 开发

```bash
pip install -e .[dev]
pytest
```
