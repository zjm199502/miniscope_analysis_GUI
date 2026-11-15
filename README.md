# Miniscope 分析软件

一个轻量化的 miniscope 单光子钙成像分析软件，带中文图形界面，面向教学和日常实验室使用。
目标是在普通笔记本上跑完整流程，并能和动物行为视频做同步分析。

## 功能

| 步骤 | 内容 |
| --- | --- |
| 1. 导入数据 | 读取 UCLA Miniscope V4 文件夹（avi、metaData.json、timeStamps.csv），按时间戳计算真实帧率，检查丢帧 |
| 2. 预处理 | 截取时间段、圆形视野、空间/时间降采样，分块写入硬盘 |
| 3. 运动校正 | 带通滤波 + FFT 相位相关，刚性或分块非刚性 |
| 4. 去除背景 | 中值滤波 + 形态学 tophat |
| 5. 细胞提取 | 相关图 × PNR 找种子，局部 HALS 非负矩阵分解（CNMF-E 精简版），人工检查、删除、合并 |
| 6. 信号与 spike | ΔF/F（滑动百分位基线），OASIS spike 推断，导出 CSV / MAT |
| 7. 行为分析 | 导入行为视频，场地/区域/比例尺，自动追踪，僵直检测；或导入 EthoVision / DeepLabCut 结果 |
| 8. 同步分析 | LED 闪光或手动对齐；联合播放；僵直/速度调制、事件前后平均；导出对齐数据 |
| 9. 建模分析 | GLM 编码模型、行为解码器、细胞组检测、CEBRA 式嵌入、HMM / AR-HMM / rSLDS 简化版、功能连接网络与图论、互信息与空间信息；全部带交叉验证和打乱检验 |

详细使用说明见 [docs/教程.md](docs/教程.md)。

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

Windows 上也可以直接双击 `packaging\install_and_run.bat`（第一次会自动安装）。
打包成独立的 exe：在仓库根目录运行 `packaging\build_windows.bat`。

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


## 发布新版本（维护者）

1. 修改 `miniscope_gui/__init__.py` 和 `pyproject.toml` 里的版本号，在 `CHANGELOG.md` 里写 `## vX.Y.Z` 一节（会显示在软件的更新对话框里）。
2. 如果换了依赖库、改了 `packaging/launcher.py`，或者新代码用到了以前没用过的标准库模块，把 `packaging/min_base.txt` 改成这个版本号（会重新打包整个程序）。
3. 合并到 main 后，在 Actions → “Build and release” 里运行，填入版本号。
   - 每个版本都会进入软件内更新通道（GitHub 上名为 `updates` 的 pre-release）。
   - 版本号末位是 0（x.y.0、x.y.10…）时，同时发布带安装包的正式 Release。
