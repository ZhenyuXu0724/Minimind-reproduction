# MiniMind 历史环境记录

以下是复现材料汇总提供的历史环境，不是对当前电脑的检测结果。

| 项目 | 汇总记录 |
| --- | --- |
| GPU | RTX 5070 Laptop GPU，8151 MiB |
| 驱动和 CUDA | 573.22，CUDA 12.8 |
| Python | 3.10.20 |
| torch | 2.11.0+cu128 |
| torchvision 和 torchaudio | 0.26.0，2.11.0 |
| transformers | 4.57.6 |
| datasets | 3.6.0 |
| numpy | 1.26.4 |
| SwanLab | 0.7.11 |

三个运行的完整依赖快照已归档在 `run_environments/`，文件内容一致。元数据确认 Python 3.10.20；pip freeze 确认表中版本。torchvision 和 torchaudio 的完整版本分别为 0.26.0+cu128、2.11.0+cu128。快照与上游 `code/requirements.txt` 分别保留，未在当前电脑安装或验证。

汇总的系统字段写为 Windows 10 和 build 10.0.26200，两者的产品名称对应关系待核对；README 仅称 Windows。运行耗时和显存口径见实验记录。不将未计量电费写成零成本。
