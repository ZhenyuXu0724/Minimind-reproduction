# MiniMind 复现与实验分析

在 RTX 5070 Laptop 8 GB 单卡上完成约 64M 参数语言模型的预训练与全参数 SFT，比较 1 轮和 2 轮微调结果。使用约 127 万条预训练文本和 91 万条指令样本，预训练记录 loss 从 8.5076 降至 1.8339。

## 完成的工作

- 排查 Windows 下 CUDA、DLL 和依赖兼容问题，完成三次训练并用 SwanLab 记录指标。
- 阅读并注释模型、数据处理、训练与推理源码，理解 RMSNorm、RoPE、GQA 和 SwiGLU。
- 对比自训模型与作者模型的生成样例，分析重复、异常符号和技术幻觉；代码改动主要为学习注释。

详细配置、曲线和结果见 [技术报告](docs/technical_report.md)；[Word 报告](docs/technical_report.docx) 附有 4 页手写学习笔记，保留原笔迹、公式和图示。SFT 2 轮在现有样例中回答更长，但技术准确性未见明确改善；历史评估未固定 seed，结论限于定性观察。

## 使用

从仓库根目录核验实验记录，仅需要 Python 3.10 或更新版本：

```powershell
python tools/summarize_exports.py
python tools/check_project.py
```

脚本检查日志统计、源文件语法、tokenizer JSON 和文档链接，GitHub Actions 执行同样的检查。

推理需要模型依赖及权重。历史环境为 Python 3.10.20、torch 2.11.0+cu128、transformers 4.57.6；依赖见 [上游清单](code/requirements.txt) 和 [实际环境快照](configs/requirements.freeze.txt)。从 `code/` 目录使用本地 SFT 1 轮权重：

```powershell
python eval_llm.py --load_from model --save_dir "../checkpoints/sft epoch=1" --weight full_sft --hidden_size 768 --num_hidden_layers 8 --device cuda --max_new_tokens 512
```

权重与数据集未上传；该推理命令按现有路径整理，尚未重新验证模型加载。SFT 2 轮使用目录 `../checkpoints/sft epoch=2` 和前缀 `full_sft_e2`。

## 文件与来源

```text
code/          模型、数据处理、训练、推理和 tokenizer
configs/       实际训练参数和环境快照
experiments/   原始日志、统计汇总、曲线和生成截图
docs/          技术报告
tools/         日志汇总与静态检查
```

基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 的提交 `4497610ec0a85d2d0a3db488fd4c1e2f12a416ab`，保留 Apache 2.0 许可文本及 [来源说明](NOTICE)。个人工作为训练复现、源码学习和实验分析。
