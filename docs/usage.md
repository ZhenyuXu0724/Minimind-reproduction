# MiniMind 仓库使用说明

可以直接阅读实验材料，或只使用 Python 标准库复核日志，无须安装模型依赖。

## 核验实验记录

从仓库根目录运行：

```powershell
python tools/summarize_exports.py
python tools/check_project.py
```

第一条命令从 CSV 与控制台日志重建 `experiments/summary.json`；第二条检查代码语法、tokenizer JSON、文档链接和指标一致性。它们不会执行训练或加载模型。GitHub Actions 使用同样的检查。

## 可选推理

推理需要兼容的 PyTorch、transformers 等运行依赖及自训权重。历史环境参考 `configs/environment_record.md`，上游依赖位于 `code/requirements.txt`，三次实际依赖快照在 `configs/run_environments/`。当前整理过程未验证安装或模型加载。

从 `code/` 目录执行下列命令可使用本地保存的 SFT 1 轮权重：

```powershell
python eval_llm.py --load_from model --save_dir "../checkpoints/sft epoch=1" --weight full_sft --hidden_size 768 --num_hidden_layers 8 --device cuda --max_new_tokens 512
```

这是按现有路径和脚本参数整理的使用示例，未执行验证。权重仅本地保留，公开仓库不包含权重文件；获得同名权重后需放到对应目录。2 轮模型使用目录 `../checkpoints/sft epoch=2` 和前缀 `full_sft_e2`，预训练模型使用目录 `../checkpoints` 和前缀 `pretrain`。

脚本将相对路径按工作目录解析，因此评估从 `code/` 执行，历史训练命令从 `code/trainer/` 执行。实际训练参数和命令在 `docs/experiment_records.md` 中归档，无须为展示项目重新训练。

历史评估每个问题随机设置 seed，不能保证逐字重现旧截图。原始报告中的对比仅是定性样例。
