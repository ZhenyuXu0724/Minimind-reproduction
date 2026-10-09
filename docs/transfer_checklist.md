# MiniMind 原材料转移清单

补充汇总已提供参数；swanlab_export 中的三份 CSV 和三份控制台日志已收到并归档。2026 年 10 月 9 日解压 transfer_remaining.zip，以下配套代码、tokenizer、依赖和元数据均已收到。无须重新训练或复制数据集。文件大小和 SHA256 见 [transfer_manifest.csv](transfer_manifest.csv)。

| 原电脑相对路径 | 工作副本目的地 | 用途 |
| --- | --- | --- |
| minimind/dataset/lm_dataset.py | code/dataset/lm_dataset.py | 数据处理 |
| minimind/trainer/trainer_utils.py | code/trainer/trainer_utils.py | 训练与加载工具 |
| minimind/model/model_lora.py | code/model/model_lora.py | 推理导入依赖 |
| minimind/model/tokenizer.json 和 tokenizer_config.json 及其他 tokenizer 配套文件 | code/model/ | 与历史训练一致的 tokenizer |
| minimind/requirements.txt 和 LICENSE | code/requirements.txt 和 LICENSE | 上游依赖及许可 |
| minimind/trainer/swanlog/run-*/files/requirements.txt | configs/run_environments/各 run 单独存放 | 三次真实依赖快照 |
| minimind/trainer/swanlog/run-*/files/swanlab-metadata.json | docs/archive/metadata/各 run 单独存放 | 运行配置来源，原件本地归档 |
| 复现Minimind/swanlab_export/run-*.csv | experiments/logs/ | 已收到并核验 |
| 复现Minimind/swanlab_export/run-*_console.txt | experiments/logs/ | 已收到并核验 |

原元数据可能含机器路径和环境字段，公开仓库使用整理后的参数，不必上传全部原件。已核对元数据中的历史 commit 和训练命令，并保留压缩包的 Apache 2.0 许可文本；CSV 指标已核验。三个自训权重已在当前电脑，无须重复转移。
