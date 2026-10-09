# MiniMind 项目整理记录

2026 年 10 月 8 日整理了 23 个原始文件。原目录保留不动，工作副本按功能分组；复制时逐一比对 SHA256。原始路径、复制路径和校验值见 `file_manifest.csv`。该清单记录的是复制时状态，评估脚本随后进行了下述语法修复。

## 文件保留方式

| 原目录 | 工作副本 | 用途 |
| --- | --- | --- |
| code_note 中的 Python | code 中的 model 和 trainer 及评估入口 | 注释代码 |
| code_note 中的 Markdown | docs/code_notes | 原学习笔记 |
| env | configs/environment_evidence | 历史环境截图 |
| experiments | checkpoints | 三个自训权重 |
| swanlab_figures | experiments/figures | 训练曲线 |
| model_outputs | experiments/evaluations | 定性生成样例 |
| Word 和 PPT | docs/archive | 原报告和汇报 |

## 代码修复

原始 `code_note/eval_llm.py` 中 `tokenizer.apply_chat_template(...)` 缺少一个闭合括号。仅在 `code/eval_llm.py` 工作副本中补齐，保留原文件。四个工作副本 Python 文件进行 AST 语法检查；通过语法检查不代表依赖齐全或能加载权重。

## 路径依赖

训练脚本导入 `model.model_minimind`、`dataset.lm_dataset` 与 `trainer.trainer_utils`。评估脚本还导入 `model.model_lora`。工作副本按照这些模块名称组织，不将训练脚本随意拆散到 configs。

训练保存目录默认 `../out`，续训状态目录写为 `../checkpoints`；评估默认从 `out` 读取权重、从 `model` 读取 tokenizer。本地归档权重所在的顶层 checkpoints 仅用于保存资料，不会自动成为脚本的加载目录。

## 补充材料后的状态

2026 年 10 月 9 日读入《复现材料汇总》，已补充历史 commit、实际训练命令、参数、数据条数、软件版本、指标、资源耗时与作者模型来源。原始汇总归档在 docs/archive，仅本地保存。

当前电脑未发现汇总所述完整工程。swanlab_export 已收到，六个原文件已复制到 experiments/logs 并逐一比对 SHA256；指标已用 CSV 与控制台日志核验。代码、tokenizer、依赖快照、上游许可证和原运行元数据均已收到，见 [转移清单](transfer_checklist.md)。数据和作者模型的历史 revision 及校验值仍未提供。

汇总中的“Windows 10”和 build 字段待核对；预训练步数是 DataLoader 次数，不是优化器更新次数；显存为定期采样的 GPU 监控值，不保证瞬时峰值。SFT 第二轮的 epoch seed 为 43，不能写成全程固定 42。

工作副本仍保留第一次整理的语法修复，不覆盖原注释代码。本轮仅更新资料，没有执行训练或推理，没有安装依赖或上传远程仓库。

## 配套材料归档

transfer_remaining.zip 中 13 个文件已按原相对路径归档，根目录 LICENSE 另复制一份上游许可文本。原有四份注释代码及权重未覆盖。全部七个 Python 文件通过 AST 语法检查，tokenizer JSON 格式有效，三个运行依赖快照一致。归档不等于运行验证，本轮没有安装包或执行模型。
