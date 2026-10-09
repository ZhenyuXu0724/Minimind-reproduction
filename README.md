# MiniMind 复现与实验分析

在 RTX 5070 Laptop 8 GB 单卡上复现约 64M 参数语言模型，完成 mini 数据预训练、全参数 SFT 及 1 轮与 2 轮生成结果对比。项目包含中文注释代码、源码笔记、SwanLab 曲线和技术报告；个人贡献集中在环境排查、训练实践与结果分析。

## 实验概览

| 实验 | 样本数 | 轮数 | 训练记录 loss 首值至末值 | 耗时 |
| --- | --- | --- | --- | --- |
| 预训练 | 1,270,238 | 1 | 8.5076 → 1.8339 | 2h 52m |
| SFT 1 轮 | 905,718 | 1 | 2.6600 → 1.6698 | 5h 24m |
| SFT 2 轮 | 905,718 | 2 | 2.6603 → 1.7262 | 10h 29m |

数值已由 [原始 CSV 和控制台日志](experiments/logs/) 核验；loss 为训练记录值，不是验证集指标。两次 SFT 均从同一个预训练权重独立开始。GPU 显存记录峰值分别为 5929、6102、6176 MB，导出显存记录的时间间隔中位数约为 61 秒。

SFT 2 轮在现有样例中回答更长，但技术准确性未见明确改善。评估未固定 seed，少量截图只能作定性分析。

## 阅读项目

- [技术报告](docs/technical_report.md)：训练过程、结果和局限。
- [使用说明](docs/usage.md)：日志核验、可选推理及路径约定。
- [实验记录](docs/experiment_records.md)：实际参数、原训练命令、指标口径。
- [个人贡献](docs/personal_contribution.md)：环境排查和简历表述参考。
- [源码复习路线](docs/review_notes.md)：模型、训练和推理流程。
- [数据来源](docs/data_sources.md)：数据规模、处理和历史版本记录。

## 文件结构

```text
code/                     模型、训练和评估的注释代码
configs/                  实际实验参数、脚本默认值、环境记录
experiments/logs/          原始 CSV 与控制台日志
experiments/summary.json   可重复生成的指标汇总
experiments/figures/       三次训练曲线
experiments/evaluations/   自训和作者模型生成截图
checkpoints/              三个自训权重，仅本地保存
docs/                     技术报告、笔记和实验记录
docs/archive/             原始报告、汇总与办公文档，仅本地归档
```

## 代码来源与运行状态

复现基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)，汇总记录 commit `4497610ec0a85d2d0a3db488fd4c1e2f12a416ab`。四个代码文件的历史修改主要为学习注释，没有模型结构或训练算法功能改动。

历史环境记录 Windows、Python 3.10.20、torch 2.11.0+cu128、transformers 4.57.6、datasets 3.6.0、SwanLab 0.7.11；三次运行的依赖快照已归档且内容一致，见 [环境记录](configs/environment_record.md)。实际参数见 [historical_runs.json](configs/historical_runs.json)，脚本默认参数单独保存，二者不混用。

数据加载模块、trainer_utils、model_lora、tokenizer、依赖清单和 Apache 2.0 许可证已补齐。三次运行元数据确认历史 commit 与训练命令。当前尚未安装运行依赖或验证模型加载，代码只完成静态语法检查；数据集未复制。文件归档状态见 [转移清单](docs/transfer_checklist.md)。

权重、数据和原始办公文档已由 `.gitignore` 排除。原始运行元数据仅本地归档；上游许可文本保留在 LICENSE 和 code/LICENSE，来源与修改说明见 NOTICE。GitHub Actions 只检查记录和源文件，不执行模型训练或推理。
