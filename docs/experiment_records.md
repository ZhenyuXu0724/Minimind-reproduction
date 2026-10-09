# MiniMind 历史实验记录

参数来自《复现材料汇总》；三次原始运行元数据已归档，命令、Python 版本与 commit 均与汇总一致。2026 年 10 月 9 日已收到三个运行的 CSV 和控制台日志，核对 loss 首值、末值、最小值、学习率、aux_loss、显存记录峰值和最终训练步数，均与汇总数值一致。原文件在 [experiments/logs](../experiments/logs/)，逐项统计和文件 SHA256 在 [summary.json](../experiments/summary.json)。

## 运行配置与结果

三次正式训练使用单卡 RTX 5070 Laptop GPU 8 GB，模型约 63.91M 参数。共同配置为 hidden_size 768、8 层、8 Q 头和 4 KV 头、词表 6400、关闭 MoE、bfloat16、AdamW 和 grad_clip 1.0。

| 指标 | Pretrain | SFT 1 轮 | SFT 2 轮 |
| --- | --- | --- | --- |
| 数据条数 | 1,270,238 | 905,718 | 905,718 |
| epochs | 1 | 1 | 2 |
| batch_size | 16 | 8 | 8 |
| accumulation_steps | 8 | 1 | 1 |
| 单卡等效 batch | 128 | 8 | 8 |
| learning_rate | 5e-4 | 1e-5 | 1e-5 |
| max_seq_len | 340 | 768 | 768 |
| DataLoader 总步数 | 79,390 | 113,215 | 226,430 |
| loss 首个记录值 | 8.5076 | 2.6600 | 2.6603 |
| loss 最后记录值 | 1.8339 | 1.6698 | 1.7262 |
| loss 最小记录值 | 1.5229 | 0.8820 | 0.8542 |
| GPU 显存记录峰值 MB | 5,929 | 6,102 | 6,176 |
| 运行耗时 | 2h 52m 11s | 5h 23m 35s | 10h 28m 31s |

累计耗时 18h 44m 17s，按汇总给出的开始与结束时间计算。显存峰值取自导出键 `__swanlab__.gpu.0.mem.value`，导出记录间隔中位数分别为 61.048、61.049、61.050 秒，可能遗漏瞬时峰值；该键不能证明是 PyTorch tensor allocated 或 reserved memory，本文按 GPU 显存监控值记录。表中 loss 是训练记录值，不是验证集均值或测试集指标；最小单批次 loss 不代表最佳模型质量。梯度累积使预训练 DataLoader 步数与优化器更新次数不同。

两次 SFT 都从同一个 pretrain 权重独立启动。SFT 2 轮不是在 SFT 1 轮成品上继续训练。初始种子均为 42；保存脚本在每个 epoch 使用 `42 + epoch`，因此第二轮种子为 43。

## 历史训练命令

历史工作目录是原工程的 `trainer/`。以下命令用于保存实验过程，本轮未执行。

```powershell
python train_pretrain.py --epochs 1 --batch_size 16 --num_workers 0 --log_interval 10 --save_interval 200 --data_path ../dataset/pretrain_t2t_mini.jsonl --save_weight pretrain --use_wandb --wandb_project MiniMind-Pretrain-Local
python train_full_sft.py --epochs 1 --batch_size 8 --num_workers 0 --log_interval 10 --save_interval 200 --data_path ../dataset/sft_t2t_mini.jsonl --from_weight pretrain --save_weight full_sft --use_wandb --wandb_project MiniMind-SFT-Local
python train_full_sft.py --epochs 2 --batch_size 8 --num_workers 0 --log_interval 10 --save_interval 200 --data_path ../dataset/sft_t2t_mini.jsonl --from_weight pretrain --save_weight full_sft_e2 --use_wandb --wandb_project MiniMind-SFT-Compare
```

`--use_wandb` 实际启用 `import swanlab as wandb`，日志写入 SwanLab。脚本默认值另存于 `../configs/script_defaults.json`，实际参数见 `../configs/historical_runs.json`。

## 历史评估

汇总记录默认采样参数 temperature 0.85、top_p 0.95、max_new_tokens 8192、do_sample True、repetition_penalty 1、单轮对话。每个问题重新随机 seed，历史输出不能逐字重现。作者模型命令有 PowerShell 历史来源，自训模型命令在汇总中明确标为按默认参数推断，不能当作原始命令证据。

作者模型来自 ModelScope `gongjy/minimind-3`。在所示问题中，SFT 2 轮回答更长但仍有技术幻觉，作者模型回答较相关；少量截图和未固定 seed 的比较不能支持普遍性能排名。

## 指标核验方法

运行 `python tools/summarize_exports.py` 可从归档文件重新生成汇总，不加载模型。三个运行分别包含 7939、11322、22644 条 loss 记录；CSV 的 step 是日志索引，不可直接当作训练 batch 次数。控制台最后进度分别为 1/1 轮的 79390/79390、1/1 轮的 113215/113215、2/2 轮的 113215/113215。

采样间隔按导出时间戳计算，不能据此推断底层监控的配置间隔。运行开始结束时间仍按材料汇总记录，不能把指标首末时间误当作完整运行耗时。
