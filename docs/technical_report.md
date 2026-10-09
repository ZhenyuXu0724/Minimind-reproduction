# MiniMind 复现技术报告

本次实践在单卡笔记本 GPU 上完成 MiniMind 预训练和全参数 SFT，并比较微调轮数对生成样例的影响。预训练 loss 明显下降；SFT 模型学到了基本对话形式，但技术问题仍有知识错误，增加轮数后未见明确的准确性改善。

## 模型与数据处理

模型约 63.91M 参数，隐藏维度 768、8 层、8 个 Q 头和 4 个 KV 头，词表大小 6400。结构为 Embedding → MiniMindBlock × 8 → RMSNorm → lm_head，使用 RoPE 位置编码、GQA 注意力和 SwiGLU FFN，本次未开启 MoE。实现见 [模型源码](../code/model/model_minimind.py)。

RMSNorm 按均方根归一化隐藏状态；RoPE 对 Q、K 做位置相关旋转，使注意力体现相对位置；GQA 让多个 Q 头共享 KV 头；SwiGLU 通过门控支路调节 FFN 输出。这些结构均来自上游模型。

数据采用官方 mini 版，无人工筛选：

| 文件 | 字节数 | 样本数 |
| --- | --- | --- |
| pretrain_t2t_mini.jsonl | 1,241,043,656 | 1,270,238 |
| sft_t2t_mini.jsonl | 1,739,201,170 | 905,718 |

数据获取：[Hugging Face 镜像](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)、[ModelScope 官方数据集](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)。历史下载于 2026 年 5 月 27 日，具体下载渠道、revision 和 SHA256 未保留；当前链接不保证与历史文件完全一致。

[数据处理源码](../code/dataset/lm_dataset.py)在预训练时将文本截断到 `max_seq_len - 2`，添加 BOS/EOS、补 PAD，并将 PAD 标签设为 -100。SFT 使用 chat template 编码，截断和补齐到定长，仅 assistant 回答及结束标记参与损失；还包含概率性添加 system 提示和移除空思考标签的处理。训练与评估采用同一份仓库 tokenizer。

模型前向输出 logits，通过错位标签预测下一个 token，计算交叉熵；SFT 从预训练权重初始化，用 AdamW 更新全部模型参数。训练循环采用混合精度、梯度累积、梯度裁剪和余弦学习率调度。脚本另保存优化器等续训状态，本仓库仅本地保留三份推理权重。

## 实验设置

历史环境为 Windows、RTX 5070 Laptop GPU 8 GB、驱动 573.22、CUDA 12.8、Python 3.10.20、torch 2.11.0+cu128、transformers 4.57.6、datasets 3.6.0、numpy 1.26.4 和 SwanLab 0.7.11。三次运行的依赖快照一致，合并保存在 [requirements.freeze.txt](../configs/requirements.freeze.txt)。

共同配置为 bfloat16、AdamW、grad_clip 1.0、num_workers 0、log_interval 10 和 save_interval 200。实际配置另存于 [historical_runs.json](../configs/historical_runs.json)。

| 参数 | Pretrain | SFT 1 轮 | SFT 2 轮 |
| --- | --- | --- | --- |
| epochs | 1 | 1 | 2 |
| batch_size | 16 | 8 | 8 |
| accumulation_steps | 8 | 1 | 1 |
| 单卡等效 batch | 128 | 8 | 8 |
| learning_rate | 5e-4 → 5e-5 | 1e-5 → 1e-6 | 1e-5 → 1e-6 |
| max_seq_len | 340 | 768 | 768 |
| DataLoader 总步数 | 79,390 | 113,215 | 226,430 |

两次 SFT 均从同一个预训练权重独立启动，2 轮实验不是在 1 轮成品上续训。初始种子为 42，第二轮使用 `42 + epoch = 43`。梯度累积使 DataLoader 步数与优化器更新次数不同。

原运行命令如下，工作目录为 `code/trainer/`：

```powershell
python train_pretrain.py --epochs 1 --batch_size 16 --num_workers 0 --log_interval 10 --save_interval 200 --data_path ../dataset/pretrain_t2t_mini.jsonl --save_weight pretrain --use_wandb --wandb_project MiniMind-Pretrain-Local
python train_full_sft.py --epochs 1 --batch_size 8 --num_workers 0 --log_interval 10 --save_interval 200 --data_path ../dataset/sft_t2t_mini.jsonl --from_weight pretrain --save_weight full_sft --use_wandb --wandb_project MiniMind-SFT-Local
python train_full_sft.py --epochs 2 --batch_size 8 --num_workers 0 --log_interval 10 --save_interval 200 --data_path ../dataset/sft_t2t_mini.jsonl --from_weight pretrain --save_weight full_sft_e2 --use_wandb --wandb_project MiniMind-SFT-Compare
```

`--use_wandb` 在源码中通过 `import swanlab as wandb` 记录到 SwanLab。

## 训练结果

| 指标 | Pretrain | SFT 1 轮 | SFT 2 轮 |
| --- | --- | --- | --- |
| 首次记录 loss | 8.5076 | 2.6600 | 2.6603 |
| 最后记录 loss | 1.8339 | 1.6698 | 1.7262 |
| 最小记录 loss | 1.5229 | 0.8820 | 0.8542 |
| GPU 显存记录峰值 MB | 5,929 | 6,102 | 6,176 |
| 耗时 | 2h 52m 11s | 5h 23m 35s | 10h 28m 31s |

loss、学习率、aux_loss、显存峰值和最后训练进度已用 [CSV 与控制台日志](../experiments/logs/)核验，统计及文件校验值见 [summary.json](../experiments/summary.json)。耗时依据运行起止时间汇总，三次累计 18h 44m 17s。

预训练 loss 从随机初始化阶段迅速下降，SFT 曲线保持波动，aux_loss 全程为 0。表中的 loss 是训练日志记录值，不是验证集均值；最小 batch loss 不代表最佳模型质量。显存指标取自 `__swanlab__.gpu.0.mem.value`，导出记录间隔中位数约 61 秒，可能遗漏瞬时峰值；不能将其直接当作 PyTorch allocated 或 reserved memory。

![预训练曲线](../experiments/figures/pretrain可视化.png)

![SFT 1 轮曲线](<../experiments/figures/sft epoch=1可视化.png>)

![SFT 2 轮曲线](<../experiments/figures/sft epoch=2可视化.png>)

## 生成样例分析

评估使用 temperature 0.85、top_p 0.95、max_new_tokens 8192、do_sample True、repetition_penalty 1、historys 0 和 open_thinking 0；脚本为每个问题重新生成随机 seed。作者模型来自 ModelScope `gongjy/minimind-3`，历史下载 revision 未保留。

| 模型 | 样例观察 |
| --- | --- |
| SFT 1 轮 | 能形成基本中文回答，但有重复、异常符号，解释 Transformer 时明显跑题 |
| SFT 2 轮 | 回答更长、格式更像助手，但仍有技术幻觉和无关代码 |
| 作者模型 | 在所示问题中结构较清晰、相关性较高，技术细节仍不够深入 |

完整截图见 [SFT 1 轮](<../experiments/evaluations/sft epoch=1/>)、[SFT 2 轮](<../experiments/evaluations/sft epoch=2/>) 和 [作者模型](../experiments/evaluations/作者权重/)。

现有样例说明训练 loss 与回答准确性并不等价。由于问题数量有限且未固定 seed，不能据此证明普遍性能排名、过拟合或能力全面提升；数据规模、训练充分程度和生成设置等因素也没有被单独控制。

## 结论与局限

本次复现打通了数据处理、预训练、全参数 SFT 和生成分析流程，模型具备基本对话形式，但知识准确性仍有限。代码与日志静态核验通过，整理过程中未重新安装训练环境或运行模型；数据本体与权重未公开，历史数据及作者权重版本也未完整固定。

后续若评估模型，可统一问题、seed 和采样设置，并分别记录相关性、准确性、重复和异常符号。LoRA、DPO、PPO、GRPO、MoE 与多模态训练不属于本次已完成内容。
