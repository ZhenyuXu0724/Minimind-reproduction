# MiniMind 全参数监督微调流程笔记：train_full_sft.py

## 1. SFT 数据准备

结构：

```text
sft_t2t_mini.jsonl
↓
SFTDataset
↓
构造对话样本
↓
tokenizer 编码
↓
input_ids / labels
```

说明：

> SFT 使用的是指令 / 对话数据，而不是普通预训练文本。
> `SFTDataset` 会读取 `sft_t2t_mini.jsonl`，将用户问题和助手回答整理成模型可以学习的格式。
> `input_ids` 是模型输入，`labels` 是训练目标。

---

## 2. 构造训练标签

结构：

```text
用户输入 + 助手回答
↓
input_ids
↓
labels
```

说明：

> SFT 的重点是让模型学习 assistant 的回答。
> 通常用户问题部分不会参与 loss 计算，可能会被标记为 `-100`。
> 这样模型不会被训练去“预测用户问题”，而是重点学习“如何回答用户”。

---

## 3. 模型与初始化

结构：

```text
MiniMindConfig
↓
init_model()
↓
MiniMindForCausalLM + tokenizer
↓
加载 pretrain 权重
```

说明：

> `MiniMindConfig` 用来设置模型结构，例如 `hidden_size`、`num_hidden_layers`、`use_moe`。
> `init_model()` 会创建模型和 tokenizer。
> Full SFT 默认从 `pretrain` 权重开始，而不是从随机初始化开始。

---

## 4. 加载预训练权重

结构：

```text
from_weight = pretrain
↓
加载预训练模型参数
↓
继续进行 SFT
```

说明：

> 预训练模型已经具备基础语言能力。
> SFT 的作用是在这个基础上进一步训练模型的指令跟随和对话回答能力。
> 如果不加载预训练权重，直接做 SFT，模型效果通常会很差。

---

## 5. 模型前向传播

结构：

```text
input_ids
↓
MiniMindForCausalLM
├── Embedding
├── MiniMindBlock × N
├── RMSNorm
└── lm_head
↓
logits
```

说明：

> `input_ids` 输入模型后，先经过 Embedding 转成向量，再经过多层 Transformer Block。
> 最后通过 `lm_head` 映射到词表大小，得到 `logits`。
> `logits` 表示模型对下一个 token 的预测分数。

---

## 6. 输出 logits

结构：

```text
hidden_states
↓
lm_head
↓
logits
```

说明：

> `logits` 的形状通常是 `[batch, seq_len, vocab_size]`。
> 它表示每个位置对词表中所有 token 的预测分数。
> 后续 loss 会根据这些预测分数和 `labels` 计算出来。

---

## 7. 计算语言模型损失

结构：

```text
logits 向左错位
labels 向右错位
↓
Cross Entropy Loss
↓
logits_loss
```

说明：

> 语言模型训练的本质仍然是预测下一个 token。
> 第 0 个位置的输出预测第 1 个 token，第 1 个位置的输出预测第 2 个 token。
> 对于 label 为 `-100` 的位置，loss 会自动忽略。

---

## 8. MoE 辅助损失

结构：

```text
如果 use_moe = 1
↓
计算 aux_loss
```

说明：

> 如果启用了 MoE，模型还会计算 `aux_loss`。
> `aux_loss` 不是主要语言建模损失，而是用于让不同 expert 的负载更均衡。
> 如果没有开启 MoE，`aux_loss` 通常为 0。

---

## 9. 总损失

结构：

```text
total_loss = logits_loss + aux_loss
```

说明：

> 训练时真正用于反向传播的是总损失。
> 普通 Full SFT 中，如果没有启用 MoE，可以近似理解为：

```text
total_loss ≈ logits_loss
```

---

## 10. 训练目标

结构：

```text
最小化 total_loss
↓
让模型学习：
用户指令 → 助手回答
```

说明：

> Full SFT 的目标不是单纯续写文本，而是让模型学会按照用户指令进行回答。
> 预训练让模型“会说话”，SFT 让模型“更会按要求说话”。

---

## 11. 混合精度训练

结构：

```text
autocast
↓
bfloat16 / float16
↓
GradScaler
```

说明：

> 混合精度可以减少显存占用，并提升训练速度。
> 源码默认使用 `bfloat16`。
> 如果使用 `float16`，会启用 `GradScaler`，防止梯度数值过小导致训练不稳定。

---

## 12. 梯度累积

结构：

```text
total_loss / accumulation_steps
↓
backward()
↓
累计梯度
```

说明：

> 梯度累积可以用多个小 batch 模拟一个大 batch。
> Full SFT 默认 `accumulation_steps = 1`，也就是每个 batch 都更新一次。
> 显存不足时可以减小 `batch_size`，同时增大 `accumulation_steps`。

---

## 13. 反向传播

结构：

```text
scaler.scale(loss).backward()
```

说明：

> 反向传播会根据 loss 计算模型中每个参数的梯度。
> 梯度表示参数应该如何调整，才能让 loss 下降。

---

## 14. 梯度裁剪

结构：

```text
clip_grad_norm_
```

说明：

> 梯度裁剪用于限制梯度大小。
> 如果梯度过大，训练可能会震荡甚至 loss 爆炸。
> 源码默认裁剪阈值是 `grad_clip = 1.0`。

---

## 15. 优化器更新

结构：

```text
optimizer.step()
↓
AdamW 更新参数
```

说明：

> 优化器根据梯度更新模型参数。
> 源码使用的是 `AdamW`。
> 因为优化器传入的是 `model.parameters()`，所以这是全参数微调，模型所有参数都会被更新。

---

## 16. 清空梯度

结构：

```text
optimizer.zero_grad()
```

说明：

> 每次参数更新之后，需要清空旧梯度。
> 否则下一轮的梯度会和上一轮残留的梯度叠加，影响训练结果。

---

## 17. 日志记录

结构：

```text
每隔 log_interval 记录：
├── loss
├── logits_loss
├── aux_loss
├── learning_rate
└── epoch_time
```

说明：

> 日志用于观察训练是否正常。
> `loss` 下降通常说明模型正在学习。
> `learning_rate` 用来观察学习率调度是否正常。
> `epoch_time` 用来估计当前 epoch 剩余训练时间。

---

## 18. 保存模型

结构：

```text
每隔 save_interval 保存：
├── out/full_sft_768.pth
└── checkpoints/
```

说明：

> `full_sft_768.pth` 是最终用于推理的模型权重。
> `checkpoints/` 中保存的是断点续训状态，包括模型、优化器、scaler、epoch、step 等信息。
> 如果训练中断，可以通过 `from_resume=1` 继续训练。

---

## 19. Epoch 循环

结构：

```text
for epoch in range(start_epoch, args.epochs):
    构造 DataLoader
    调用 train_epoch()
    训练一个完整 epoch
```

说明：

> 一个 epoch 表示模型完整遍历一遍 SFT 训练数据。
> 多个 epoch 表示模型会重复学习多轮指令数据。

---

## 20. 训练结束

结构：

```text
训练完成
↓
保存最终权重
↓
清理分布式进程
```

说明：

> 训练结束后，如果使用了分布式训练，会调用 `dist.destroy_process_group()` 清理进程。
> 最终得到的 `full_sft_768.pth` 可以在推理脚本中通过 `--weight full_sft` 加载。

---

## 21. Full SFT 总体流程总结

```text
SFT 对话数据
↓
SFTDataset 构造样本
↓
tokenizer 编码
↓
input_ids / labels
↓
加载 pretrain 权重
↓
MiniMindForCausalLM 前向传播
↓
得到 logits
↓
和 labels 计算 logits_loss
↓
加上可选 aux_loss
↓
total_loss
↓
反向传播
↓
梯度裁剪
↓
AdamW 更新全部模型参数
↓
清空梯度
↓
日志记录
↓
保存 full_sft 权重和 checkpoint
↓
进入下一个 batch / epoch
```

一句话总结：

> `train_full_sft.py` 的核心作用是：在预训练模型的基础上，使用 SFT 指令 / 对话数据对 MiniMind 进行全参数微调。它通过 `SFTDataset` 构造 `input_ids` 和 `labels`，让模型学习“用户指令 → 助手回答”的映射关系，并通过 AdamW 更新模型全部参数，最终保存为 `full_sft_768.pth`。
