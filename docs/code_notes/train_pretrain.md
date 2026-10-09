# MiniMind 预训练流程笔记：train_pretrain.py

## 1. 数据准备

结构：

```text
预训练数据 jsonl
↓
PretrainDataset
↓
tokenizer 编码
↓
input_ids / labels
```

说明：

> 预训练数据首先从 `pretrain_t2t_mini.jsonl` 中读取，然后通过 tokenizer 转换成 token id。
> `input_ids` 是模型的输入，`labels` 是模型要预测的目标。

---

## 2. 模型输入

结构：

```text
input_ids
labels
```

说明：

> `input_ids` 会送入 MiniMind 模型进行前向传播。
> `labels` 不直接作为模型输入参与计算，而是用来和模型输出的 logits 计算 loss。

---

## 3. 模型前向传播

结构：

```text
MiniMindForCausalLM
├── Embedding
├── MiniMindBlock × N
├── RMSNorm
└── lm_head
```

说明：

> 模型先把 token id 转成向量，然后经过多层 Transformer Block，最后通过 `lm_head` 映射到词表大小，得到每个位置对下一个 token 的预测分数。

---

## 4. 输出 logits

结构：

```text
hidden_states
↓
lm_head
↓
logits
```

说明：

> `logits` 是模型对词表中每个 token 的预测分数。
> 它的形状通常是 `[batch, seq_len, vocab_size]`。

---

## 5. 计算损失

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

> 语言模型训练的目标是“用当前位置预测下一个 token”。
> 所以 logits 和 labels 要错开一位：第 0 个位置的输出预测第 1 个 token，第 1 个位置的输出预测第 2 个 token。

---

## 6. MoE 辅助损失

结构：

```text
如果开启 MoE
↓
计算 aux_loss
```

说明：

> 如果模型使用 MoE 结构，还会额外计算 `aux_loss`。
> 它主要用于让不同 expert 的负载更均衡，避免所有 token 都集中到同一个 expert。

---

## 7. 总损失

结构：

```text
total_loss = logits_loss + aux_loss
```

说明：

> 训练时真正用于反向传播的是总损失。
> 如果没有开启 MoE，`aux_loss` 通常为 0，此时总损失基本等于 `logits_loss`。

---

## 8. 梯度累积

结构：

```text
total_loss / accumulation_steps
↓
backward
↓
累计梯度
```

说明：

> 由于显存有限，不能一次使用很大的 batch。
> 梯度累积的作用是先用多个小 batch 累积梯度，再统一更新一次模型参数，相当于模拟更大的 batch size。

---

## 9. 反向传播

结构：

```text
loss.backward()
```

说明：

> 反向传播会根据 loss 计算每个参数的梯度。
> 梯度表示当前参数应该往哪个方向调整，才能让 loss 降低。

---

## 10. 梯度裁剪

结构：

```text
clip_grad_norm_
```

说明：

> 梯度裁剪用于防止梯度过大。
> 如果梯度太大，训练可能会不稳定，甚至出现 loss 爆炸。

---

## 11. 优化器更新

结构：

```text
optimizer.step()
```

说明：

> 优化器根据梯度更新模型参数。
> 这份源码中使用的是 `AdamW`，它是训练 Transformer 和大语言模型常用的优化器。

---

## 12. 混合精度训练

结构：

```text
autocast
GradScaler
```

说明：

> 混合精度训练可以减少显存占用，并提高训练速度。
> 源码默认使用 `bfloat16`，如果使用 `float16`，还会启用 `GradScaler` 来避免梯度数值过小。

---

## 13. 清空梯度

结构：

```text
optimizer.zero_grad()
```

说明：

> 每次参数更新后，需要清空旧梯度。
> 否则下一轮训练的梯度会和上一轮残留的梯度混在一起。

---

## 14. 日志输出

结构：

```text
每隔 log_interval 输出：
├── loss
├── logits_loss
├── aux_loss
├── learning_rate
└── epoch_time
```

说明：

> 日志用于观察训练过程是否正常。
> 如果 loss 整体下降，说明模型正在学习；如果 loss 剧烈震荡或变成 NaN，说明训练可能有问题。

---

## 15. 保存模型

结构：

```text
每隔 save_interval 保存：
├── out/pretrain_xxx.pth
└── checkpoint 文件
```

说明：

> `.pth` 文件主要用于推理或后续训练。
> checkpoint 文件用于断点续训，里面不仅包含模型参数，还包含优化器状态、scaler 状态、epoch 和 step。

---

## 16. Epoch 循环

结构：

```text
for epoch in epochs:
    遍历全部训练数据
    完成一个 epoch
    进入下一个 epoch
```

说明：

> 一个 epoch 表示模型完整看完一遍训练数据。
> 多个 epoch 表示模型会重复学习多轮数据。

---

## 17. 预训练总体流程总结

```text
预训练数据
↓
tokenizer 编码
↓
input_ids / labels
↓
MiniMindForCausalLM
↓
logits
↓
和 labels 计算 next-token loss
↓
加上 aux_loss
↓
梯度累积
↓
反向传播
↓
梯度裁剪
↓
AdamW 更新参数
↓
清空梯度
↓
日志记录
↓
保存权重和 checkpoint
↓
进入下一个 batch / epoch
```

一句话总结：

> `train_pretrain.py` 的核心作用是让 MiniMind 通过大量文本数据学习“根据前文预测下一个 token”的能力。训练过程中，模型根据 `input_ids` 输出 `logits`，再和右移后的 `labels` 计算语言模型损失，通过反向传播和 AdamW 不断更新参数，最终保存为 `pretrain_xxx.pth` 权重。
