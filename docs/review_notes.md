# MiniMind 代码复习路线

沿着数据进入模型、计算损失、更新参数、保存权重和生成文本的顺序回顾，结合已有训练结果解释代码的作用。

## 模型结构

阅读 `code/model/model_minimind.py` 与 `code_notes/model_minimind.md`。画出 Embedding、Transformer Block、RMSNorm 和 lm_head 的数据流，核对 GQA 的 Q 与 KV 头数、RoPE 的相对位置作用和 SwiGLU 两条支路。区分位置编码允许的长度与模型实际训练过的长度。

## 预训练

阅读 `code/trainer/train_pretrain.py` 与原预训练笔记。解释 input_ids 与 labels、错位预测、交叉熵、梯度累积、梯度裁剪与 AdamW。注意保存脚本中的累积步数是默认值；多卡情况下有效 batch 还与进程数有关。数据加载模块恢复后再核对具体标签构造。

## 全参数微调

阅读 `code/trainer/train_full_sft.py` 与原 SFT 笔记。比较预训练和 SFT 的权重初始化、数据格式、学习率及标签屏蔽。`model.parameters()` 交给优化器表示全参数训练；支持 LoRA 的推理入口并不能证明本次做过 LoRA 训练。

## 推理与比较

阅读 `code/eval_llm.py` 和原评估笔记。解释模型结构与权重匹配、chat template、temperature、top_p 和自回归生成。原脚本每个问题使用随机 seed，旧截图只作定性观察。后续统一问题、seed、采样参数、最大生成长度和 tokenizer，再比较三个模型。

## 实验分析

结合 `experiments/figures` 与 `experiments/evaluations`，区分训练 loss、回答相关性、知识准确性、重复与异常符号。增加 epoch 可能改善格式，但现有样例不能证明普遍准确性提升；训练曲线也不能单独证明泛化能力。
