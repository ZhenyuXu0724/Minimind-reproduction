1.作用:
推理脚本，用于加载已经训练好的模型权重，根据用户的输入生成回答
加载模型 → 构造输入 → 调用 generate → 输出回答
2.整体运行流程：
  1. 读取命令行参数
  2. 加载 tokenizer
  3. 根据参数创建 MiniMind 模型结构
  4. 从 out 目录加载 .pth 权重
  5. 如果指定 LoRA，则加载 LoRA 权重
  6. 切换到 eval 模式
  7. 选择自动测试或手动输入
  8. 把用户问题转换成模型输入
  9. 调用 model.generate() 生成回答
  10. 解码输出并计算生成速度
3.重要的参数
| 参数                   | 作用                | 我的理解
| `--weight`            | 指定权重类型          | 决定加载 pretrain/full_sft 等权重
| `--hidden_size`       | 隐藏层维度           | 必须和训练时一致
| `--num_hidden_layers` | Transformer 层数    | 必须和训练时一致
| `--use_moe`           | 是否使用 MoE         | 必须和训练时一致
| `--temperature`       | 控制生成随机性        | 越高越随机
| `--top_p`             | 控制采样范围          | 常和 temperature 一起使用
| `--historys`          | 是否保留历史对话       | 影响多轮对话能力
| `--max_new_tokens`    | 最大生成长度          | 太大可能生成很久
4.关键机制
  1.tokenizer
  模型不能直接处理文字，必须先将文本转换为 token id。
  流程：
  2.模型结构与权重
  模型结构由 MiniMindConfig 创建，模型参数由 .pth 文件加载。二者必须一致
  3.pretrain 和 chat 模型的区别
  SFT 模型需要对话模板，而 pretrain 模型更像文本续写模型
  4.generate生成机制
  model.generate() 会逐 token 生成文本。
  每一步流程：
  已有 token → 预测下一个 token 概率 → 采样 → 拼接 → 继续生成
  5.随机种子
  每次随机设置 seed，因此同一问题可能有不同回答。
  如果要公平比较模型，获得可复现的回答，应改成固定 seed。
