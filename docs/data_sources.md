# MiniMind 数据与代码来源

复现材料汇总记录上游代码提交 `4497610ec0a85d2d0a3db488fd4c1e2f12a416ab`，来自 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)。三个运行的原始元数据均记录该 commit。许可证、配套模块和 tokenizer 已收到并归档；当前仓库是文件整理副本，未检出上游完整历史。

用户说明数据来自 Hugging Face mini 版；补充汇总同时列出 [ModelScope 官方数据集](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 与 [Hugging Face 镜像](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)。历史下载命令未记录，因此不将实际下载渠道写成已核实。

| 文件 | 历史大小 B | 历史条数 | 实际使用 |
| --- | --- | --- | --- |
| pretrain_t2t_mini.jsonl | 1,241,043,656 | 1,270,238 | 全量，1 epoch |
| sft_t2t_mini.jsonl | 1,739,201,170 | 905,718 | 全量，SFT 各 1 和 2 epoch |

汇总记录无人工筛选，下载时间为 2026 年 5 月 27 日。历史数据 revision 与 SHA256 未提供，文件名、字节数和行数仍不能唯一确定内容版本。当前未保存数据本体。

预处理记录为：预训练截断到 max_seq_len 减 2，加 BOS 和 EOS，再补 PAD；PAD 标签为 -100。SFT 用 chat template，截断到 max_seq_len，仅 assistant 段参与 loss，另含随机 system 提示和空思考标签处理。具体实现等待原 `lm_dataset.py` 核对。

tokenizer 记录来自原仓库，词表 6400，tokenizer.json 为 482,372 B，训练和评估使用同一份；文件已放入 `code/model/` 并通过 JSON 格式检查。

作者模型记录来自 ModelScope `gongjy/minimind-3`，模型约 64M 参数，下载于 2026 年 5 月 24 日。下载 revision 和文件校验值未提供。模型来源已知，内容版本仍待固定。
