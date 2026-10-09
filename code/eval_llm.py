import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer#用了Hugging Face的Transformer库
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM#MiniMindConfig是模型的配置类，MiniMindForCausalLM是模型本身，CausalLLM表示因果语言模型，即根据前一个token生成后一个token
from model.model_lora import *#导入LoRA相关代码
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):#接收一个初始的arg参数
    #用AutoTokenizer来加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    #初始化模型
    if 'model' in args.load_from:#判断加载本地模型还是Transformer模型
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,#内部向量维度
            num_hidden_layers=args.num_hidden_layers,#隐藏层数
            use_moe=bool(args.use_moe),#是否使用MoE
            inference_rope_scaling=args.inference_rope_scaling#位置编码有关，拓展上下文长度
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        #pth文件地址
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        #加载模型权重
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)#strict=true表示模型结构与权重文件完全匹配
        if args.lora_weight != 'None':
            apply_lora(model)#加上LoRA结构
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')#加载LoRA权重
    else:
        #加载Transformer格式模型
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)#打印模型参数量
    #切换为eval模型，训练是会启动dropout等机制，但是推理时不需要
    return model.half().eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()

    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]

    conversation = []
    #获取模型和分词器
    model, tokenizer = init_model(args)
    #选择输入模式
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    #文本流式生成
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    #构造prompt迭代器
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        #设置种子保证可复现，让随机变得可控，输出更加稳定
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        #添加历史对话
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:#区分是预训练得到的权重还是以后的模型，预训练没有学习用户-助手的对话模式，只需给出普通文本
            inputs = tokenizer.bos_token + prompt
        else:#对于以后的模型，需要告诉他前面是用户说的话，现在轮到助手回答了
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))#是否加上思考模式

        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)#tokenizer编码
        print('🧠: ', end='')
        st = time.time()
        generated_ids = model.generate(#调用model.generate进行回答
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],#掩码告诉模型哪些位置是有效的
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)#解码模型回答
        conversation.append({"role": "assistant", "content": response})#将助手的回答加入历史
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()
