import os
import json
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

def main():

    model_path = "/home/zb/ChatGLM-6B-main/chatglm-6b-int4"
    a_model_path = "/home/zb/ChatGLM-6B-main/ptuning/output/task2-chatglm-6b-pt-128-2e-2/checkpoint-3000"

    tokenizer = AutoTokenizer.from_pretrained(a_model_path, trust_remote_code=True)

    # Fine-tuning 后的表现测试
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
    model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
    # 此处使用你的 ptuning 工作目录
    prefix_state_dict = torch.load(os.path.join(a_model_path, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    #V100 机型上可以不进行量化
    print(f"Quantized to 4 bit")
    model = model.quantize(4)
    model = model.half().cuda()
    model.transformer.prefix_encoder.float()
    model = model.eval()

    with open('datasets/processed/qa_test2_task2.json', 'r', encoding='utf-8') as fr, open('datasets/res/task2.txt', 'w', encoding='utf-8') as fw:
        for q in fr:
            query = json.loads(q)['query']
            response, _ = model.chat(tokenizer, repr(query), history=[])
            fw.write(response)
            fw.write('\n')

if __name__=='__main__':
    main()