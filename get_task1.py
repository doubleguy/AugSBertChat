import os
import json
import torch
from collections import deque
from transformers import AutoConfig, AutoTokenizer, AutoModel

def main(model_path, a_model_path, input_prompt_file_path, input_raw_file, output_file):

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

    with open(input_prompt_file_path, 'r', encoding='utf-8') as fr, \
        open(input_raw_file, 'r', encoding='utf-8') as fr1, \
        open(output_file, 'w', encoding='utf-8') as fw:
        answer = deque([])
        count = 1
        for q in fr:
            print()
            query = json.loads(q)['query']
            response, _ = model.chat(tokenizer, repr(query), history=[])
            print(f'processing {count} line, response:{response}')
            try:
                response = float(response)
                response = 0.5 if response>1.0 or response<0.0 else response
            except:
                response = 0.5
            response = str(response)
            answer.append(response)
            count += 1
            
        for row in fr1:
            r = json.loads(row)
            for i, reply in enumerate(r['replys']):
                tmp = answer.popleft()
                if i==len(r['replys'])-1:
                    fw.write(tmp)
                else:
                    fw.write(tmp+'\t')
            fw.write('\n')

if __name__=='__main__':
    model_path = "/home/zb/ChatGLM-6B-main/chatglm-6b-int4"
    a_model_path = "/home/zb/ChatGLM-6B-main/ptuning/output/task11-chatglm-6b-pt-128-2e-2/checkpoint-1000"
    input_prompt_file_path = 'datasets/processed/qa_test_task1_glm.json'
    input_raw_file = 'datasets/raw/datasets_test_track1.jsonl'
    output_file = 'datasets/res/task11_glm_1000.txt'
    
    # a_model_path = "/home/zb/ChatGLM-6B-main/ptuning/output/task11-chatglm-6b-pt-128-2e-2/checkpoint-3000"
    # input_prompt_file_path = 'datasets/processed/qa_dev_task1_glm.json'
    # input_raw_file = 'datasets/raw/datasets_dev.jsonl'
    # output_file = 'datasets/res/task111_glm.txt'
    main(model_path, a_model_path, input_prompt_file_path, input_raw_file, output_file)