import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util

prompt_start = "假设你是小爱同学，请为下面这句话生成一条回复\n"   
query = "你的声音好好听啊"
prompt_mid = "\n回复："
input = prompt_start + query + prompt_mid

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


bert_model_path = "task1_sbert/output/bert-2023-05-31_02-45-44"
bert_model = SentenceTransformer(model_name_or_path=bert_model_path)
best_response = None
best_score = 0
for _ in range(3):
    response, history = model.chat(tokenizer, input, history=[],  top_p=2)
    # Compute embedding for both query and reply
    query_embedding = bert_model.encode(query, convert_to_tensor=True)
    reply_embedding = bert_model.encode(response, convert_to_tensor=True)
    # Compute cosine-similarities
    cosine_scores = util.cos_sim(query_embedding, reply_embedding)
    if cosine_scores>best_score:
        best_score = cosine_scores
        best_response = response
    print(response, cosine_scores, best_score)
    print(best_response)

# 与微调前进行对比
# model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
# model = model.eval()

# response, history = model.chat(tokenizer, input, history=[])
# print(response)