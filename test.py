import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

prompt_start = "指令：请根据问答对给答案打分，打分区间为[0,1]\n[示例1]\n问句：你想要红包吗\n回复：谢谢你，小爱并不需要钱， \
希望你可以多点时间陪小爱~\n答案：1.0\n[示例2]\n问句：小爱你能看见我吗\n 回复：别着急啊，你的小宝贝正在思考怎么回复你。\n \
答案：0.6\n[示例3]\n问句：我唱歌好听吗\n回复：能让你喜欢的应该是个很棒的人吧~\n答案：0.0\n请根据上面的示例给下面的问答对中的答案打分\n问句："   
query = "你去洗吧"
prompt_mid = "\n回复："
reply = "你要洗什么啊"
prompt_end = "\n答案："
input = prompt_start + query + prompt_mid + reply + prompt_end

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
#print(f"Quantized to 4 bit")
# model = model.quantize(4)
# model = model.half().cuda()
# model.transformer.prefix_encoder.float()
# model = model.eval()

# response, history = model.chat(tokenizer, input, history=[])
# print(response)

# 与微调前进行对比
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
model = model.eval()

response, history = model.chat(tokenizer, input, history=[])
print(response)