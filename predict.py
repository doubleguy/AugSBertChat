from model import BertForTextClassfication
import torch
from transformers import BertTokenizer
from config import Config
import time
import logging


def load_model(device, model_path):
    myModel = BertForTextClassfication().to(device)
    myModel.load_state_dict(torch.load(model_path))
    myModel.eval()
    return myModel


def process_text(text, bert_pred, device, max_len=40):
    tokenizer = BertTokenizer.from_pretrained(bert_pred)
    token_id = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenizer.tokenize(text))
    mask = [1] * len(token_id) + [0] * (max_len + 2 - len(token_id))
    token_ids = token_id + [0] * (max_len + 2 - len(token_id))
    token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
    mask = torch.tensor(mask).unsqueeze(0).to(device)
    x = torch.stack([token_ids, mask])
    return x
    
if __name__ == "__main__":
    start = time.time()
    config = Config()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = load_model(device, config.save_model_best)

    texts = ["你的声音好好听啊六六六，兄弟你可真会说话！", "你的声音好好听啊也不知道为什么，就是愿意和你说话。", \
            '我会在努力的好的，小爱支持你，加油。','你可以哄我睡觉吗要怎么哄']
            
    logging.info("模型预测结果：")
    for text in texts:
        print(text)
        input_ids, mask_attention = process_text(text, config.pretrained_model, device)
        logits = model(input_ids, mask_attention)
        logits = torch.squeeze(logits, 1)
        print(logits)
    end = time.time()
    logging.info(f"耗时为：{end - start} s")