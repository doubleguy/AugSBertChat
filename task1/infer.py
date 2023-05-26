from model import BertForTextClassfication
import torch
from transformers import BertTokenizer
from config import Config
import time
import logging
import json
from collections import deque

def load_test_data(test_path):
    test_data = []
    
    with open(test_path, 'r', encoding='utf-8') as f:
        for row in f:
            r = json.loads(row)
            query, replys = r['query'], r['replys']
            for reply in replys:
                test_data.append((query, reply['reply']))
    return test_data

def load_model(device, model_path):
    myModel = BertForTextClassfication().to(device)
    myModel.load_state_dict(torch.load(model_path))
    myModel.eval()
    return myModel

def process_text(text, bert_pred, device, max_len=40):
    tokenizer = BertTokenizer.from_pretrained(bert_pred)
    
    tokenized_input = tokenizer(
        text[0], 
        text[1],
        max_length=max_len,
        pad_to_max_length=True,
        truncation=True,
    )

    token_ids = tokenized_input['input_ids']
    mask = tokenized_input['attention_mask']
    token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
    mask = torch.tensor(mask).unsqueeze(0).to(device)
    x = torch.stack([token_ids, mask])
    return x

def get_test_answer(texts, model, config, device):
    with open(config.test_data_path, 'r', encoding='utf-8') as fr, open(config.test_answer_path, 'w') as fw:
        i = 0
        for row in fr:
            r = json.loads(row)
            pred = []
            for reply in r['replys']:
                input_ids, mask_attention = process_text(texts[i], config.pretrained_model, device)
                logits = model(input_ids, mask_attention)
                logits = torch.squeeze(logits, 1)
                pred.append(str(logits.item()))
                i += 1
            fw.write('\t'.join(pred) + '\n')

if __name__ == "__main__":
    start = time.time()
    config = Config()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = load_model(device, config.save_model_best)

    # texts = [("你的声音好好听啊六六六,","兄弟你可真会说话！"), ("你的声音好好听啊也不知道为什么","就是愿意和你说话。")]
    texts = load_test_data(config.test_data_path)        
        
    logging.info("模型预测结果：")
    get_test_answer(texts, model, config, device)
    end = time.time()
    logging.info(f"耗时为：{end - start} s")