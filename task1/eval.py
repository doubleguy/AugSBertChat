import time
import torch
from config import Config
from tqdm import tqdm
from dataset import MyDataset
from torch.utils.data import DataLoader
from model import BertForTextClassfication
from evaluation import kl_sim_score
from collections import deque
import transformers
transformers.logging.set_verbosity_error()

def get_predict_like_probability_file(all_pred):
    all_pred = deque(all_pred)
    with open('../datasets/raw/gold_dev.txt', 'r') as f1, open('../datasets/res/task1.txt', 'w') as f2:
        for line in f1:
            num = len(line.strip().split())
            for _ in range(num-1):
                f2.write(str(all_pred.popleft())+'\t')
            f2.write(str(all_pred.popleft())+'\n')

def do_eval(config):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    eval_dataset = MyDataset(config.eval_data_path)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.per_device_eval_batch_size)

    model = BertForTextClassfication().to(device)
    model.load_state_dict(torch.load(config.save_model_best))

    model.eval()
    all_pred, all_label = [], []
    with torch.no_grad():
        for batch_text, batch_label in tqdm(eval_dataloader):
            batch_text[0], batch_text[1], batch_label = batch_text[0].to(device), batch_text[1].to(device), batch_label.to(device)
            pred = model(batch_text[0], batch_text[1])
            pred = torch.squeeze(pred, 1).cpu().numpy().tolist()
            label = batch_label.cpu().numpy().tolist()
            all_pred.extend(pred)
            all_label.extend(label)
    
    get_predict_like_probability_file(all_pred)
    score = kl_sim_score(all_label, all_pred)
    print(f"eval score:{score:.4f}")

if __name__ == "__main__":
    start = time.time()
    config = Config()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    do_eval(config)
    end = time.time()
    print(f"耗时为：{end - start} s")
    