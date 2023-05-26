import torch
import os
from config import Config
from dataset import MyDataset
from transformers import AutoTokenizer, AutoModel, logging, get_linear_schedule_with_warmup
logging.set_verbosity_error()
from torch.optim import AdamW, SGD
from torch import nn
from torch.utils.data import DataLoader
from model import BertForTextClassfication
from tqdm import tqdm
import time
from eval import do_eval
from evaluation import kl_sim_score

def seed_torch(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

start = time.time()
config = Config()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# random seed
seed_torch(config.seed)

# get model input data
train_ds = MyDataset(config.train_data_path)
eval_ds = MyDataset(config.eval_data_path)
train_dataloader = DataLoader(train_ds, batch_size=config.per_device_train_batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_ds, batch_size=config.per_device_eval_batch_size, shuffle=True)

# get tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)

# get model
model = BertForTextClassfication().to(device)
optimizer = AdamW(model.parameters(), lr=config.learning_rate)
# optimizer = SGD(model.parameters(), lr=config.learning_rate)
# Warmup
len_dataset = len(train_ds)
total_steps = (len_dataset // config.per_device_train_batch_size) * config.num_train_epochs if len_dataset % \
    config.per_device_train_batch_size == 0 else (len_dataset // config.per_device_train_batch_size + 1) * config.num_train_epochs
warm_up_ratio = 0.1
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio * total_steps, num_training_steps=total_steps)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', min_lr=1e-6, factor=0.5)
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
# criterion = nn.SmoothL1Loss()
# criterion = nn.CrossEntropyLoss()

score_max = float("-inf")
loss_min = float("inf")
for epoch in range(config.num_train_epochs):
    loss_sum, count = 0, 0
    model.train()
    for batch_index, (batch_text,batch_label) in enumerate(train_dataloader):
        batch_text[0], batch_text[1], batch_label = batch_text[0].to(device), batch_text[1].to(device), batch_label.to(device)
        # shape:[batch_size,1]
        logits = model(batch_text[0], batch_text[1])
        logits = torch.squeeze(logits, 1)
        loss = criterion(logits,batch_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_sum += loss
        count += 1
        
    # logging:输出当前epoch的loss
    print(f'epoch: {epoch + 1}/{config.num_train_epochs+1} Train_loss: {loss_sum / count:.2f}')
    loss_sum, count = 0.0, 0
        
    model.eval()
    all_pred, all_label = [], []
    with torch.no_grad():
        for batch_text, batch_label in tqdm(eval_dataloader):
            batch_text[0], batch_text[1], batch_label = batch_text[0].to(device), batch_text[1].to(device), batch_label.to(device)
            logits = model(batch_text[0], batch_text[1])
            loss = criterion(logits, batch_label)
            pred = torch.squeeze(logits, 1).cpu().numpy().tolist()
            label = batch_label.cpu().numpy().tolist()
            all_pred.extend(pred)
            all_label.extend(label)
            loss_sum += loss
            count += 1
            
    score = kl_sim_score(all_label, all_pred)
    print(f"Eval loss: {loss_sum / count:.2f}, eval score:{score:.4f}")
    
    if loss_sum/count < loss_min and score > score_max:
        print(f"模型更新 score_max:{score_max}->{score}, loss_min:{loss_min}->{loss_sum / count:.2f}")
        score_max = score
        loss_min = loss_sum/count
        torch.save(model.state_dict(), config.save_model_best)
        print(f"已保存最佳模型")

    torch.save(model.state_dict(), config.save_model_last)
    # scheduler.step(score)
    end = time.time()
    print(f"运行时间：{(end-start)/60%60:.4f} min")
    # do_eval(config)


if __name__ == '__main__':
    pass