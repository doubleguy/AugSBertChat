import torch
from config import Config
from dataset import MyDataset
from transformers import AutoTokenizer, AutoModel, logging
logging.set_verbosity_error()
from torch.optim import AdamW, SGD
from torch import nn
from torch.utils.data import DataLoader
from model import BertForTextClassfication
from tqdm import tqdm
import time
from test import do_test
from evaluation import kl_sim_score

start = time.time()
config = Config()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# get model input data
train_ds = MyDataset(config.train_data_path)
eval_ds = MyDataset(config.eval_data_path)
train_dataloader = DataLoader(train_ds, batch_size=config.per_device_train_batch_size)
eval_dataloader = DataLoader(eval_ds, batch_size=config.per_device_eval_batch_size)

# get tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)

# get model
model = BertForTextClassfication().to(device)
# optimizer = AdamW(model.parameters(), lr=config.learning_rate)
optimizer = SGD(model.parameters(), lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', min_lr=1e-6, factor=0.5)
# criterion = nn.L1Loss()
criterion = nn.SmoothL1Loss()

score_max = float("-inf")
for epoch in range(config.num_train_epochs):
    loss_sum, count = 0, 0
    model.train()
    for batch_index, (batch_text,batch_label) in enumerate(train_dataloader):
        # shape:[batch_size,1]
        logits = model(batch_text[0], batch_text[1])
        logits = torch.squeeze(logits, 1)
        loss = criterion(logits,batch_label)
        # print(f'logits: {logits} batch_label: {batch_label} loss: {loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss
        count += 1
        
    # logging:输出当前epoch的loss
    print(f'epoch: {epoch + 1}/{config.num_train_epochs+1} Train_loss: {loss_sum / count:.2f}')
    loss_sum, count = 0.0, 0
        
    model.eval()
    all_pred, all_label = [], []
    with torch.no_grad():
        for batch_text, batch_label in tqdm(eval_dataloader):
            pred = model(batch_text[0], batch_text[1])
            pred = torch.squeeze(pred, 1).cpu().numpy().tolist()
            label = batch_label.cpu().numpy().tolist()
            all_pred.extend(pred)
            all_label.extend(label)
            
    score = kl_sim_score(all_label, all_pred)
    print(f"eval score:{score:.4f}")
    if score > score_max:
        print(score, score_max)
        score_max = score
        torch.save(model.state_dict(), config.save_model_best)
        print(f"已保存最佳模型")

    torch.save(model.state_dict(), config.save_model_last)
    scheduler.step(score)
    end = time.time()
    print(f"运行时间：{(end-start)/60%60:.4f} min")

    # do_test(config)


if __name__ == '__main__':
    pass