import json
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

# 得到一问一答句子对的Dataset，直接用下标取第i条问答对
# eg: {'query': '你的声音好好听啊', 'replies': '六六六，兄弟你可真会说话！', 'label': [4, 1]}
# label[0]表示喜欢，label[1]表示不喜欢
def read_data(data_path):
    with open(data_path, encoding='utf-8') as f:
        query = []
        reply = []
        label = []
        for row in f:
            data = json.loads(row)
            query.append(data['query'])
            reply.append(data['replies']['reply'])
            # label.append((data['replies']['like'], data['replies']['dislike']))
            label.append(data['replies']['like']/(data['replies']['like'] + data['replies']['dislike']))
        return query, reply, label


class MyDataset(Dataset):
    def __init__(self, data_path, max_length=40, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        query, reply, label = read_data(data_path)
        self.query = query
        self.reply = reply
        self.label = label
        self.tokenizer = BertTokenizer.from_pretrained('./PLM/bert-base-chinese/')
        self.max_len = max_length
        self.device = device

    def __getitem__(self, index):
        query = self.query[index]
        reply = self.reply[index]
        label = self.label[index]
        if len(query)+len(reply)-self.max_len>0:
            query = query[:len(query)+len(reply)-self.max_len]

        # 分词
        query_id = self.tokenizer.tokenize(query)
        reply_id = self.tokenizer.tokenize(reply)
        # 加上特殊标志
        text_id = ["[CLS]"] + query_id + ["[SEP]"] + reply_id

        # 编码
        token_id = self.tokenizer.convert_tokens_to_ids(text_id)
        # 掩码  -》
        mask = [1] * len(token_id) + [0] * (self.max_len + 2 - len(token_id))
        # 编码后  -》长度一致
        token_ids = token_id + [0] * (self.max_len + 2 - len(token_id))

        # 转化成tensor
        token_ids = torch.tensor(token_ids).to(self.device)
        mask = torch.tensor(mask).to(self.device)
        label = torch.tensor(label).to(self.device)

        return (token_ids, mask), label

    def __len__(self):
        # 得到文本的长度
        return len(self.reply)


if __name__ == "__main__":
    train_path = 'datasets/qa_train.jsonl'
    eval_path = 'datasets/qa_dev.jsonl'
    model_path = 'PLM/bert-base-chinese'
    trainDataset = MyDataset(eval_path)
    trainDataloader = DataLoader(trainDataset, batch_size=1, shuffle=False)
    for batch_text, batch_label in trainDataloader:
        print(batch_text, batch_label)
        exit()