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
    def __init__(self, data_path, max_length=40):
        query, reply, label = read_data(data_path)
        self.query = query
        self.reply = reply
        self.label = label
        self.tokenizer = BertTokenizer.from_pretrained('./PLM/bert-base-chinese/')
        self.max_len = max_length

    def __getitem__(self, index):
        query = self.query[index]
        reply = self.reply[index]
        label = self.label[index]

        tokenized_input = self.tokenizer(
            query, 
            reply,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
        )

        # 转化成tensor
        token_ids = torch.tensor(tokenized_input['input_ids'])
        mask = torch.tensor(tokenized_input['attention_mask'])
        label = torch.tensor(label)

        return (token_ids, mask), label

    def __len__(self):
        # 得到文本的长度
        return len(self.reply)


if __name__ == "__main__":
    train_path = '../datasets/processed/qa_train_task1.jsonl'
    eval_path = '../datasets/processed/qa_dev_task1.jsonl'
    model_path = 'PLM/bert-base-chinese'
    trainDataset = MyDataset(train_path)
    trainDataloader = DataLoader(trainDataset, batch_size=256, shuffle=False)
    for batch_text, batch_label in trainDataloader:
        # print(batch_text, batch_label)
        print(batch_text[0][0], batch_text[1][0])
        exit()