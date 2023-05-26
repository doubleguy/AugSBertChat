from transformers import AutoModel
import torch.nn as nn
import torch

class BertForTextClassfication(nn.Module):
    def __init__(self, embedding_size=768, hidden_size=128, num_labels=1, dropout=0.1):
        super(BertForTextClassfication, self).__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained('./PLM/bert-base-chinese/')
        # 让 bert 模型进行微调（参数在训练过程中变化）
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.linear = nn.Linear(embedding_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.MSELoss()
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.linear(bert_output.pooler_output)
        output = self.classifier(output)
        # logits = self.sigmoid(output)
        # return logits
        return output