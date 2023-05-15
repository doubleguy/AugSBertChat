import json
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# 得到一问一答句子对的Dataset，直接用下标取第i条问答对
# eg: {'query': '你的声音好好听啊', 'replies': '六六六，兄弟你可真会说话！', 'label': [4, 1]}
# label[0]表示喜欢，label[1]表示不喜欢
def _get_qa_pair_ds(data_path):
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
        ds = Dataset.from_dict({'query': query, 'replies': reply, 'label': label})
        return ds 
    
# 得到tokenized的问答对数据集，tokenized_ds包含query、replies、label、input_ids、token_type_ids、attention_mask
"""
{'query': '你的声音好好听啊', 
 'replies': '六六六，兄弟你可真会说话！', 
 'label': [4, 1], 
 'input_ids': [[101, 872, 4638, 1898, 102, 1063, 1063, 1063, 8024, 102]], 
 'token_type_ids': [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], 
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
}
"""
def get_tokenized_dataset(data_path, model_path):
    ds = _get_qa_pair_ds(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenized_ds = ds.map(
        lambda x:tokenizer(
            x['query'], x['replies'],
            max_length=40,
            pad_to_max_length=True,
            truncation=True,
        ),
        # 批处理
        batched=True,
        # load_from_cache_file=False
    )
    return tokenized_ds

def get_datasetdict(train_dataset, eval_dataset, test_dataset=None):
    return DatasetDict({'train': train_dataset, 'eval': eval_dataset}) if test_dataset \
        else DatasetDict({'train': train_dataset, 'eval': eval_dataset, 'test': test_dataset})

if __name__ == '__main__':
    train_path = 'datasets\qa_train.jsonl'
    eval_path = 'datasets\qa_dev.jsonl'
    model_path = 'PLM\\bert-base-chinese'
    # ds = _get_qa_pair_ds(data_path)
    # print(ds[0])
    # train_ds = get_tokenized_dataset(train_path, model_path)
    eval_ds = get_tokenized_dataset(eval_path, model_path)
    # datasetdict = get_datasetdict(train_ds, eval_ds)
    # print(eval_ds[:3])
    # print(eval_ds)
    # print(datasetdict)
    # print(datasetdict['eval'][0])
    eval_dataloader = DataLoader(eval_ds,batch_size=3)
    a=0
    for i in eval_dataloader:
        print(i)
        a+=1
        if a==6:
            break




 