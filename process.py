import json

# 得到input文件对应的标准提交答案，也就是每一条reply中like占like+dislike的比例
def get_gold(input_file, gold_file):
    with open(input_file, 'r', encoding='utf-8') as fr, \
            open(gold_file, 'w', encoding='utf-8') as fw:
        for line in fr:
            data = json.loads(line)
            gold = [str(d['like'] / (d['like'] + d['dislike'])) for d in data['replys']]
            fw.write('\t'.join(gold) + '\n')

# 得到问答对，将源文件中的每个问句对应一个答案，单独拆成一行。
"""
原始文件：
    {"query": "你可以哄我睡觉吗", "replys": [{"reply": "要怎么哄", "dislike": 30, "like": 12}, {"reply": "亲爱哒，要怎么哄", "dislike": 3, "like": 7}]}
处理后文件：
    {"query": "你可以哄我睡觉吗", "replies": {"reply": "要怎么哄", "dislike": 30, "like": 12}}
    {"query": "你可以哄我睡觉吗", "replies": {"reply": "亲爱哒，要怎么哄", "dislike": 3, "like": 7}}
"""
def get_qa_pair(data_path, output_path):
        with open(data_path, encoding='utf-8') as fr,\
            open(output_path, encoding='utf-8', mode='w') as fw:
            for row in fr:
                data = json.loads(row)
                for r in data['replys']:
                    json.dump({'query': data['query'], 'replies': r}, fw, ensure_ascii=False)
                    fw.write('\n')

if __name__ == "__main__":
    gold_train_file = "./datasets/gold_train.txt"
    input_train_file = "./datasets/datasets_train.jsonl"
    gold_dev_file = "./datasets/gold_dev.txt"
    input_dev_file = "./datasets/datasets_dev.jsonl"
    qa_train_file = "./datasets/qa_train.jsonl"
    qa_dev_file = "./datasets/qa_dev.jsonl"
    get_gold(input_train_file, gold_train_file)
    get_gold(input_dev_file, gold_dev_file)
    get_qa_pair(input_train_file, qa_train_file)
    get_qa_pair(input_dev_file, qa_dev_file)
