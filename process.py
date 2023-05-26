import json

# 得到input文件对应的标准提交答案，也就是每一条reply中like占like+dislike的比例
def get_gold(input_file, gold_file):
    with open(input_file, 'r', encoding='utf-8') as fr, \
            open(gold_file, 'w', encoding='utf-8') as fw:
        for line in fr:
            data = json.loads(line)
            gold = [str(d['like'] / (d['like'] + d['dislike'])) for d in data['replys']]
            fw.write('\t'.join(gold) + '\n')

# 将原始数据去除nan
def get_raw_data_without_nan(input_file, outputfile, mode='train'):
    with open(input_file, encoding='utf-8') as fr,\
        open(outputfile, encoding='utf-8', mode='w') as fw:
        for row in fr:
            print(row)
            data = json.loads(row)
            tmp = []
            for r in data['replys']:
                if r['reply']!='nan':
                    tmp.append(r)
            if len(tmp)==0 and mode!='test':
                continue
            json.dump({'query': data['query'], 'replys': tmp}, fw, ensure_ascii=False)
            fw.write('\n')

# 得到问答对，将源文件中的每个问句对应一个答案，单独拆成一行。
"""
原始文件：
    {"query": "你可以哄我睡觉吗", "replys": [{"reply": "要怎么哄", "dislike": 30, "like": 12}, {"reply": "亲爱哒，要怎么哄", "dislike": 3, "like": 7}]}
处理后文件：
    task1
        {"query": "你可以哄我睡觉吗", "replies": {"reply": "要怎么哄", "dislike": 30, "like": 12}}
        {"query": "你可以哄我睡觉吗", "replies": {"reply": "亲爱哒，要怎么哄", "dislike": 3, "like": 7}}
    task2
        {"query": "你可以哄我睡觉吗", "replies": "要怎么哄"}
        {"query": "你可以哄我睡觉吗", "replies": "亲爱哒，要怎么哄"}
"""
def get_qa_pair(data_path, output_path, task='task1'):
    if task=='task1':
        with open(data_path, encoding='utf-8') as fr,\
            open(output_path, encoding='utf-8', mode='w') as fw:
            for row in fr:
                data = json.loads(row)
                for r in data['replys']:
                    json.dump({'query': data['query'], 'replies': r}, fw, ensure_ascii=False)
                    fw.write('\n')
              
    prompt = "假设你是小爱同学，请为下面这句话生成一条回复\n"          
    if task=='task2':
        with open(data_path, encoding='utf-8') as fr,\
            open(output_path, encoding='utf-8', mode='w') as fw:
            for row in fr:
                data = json.loads(row)
                for r in data['replys']:
                    if r['like']>=r['dislike'] or len(data['replys'])<=2:
                        json.dump({'query': prompt + data['query'], 'replies': r['reply']}, fw, ensure_ascii=False)
                        fw.write('\n')
    
    if task=='test1':
        with open(data_path, encoding='utf-8') as fr,\
            open(output_path, encoding='utf-8', mode='w') as fw:
            for row in fr:
                data = json.loads(row)
                for r in data['replys']:
                    json.dump({'query': prompt + data['query'], 'replies': r['reply']}, fw, ensure_ascii=False)
                    fw.write('\n')
                    
    if task=='test2':
        with open(data_path, encoding='utf-8') as fr,\
            open(output_path, encoding='utf-8', mode='w') as fw:
            for row in fr:
                data = json.loads(row)
                json.dump({'query': prompt + data['query']}, fw, ensure_ascii=False)
                fw.write('\n')
                
    if task=='task1_glm':
        prompt_start = "指令：请根据问答对给答案打分，打分区间为[0,1]\n[示例1]\n问句：你想要红包吗\n回复：谢谢你，小爱并不需要钱，希望你可以多点时间陪小爱~\n答案：1.0\n[示例2]\n问句：小爱你能看见我吗\n回复：别着急啊，你的小宝贝正在思考怎么回复你。\n答案：0.6\n[示例3]\n问句：我唱歌好听吗\n回复：能让你喜欢的应该是个很棒的人吧~\n答案：0.0\n请根据上面的示例给下面的问答对中的答案打分\n问句："    
        prompt_mid = "\n回复："
        prompt_end = "\n答案："
        
        with open(data_path, encoding='utf-8') as fr,\
            open(output_path, encoding='utf-8', mode='w') as fw:
            for row in fr:
                data = json.loads(row)
                for r in data['replys']:
                    json.dump({'query': prompt_start + data['query'] + prompt_mid + r['reply'] + prompt_end, 'replies': str(r['like']/(r['like']+r['dislike']))}, fw, ensure_ascii=False)
                    fw.write('\n')
    
    if task=='task1_glm_test':
        prompt_start = "指令：请根据问答对给答案打分，打分区间为[0,1]\n[示例1]\n问句：你想要红包吗\n回复：谢谢你，小爱并不需要钱，希望你可以多点时间陪小爱~\n答案：1.0\n[示例2]\n问句：小爱你能看见我吗\n回复：别着急啊，你的小宝贝正在思考怎么回复你。\n答案：0.6\n[示例3]\n问句：我唱歌好听吗\n回复：能让你喜欢的应该是个很棒的人吧~\n答案：0.0\n请根据上面的示例给下面的问答对中的答案打分\n问句："   
        prompt_mid = "\n回复："
        prompt_end = "\n答案："
        
        with open(data_path, encoding='utf-8') as fr,\
            open(output_path, encoding='utf-8', mode='w') as fw:
            for row in fr:
                data = json.loads(row)
                for r in data['replys']:
                    json.dump({'query': prompt_start + data['query'] + prompt_mid + r['reply'] + prompt_end}, fw, ensure_ascii=False)
                    fw.write('\n')
    
if __name__ == "__main__":
    input_train_file = "./datasets/raw/datasets_train.jsonl"
    input_dev_file = "./datasets/raw/datasets_dev.jsonl"
    input_test1_file = "./datasets/raw/datasets_test_track1.jsonl"
    input_test2_file = "./datasets/raw/datasets_test_track2.jsonl"
    gold_train_file = "./datasets/raw/gold_train.txt"
    gold_dev_file = "./datasets/raw/gold_dev.txt"
    qa_train_file = "./datasets/processed/qa_train_task1.jsonl"
    qa_dev_file = "./datasets/processed/qa_dev_task1.jsonl"
    qa_train_file_task1_glm = "./datasets/processed/qa_train_task1_glm.json"
    qa_dev_file_task1_glm = "./datasets/processed/qa_dev_task1_glm.json"
    qa_test_file_task1_glm = "./datasets/processed/qa_test_task1_glm.json"
    qa_train_file_task2 = "./datasets/processed/qa_train_task2.json"
    qa_dev_file_task2 = "./datasets/processed/qa_dev_task2.json"
    qa_test1_file_task2 = "./datasets/processed/qa_test1_task2.json"
    qa_test2_file_task2 = "./datasets/processed/qa_test2_task2.json"
    
    # get data without nan
    # get_raw_data_without_nan(input_test1_file, "./datasets/raw/ffdatasets_test_track1.jsonl")
    # get_raw_data_without_nan(input_test2_file, "./datasets/raw/ffdatasets_test_track2.jsonl", 'test')
    
    # get train and eval gold file
    get_gold(input_train_file, gold_train_file)
    get_gold(input_dev_file, gold_dev_file)
    
    # task1
    get_qa_pair(input_train_file, qa_train_file)# bert微调
    get_qa_pair(input_dev_file, qa_dev_file)# bert微调
    get_qa_pair(input_train_file, qa_train_file_task1_glm, 'task1_glm')# glm P-tuningv2
    get_qa_pair(input_dev_file, qa_dev_file_task1_glm, 'task1_glm')# glm P-tuningv2
    get_qa_pair(input_test1_file, qa_test_file_task1_glm, 'task1_glm_test')# glm P-tuningv2
    
    # task2
    get_qa_pair(input_train_file, qa_train_file_task2, 'task2')
    get_qa_pair(input_dev_file, qa_dev_file_task2, 'task2')
    get_qa_pair(input_test1_file, qa_test1_file_task2, 'test1')
    get_qa_pair(input_test2_file, qa_test2_file_task2, 'test2')
    
    