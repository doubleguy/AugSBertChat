import json
from sentence_transformers import SentenceTransformer, util
import argparse
import os
from collections import deque

def process_data(data_path, mode='train'):
    querys = []
    replys = []
    with open(data_path, encoding='utf-8') as f:
        for row in f:
            row_json = json.loads(row)
            q = row_json['query']
            r = row_json['replies']['reply']
            querys.append(q)
            replys.append(r)
    return (querys, replys)

"""
input data: (query,reply) and model
output: deque[sim_cores]
"""
def get_cos_sim(model, data, output_path=None, source_path=None):
    sim_scores = deque([])
    query = data[0]
    reply = data[1]
    # Compute embedding for both query and reply
    query_embedding = model.encode(data[0], convert_to_tensor=True)
    reply_embedding = model.encode(data[1], convert_to_tensor=True)
    # Compute cosine-similarities
    cosine_scores = util.cos_sim(query_embedding, reply_embedding)
    
    # Output the pairs with their score
    for i in range(len(query)):
        print("{} \t\t {} \t\t Score: {:.4f}".format(query[i], reply[i], cosine_scores[i][i]))
        sim_scores.append(str(cosine_scores[i][i].item()))

    # write sim_scores to txt file
    if output_path and source_path:
        with open(source_path, encoding='utf-8') as fr, open(output_path, encoding='utf-8', mode='w') as fw:
            for source_row in fr:
                source_row_json = json.loads(source_row)
                tmp = []
                for _ in range(len(source_row_json['replys'])):
                    tmp.append(sim_scores.popleft())
                fw.write("\t".join(tmp)+"\n")

def main(args):
    """
    应当得到问答对一一对应的数据
    eg:
        query:[q1,q2,...,qn]
        reply:[r1,r2,...,rn]
        train_data = dev_data = (query,reply)
    """
    train_data = process_data(args.train_data)
    dev_data = process_data(args.dev_data)
    test_data = process_data(args.test_data)

    model = SentenceTransformer(model_name_or_path=args.model_path)
    # get_cos_sim(model, train_data, args.train_output_path, args.train_source_path)
    get_cos_sim(model, dev_data, args.dev_output_path, args.dev_source_path)
    get_cos_sim(model, test_data, args.test_output_path, args.test_source_path)
  
if __name__ == '__main__':
    train_data = "../datasets/processed/qa_train_task1.jsonl"
    dev_data = "../datasets/processed/qa_dev_task1.jsonl"
    test_data = "../datasets/processed/qa_test1_sbert_task1.jsonl"
    
    # model_path = "/home/zb/ChatGLM-6B-main/task1_sbert/output/task1_sbert-2023-05-29_23-41-35"
    model_path = "/home/zb/ChatGLM-6B-main/task1_sbert/output/task1_ce-2023-05-30_10-25-19"
    
    train_output_path = "../datasets/res/task1_sbert_train_0530.txt"
    dev_output_path = "../datasets/res/task1_sbert_dev_0530.txt"
    test_output_path = "../datasets/res/task1_sbert_test_0530.txt"
    train_source_path = "../datasets/raw/datasets_train.jsonl"
    dev_source_path = "../datasets/raw/datasets_dev.jsonl"
    test_source_path = "../datasets/raw/datasets_test_track1.jsonl"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default=train_data, type=str, help ='input train files path')
    parser.add_argument('--dev_data', default=dev_data, type=str, help='input dev files path') 
    parser.add_argument('--test_data', default=test_data, type=str, help='input test files path')    
    parser.add_argument('--model_path', default=model_path, type=str, help='fintuned model path')
    parser.add_argument('--train_output_path', default=train_output_path, type=str, help='model output path, get txt file')
    parser.add_argument('--dev_output_path', default=dev_output_path, type=str, help='model output path, get txt file')
    parser.add_argument('--test_output_path', default=test_output_path, type=str, help='model output path, get txt file')
    parser.add_argument('--train_source_path', default=train_source_path, type=str, help='let output txt file format same as source file if output_path is given')
    parser.add_argument('--dev_source_path', default=dev_source_path, type=str, help='let output txt file format same as source file if output_path is given')
    parser.add_argument('--test_source_path', default=test_source_path, type=str, help='let output txt file format same as source file if output_path is given')
    args = parser.parse_args()
    main(args)
