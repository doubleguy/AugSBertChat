"""
This examples trains a CrossEncoder for the STSbenchmark task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it output a continious labels 0...1 to indicate the similarity between the input pair.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_stsbenchmark.py
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
import json

# 给定路径，将数据转化成sbert的输入
def process_data(data_path):
    samples = []
    with open(data_path, encoding='utf-8') as f:
        for row in f:
            row_json = json.loads(row)
            q = row_json['query']
            for reply in row_json['replys']:
                if reply['reply']: 
                    samples.append(
                        InputExample(
                            texts=[q, reply['reply']], label=reply['like']/(reply['like']+reply['dislike'])
                            )
                        )
    return samples

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout


#Define our Cross-Encoder
model_name = "/home/zb/ChatGLM-6B-main/task1_sbert/PLM/text2vec-base-chinese"
train_batch_size = 16
num_epochs = 20
model_save_path = 'output/task1_ce-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#We use distilroberta-base as base model and set num_labels=1, which predicts a continous score between 0 and 1
model = CrossEncoder(model_name, num_labels=1)

# Read STSb dataset
logger.info("Read train dataset")

train_samples = process_data('/home/zb/ChatGLM-6B-main/datasets/raw/datasets_train.jsonl')
dev_samples = process_data('/home/zb/ChatGLM-6B-main/datasets/raw/datasets_dev.jsonl')

# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)


# We add an evaluator, which evaluates the performance during training
evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='dev')


# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=500,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


##### Load model and eval on test set
# model = CrossEncoder(model_save_path)

# evaluator = CECorrelationEvaluator.from_input_examples(test_samples, name='test')
# evaluator(model)