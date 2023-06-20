"""
This examples trains a CrossEncoder for the STSbenchmark task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it output a continious labels 0...1 to indicate the similarity between the input pair.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_stsbenchmark.py
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import CosineSimilarityLoss
import logging
from datetime import datetime
import json
import torch.nn as nn

# 给定路径，将数据转化成sbert的输入
def process_data(data_path):
    samples = []
    with open(data_path, encoding='utf-8') as f:
        for row in f:
            row_json = json.loads(row)
            q = row_json['query']
            for reply in row_json['replys']:
                if reply['reply']: 
                    reply['like'] = float(reply['like'])
                    reply['dislike'] = float(reply['dislike'])
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
train_batch_size = 2
num_epochs = 15
model_save_path = 'output/task1_sbert_large-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

word_embedding_model = models.Transformer('hfl/chinese-roberta-wwm-ext-large', max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

# Read STSb dataset
logger.info("Read train dataset")

train_samples = process_data('../datasets/raw/datasets_train.jsonl')
dev_samples = process_data('../datasets/raw/datasets_dev.jsonl')

# train_samples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
#    InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]


# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# We add an evaluator, which evaluates the performance during training
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='dev')
# evaluator = EmbeddingSimilarityEvaluator(dev_samples[0], dev_samples[1], dev_samples[2])

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))

train_loss = CosineSimilarityLoss(model)

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], 
          evaluator=evaluator,
          evaluation_steps=1000,
          epochs=num_epochs, 
          warmup_steps=warmup_steps,
          output_path=model_save_path)


