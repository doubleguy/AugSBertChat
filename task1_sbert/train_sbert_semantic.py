"""
The script shows how to train Augmented SBERT (In-Domain) strategy for STSb dataset with Semantic Search Sampling.


Methodology:
Three steps are followed for AugSBERT data-augmentation strategy with Semantic Search - 
    1. Fine-tune cross-encoder (BERT) on gold STSb dataset
    2. Fine-tuned Cross-encoder is used to label on Sem. Search sampled unlabeled pairs (silver STSb dataset) 
    3. Bi-encoder (SBERT) is finally fine-tuned on both gold + silver STSb dataset

Citation: https://arxiv.org/abs/2010.08240

Usage:
python train_sts_indomain_semantic.py

OR
python train_sts_indomain_semantic.py pretrained_transformer_model_name top_k

python train_sts_indomain_semantic.py bert-base-uncased 3
"""
from torch.utils.data import DataLoader
from sentence_transformers import models, losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime
import logging
import torch
import tqdm
import sys
import math
from sentence_transformers.losses import CosineSimilarityLoss
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


#Define our sbert
# model_name = "./PLM/text2vec-bert-chinese"
# model_name = "./PLM/bert-base-chinese"
train_batch_size = 8
num_epochs = 10

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
top_k = int(sys.argv[1]) if len(sys.argv) > 2 else 3
max_seq_length = 128
cross_encoder_path = 'output/bert_semantic/cross-encoder_'+model_name.split('/')[-1].replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
bi_encoder_path = 'output/bert_semantic/bi-encoder_'+model_name.split('/')[-1].replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

###### Cross-encoder (simpletransformers) ######
logging.info("Loading cross-encoder model: {}".format(model_name))
# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for cross-encoder model
cross_encoder = CrossEncoder(model_name, num_labels=1)
###### Bi-encoder (sentence-transformers) ######
logging.info("Loading bi-encoder model: {}".format(model_name))
bi_encoder = SentenceTransformer(model_name)

#####################################################
#
# Step 1: Train cross-encoder model with STSbenchmark
#
#####################################################

logging.info("Step 1: Train cross-encoder: {} with (gold dataset)".format(model_name))

train_samples = process_data('../datasets/raw/datasets_train.jsonl')
dev_samples = process_data('../datasets/raw/datasets_dev.jsonl')

# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# We add an evaluator, which evaluates the performance during training
evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='dev')
# evaluator = EmbeddingSimilarityEvaluator(dev_samples[0], dev_samples[1], dev_samples[2])

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))

#Tune the model
cross_encoder.fit(train_dataloader,
          evaluator=evaluator,
          evaluation_steps=1000,
          epochs=num_epochs, 
          warmup_steps=warmup_steps,
          output_path=cross_encoder_path)

############################################################################
#
# Step 2: Find silver pairs to label
#
############################################################################

#### Top k similar sentences to be retrieved ####
#### Larger the k, bigger the silver dataset ####

logging.info("Step 2.1: Generate STSbenchmark (silver dataset) using pretrained SBERT \
    model and top-{} semantic search combinations".format(top_k))

silver_data = []
sentences = set()

for sample in train_samples:
    sentences.update(sample.texts)

sentences = list(sentences) # unique sentences
sent2idx = {sentence: idx for idx, sentence in enumerate(sentences)} # storing id and sentence in dictionary
duplicates = set((sent2idx[data.texts[0]], sent2idx[data.texts[1]]) for data in train_samples) # not to include gold pairs of sentences again


# For simplicity we use a pretrained model
semantic_model_name = model_name
semantic_search_model = SentenceTransformer(semantic_model_name)
logging.info("Encoding unique sentences with semantic search model: {}".format(semantic_model_name))

# encoding all unique sentences present in the training dataset
embeddings = semantic_search_model.encode(sentences, batch_size=train_batch_size, convert_to_tensor=True)

logging.info("Retrieve top-{} with semantic search model: {}".format(top_k, semantic_model_name))

# retrieving top-k sentences given a sentence from the dataset
progress = tqdm.tqdm(total=len(sent2idx))
for idx in range(len(sentences)):
    sentence_embedding = embeddings[idx]
    cos_scores = util.cos_sim(sentence_embedding, embeddings)[0]
    cos_scores = cos_scores.cpu()
    progress.update(1)

    #We use torch.topk to find the highest 5 scores
    top_results = torch.topk(cos_scores, k=top_k+1)
    
    for score, iid in zip(top_results[0], top_results[1]):
        if iid != idx and (iid, idx) not in duplicates:
            silver_data.append((sentences[idx], sentences[iid]))
            duplicates.add((idx,iid))

progress.reset()
progress.close()

logging.info("Length of silver_dataset generated: {}".format(len(silver_data)))
logging.info("Step 2.2: Label silver dataset with cross-encoder: {}".format(model_name))
cross_encoder = CrossEncoder(cross_encoder_path)
silver_scores = cross_encoder.predict(silver_data)

# All model predictions should be between [0,1]
assert all(0.0 <= score <= 1.0 for score in silver_scores)


############################################################################################
#
# Step 3: Train bi-encoder model with both STSbenchmark and labeled AllNlI - Augmented SBERT
#
############################################################################################

logging.info("Step 3: Train bi-encoder: {} with STSbenchmark (gold + silver dataset)".format(model_name))

# Convert the dataset to a DataLoader ready for training
logging.info("Read gold and silver train dataset")
silver_samples = list(InputExample(texts=[data[0], data[1]], label=score) for \
    data, score in zip(silver_data, silver_scores))


train_dataloader = DataLoader(train_samples + silver_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=bi_encoder)

logging.info("Read dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='dev')

# Configure the training.
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the bi-encoder model
bi_encoder.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=bi_encoder_path)