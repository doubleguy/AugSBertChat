"""
Usage:
sh train_sbert_eda.sh
"""
import logging
import json
import torch
import math
import tqdm
import nlpaug.augmenter.word as naw
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

from datetime import datetime

from sentence_transformers.losses import CosineSimilarityLoss


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
# model_name = "./PLM/text2vec-base-chinese"
# model_name = "./PLM/bert-base-chinese"
model_name = "./PLM/chinese-roberta-wwm-ext"
train_batch_size = 16
num_epochs = 10

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
device = "cuda" if torch.cuda.is_available() else "cpu"
model_save_path = 'output/robert_eda_'+model_name.split('/')[-1].replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#We use distilroberta-base as base model and set num_labels=1, which predicts a continous score between 0 and 1
# model = CrossEncoder(model_name, num_labels=1)
model = SentenceTransformer(model_name)
# Read STSb dataset
logger.info("Read train dataset")

train_samples = process_data('/home/zb/ChatGLM-6B-main/datasets/raw/datasets_train.jsonl')
dev_samples = process_data('/home/zb/ChatGLM-6B-main/datasets/raw/datasets_dev.jsonl')
# train_samples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
#    InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]

##################################################################################
#
# Data Augmentation: Synonym Replacement with BERT using nlpaug
#
##################################################################################

logging.info("Starting with synonym replacement...")

#### Synonym replacement using BERT ####
aug = naw.ContextualWordEmbsAug(model_path='./PLM/bert-base-chinese', action="insert", device=device, aug_p=0.1, batch_size=128)

silver_samples = []
progress = tqdm.tqdm(total=len(train_samples))

for sample in train_samples:
    augmented_texts = aug.augment(sample.texts)
    inp_example = InputExample(texts=augmented_texts, label=sample.label)
    silver_samples.append(inp_example)
    progress.update(1)

progress.reset()
progress.close()
logging.info("Textual augmentation completed....")
logging.info("Number of silver pairs generated: {}".format(len(silver_samples)))

###################################################################
#
# Train SBERT model with both (train + silver) dataset
#
###################################################################

# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
logging.info("Read gold + silver dataset")
train_dataloader = DataLoader(train_samples + silver_samples, shuffle=True, batch_size=train_batch_size)

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

##### Load model and eval on test set
# model = SentenceTransformer("/home/zb/ChatGLM-6B-main/task1_sbert/output/task1_sbert-2023-05-29_19-32-12")
# print(model.encode('aaa'))
# evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='test')
# evaluator(model)


