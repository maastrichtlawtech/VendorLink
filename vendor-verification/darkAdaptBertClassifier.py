"""
Python version : 3.8
Description : Trains a BERT classifier with adapter layer in a closed-set vendor verification setting.
"""

# %% Importing Libraries
import os, sys, time
import random
import argparse
from pathlib import Path
import logging

import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import f1_score

import torch


from transformers import BertTokenizer, BertConfig, BertModelWithHeads
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction

from datasets import load_dataset

# %% Loading custom libraries 
sys.path.append('../metrics/')
from performance import f1_score_func, accuracy_per_class

# Loading the custom library
sys.path.append('../process/')
from load_data import FetchData

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Sybil Identification Using Adapter layers in Bert")
parser.add_argument('--model', type=str, default="adapt-bert-classifier", help="Name of the model to be trained")
parser.add_argument('--mode', type=str, default="train", help="Can be train or evaluate")
parser.add_argument('--data',  type=str, default='alpha-dreams', help="""Datasets to be evaluated (can be "shared", "alpha" for alphabay, "dreams", "silk" for silk-road, "alpha-dreams", "dreams-silk", "alpha-silk", or "alpha-dreams-silk")""")
parser.add_argument('--data_dir', type=str, default='../data', help="""Data directory of the pre-processed data (if your data is not pre-processed, we recommend you to go to the statistical model and process it first.)""")
parser.add_argument('--save_dir', type=str, default=os.path.join(os.getcwd(), "../models"), help="""Directory for models to be saved""")
parser.add_argument('-lr', type=float, default=4e-5, help="learning rate")
parser.add_argument('--batch_size', type=int, default=32, help="Batch Size")
parser.add_argument('--nb_epochs', type=int, default=20, help="Number of Epochs")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
parser.add_argument('--version', type=str, default='full', help='Run of small data of full data. can be ("small" or "full").')
parser.add_argument('--setting', type=str, default='high', help="Low for low-resource setting and High for high-resource setting")
parser.add_argument('--n_splits', type=int, default=5, help='Number of trials to perform cross validation')
parser.add_argument('--split_ratio', type=float, default=0.25, help="Splitting ratio for the dataset")
parser.add_argument('--preprocess_flag', action='store_true', help='Preprocess data')
parser.add_argument('--ads_count', type=int, default=20, help="Minimum number of advertisements per vendor")
parser.add_argument('--cuda', action='store_true', help='use CUDA')
args = parser.parse_args()

logging.basicConfig(level=logging.ERROR)
# Creating a directory if the path doesn't exist
Path(os.path.join(args.save_dir, args.model)).mkdir(parents=True, exist_ok=True)
Path(os.path.join(args.save_dir, args.model, "best_model")).mkdir(parents=True, exist_ok=True)

# setting random seed
# pl.seed_everything(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% Loading the datasets
alpha_df = pd.read_csv(os.path.join(args.data_dir, "preprocessed_alpha.csv"), error_bad_lines=False, 
                            lineterminator='\n', usecols=['marketplace', 'title', 'vendor', 'prediction', 'ships_to', 'ships_from', 'description']).drop_duplicates()
dreams_df = pd.read_csv(os.path.join(args.data_dir, "preprocessed_dreams.csv"), error_bad_lines=False, 
                            lineterminator='\n', usecols=['marketplace', 'title', 'vendor', 'prediction', 'ships_to', 'ships_from', 'description']).drop_duplicates()
silk_df = pd.read_csv(os.path.join(args.data_dir, "preprocessed_silk.csv"), error_bad_lines=False, 
                            lineterminator='\n', usecols=['marketplace', 'title', 'vendor', 'prediction', 'ships_to', 'ships_from', 'description']).drop_duplicates()
agora_df = pd.read_csv(os.path.join(args.data_dir, "preprocessed_agora.csv"), error_bad_lines=False, encoding = "ISO-8859-1")
agora_df = agora_df[['Vendor', ' Item', ' Item Description']]
agora_df.columns = ['vendor', 'title', 'description']
agora_df['vendor'] = agora_df['vendor'].apply(lambda x : str(x).lower())

data_df = {"alpha":alpha_df, "dreams":dreams_df, "silk":silk_df, "agora":agora_df}

# %% Loading data
if args.data == "shared":
    if args.setting == "high":
        [(train_alpha, train_dreams, train_silk), (test_alpha, test_dreams, test_silk)] = FetchData(data_df, args.version, args.data, args.split_ratio, args.preprocess_flag, args.setting, args.ads_count,  args.seed).split_data()
    else:
        [(train_valhalla, train_traderoute, train_berlusconi), (test_valhalla, test_traderoute, test_berlusconi)] = FetchData(data_df, args.version, args.data, args.split_ratio, args.preprocess_flag, args.setting, args.ads_count,  args.seed).split_data()
elif args.data == "alpha" or args.data == "dreams" or args.data == "silk":
    [(train_data, train_alpha, train_dreams, train_silk), (test_data, test_alpha, test_dreams, test_silk)] = FetchData(data_df, args.version, args.data, args.split_ratio, args.preprocess_flag, args.setting, args.ads_count, args.seed).split_data()
elif args.data == "valhalla" or args.data == "traderoute" or args.data == "berlusconi":
    [(train_data, train_valhalla, train_traderoute, train_berlusconi), (test_data, test_valhalla, test_traderoute, test_berlusconi)]  = FetchData(data_df, args.version, args.data, args.split_ratio, args.preprocess_flag, args.setting, args.ads_count, args.seed).split_data()

elif args.data == "alpha-dreams" or args.data == "dreams-silk" or args.data == "alpha-silk":
    [(train_alpha_dreams, train_dreams_silk, train_alpha_silk, train_alpha_dreams_silk, train_alpha, train_dreams, train_silk), (test_alpha_dreams, test_dreams_silk, test_alpha_silk, test_alpha_dreams_silk, test_alpha, test_dreams, test_silk)] = FetchData(data_df, args.version, args.data,  args.split_ratio, args.preprocess_flag, args.setting, args.ads_count,  args.seed).split_data()
elif args.data == "valhalla-berlusconi" or args.data == "traderoute-agora":
    [(train_valhalla_traderoute, train_traderoute_berlusconi, train_valhalla_berlusconi, train_traderoute_agora, train_valhalla_traderoute_berlusconi, train_valhalla, train_traderoute, train_berlusconi, train_agora),
                        (test_valhalla_traderoute, test_traderoute_berlusconi, test_valhalla_berlusconi, test_traderoute_agora, test_valhalla_traderoute_berlusconi, test_valhalla, test_traderoute, test_berlusconi, test_agora)] = FetchData(data_df, args.version, args.data,  args.split_ratio, args.preprocess_flag, args.setting, args.ads_count,  args.seed).split_data()

else:
    raise Exception("""Datasets to be evaluated (can be "shared" for shared vendors across different markets, "alpha" for alphabay, "dreams", "silk" for silk-road, "alpha-dreams", "traderoute-berlusconi", "valhalla-traderoute", "valhalla-berlusconi", "dreams-silk", "alpha-silk", or "alpha-dreams-silk")""")

data_files = {"train": "train.csv", "test": "test.csv", "valid":"valid.csv"}
print("Loading the Merged dataset .... ")
dataset = load_dataset("Vageesh/merged", data_files=data_files, use_auth_token="#############")

if args.data == "alpha-dreams" or args.data == "dreams-silk" or args.data == "alpha-silk":
    data = pd.concat([train_dreams, test_dreams, train_alpha, test_alpha, train_silk, test_silk])
    all_vendors = list(data['vendor'].unique())
    vendor_to_idx_dict = {vendor_name:index for index, vendor_name in enumerate(all_vendors)}
    idx_to_vendor_dict = {v: k for k, v in vendor_to_idx_dict.items()}
else:
    raise Exception("Pipeline not implemented for data parameter outside alpha-dreams, dreams-silk, and alpha-silk")

# Loading the Bert tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

def encode_batch(batch):
    """Encodes a batch of input data using the model tokenizer."""
    return tokenizer(batch["text"], max_length=512, truncation=True, padding="max_length")

# Encode the input data
dataset = dataset.map(encode_batch, batched=True)
# The transformers model expects the target class column to be named "labels"
# dataset.rename_column_("label", "labels")
# Transform to pytorch tensors and only output the required columns
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Initializing the model
config = BertConfig.from_pretrained("bert-base-cased", num_labels=len(vendor_to_idx_dict), )
model = BertModelWithHeads.from_pretrained("bert-base-cased", config=config, )

# Add a new adapter
model.add_adapter("dark-adapt-bert")
# Add a matching classification head
model.add_classification_head("dark-adapt-bert", num_labels=len(vendor_to_idx_dict), id2label=idx_to_vendor_dict)
# Activate the adapter
model.train_adapter("dark-adapt-bert")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters :", total_params)

# Finetuning the adapter layers
training_args = TrainingArguments(
                learning_rate=args.lr,
                num_train_epochs=args.nb_epochs,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                logging_steps=10000,
                output_dir=os.path.join(args.save_dir, args.model),
                overwrite_output_dir=True,
                # The next line is important to ensure the dataset labels are properly passed to the model
                remove_unused_columns=False,
                )

# Function to compute F1
def compute_f1_accuracy(p: EvalPrediction):
    preds_flat = np.argmax(p.predictions, axis=1).flatten()
    labels_flat = p.label_ids.flatten()
    print(sklearn.metrics.classification_report(np.array(labels_flat), np.array(preds_flat), digits=4))
    return {"acc": f1_score(labels_flat, preds_flat, average='weighted')}

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    compute_metrics=compute_f1_accuracy,
)

start_time = time.time()
# Training the model
trainer.train()
end_time = time.time()
print("training time:", end_time - start_time)

# Evaluating the trained model
trainer.evaluate()

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_f1_accuracy,
)

trainer.evaluate()

model.save_adapter(os.path.join(args.save_dir, args.model, "best_model"), "dark-adapt-bert")
    
