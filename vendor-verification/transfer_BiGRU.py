"""
Python version : 3.8
Description : Trains a 2-layered BiGRU classifier initialized with pre-trained BERT-based Classifier
"""

# %% Loading Libraries
import os, sys, time
import random
import argparse
from pathlib import Path
import logging
from torch import cuda
from tqdm import tqdm 
import pickle

from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from transformers import BertTokenizer, BertForSequenceClassification

# %% Loading custom evaluation libraries 
sys.path.append('../metrics/')
from performance import f1_score_func, accuracy_per_class

# Loading the custom utility library
sys.path.append('../utilities/')
from load_data import FetchData
from utils import merge_and_create_dataframe
from traininghelpers import trainGRUBERT, evaluateGRUBERT

# Loading the custom GRU achitecture
sys.path.append('../architectures/')
from GRUClassifier import BiGRUBertClassifier

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Trainining a 2-layered BiGRU classifier initialized with pre-trained BERT-based Classifier")
parser.add_argument('--pretrained_model', type=str, default="bert", help="Pre-trained model to intialize the BiGRU model")
parser.add_argument('--mode', type=str, default="train", help="Can be train or evaluate")
parser.add_argument('--data_to_load_pretrain_model',  type=str, default='alpha-dreams', help="""Datasets to be loaded for pre-trained model (can be "alpha-dreams", "dreams-silk", "alpha-silk", or "alpha-dreams-silk")""")
parser.add_argument('--data_to_train',  type=str, default='valhalla-berlusconi', help="""Datasets to be evaluated (can be "traderoute-agora" or "valhalla-berlusconi")""")
parser.add_argument('--data_dir', type=str, default='../data', help="""Data directory of the pre-processed data (if your data is not pre-processed, we recommend you to go to the statistical model and process it first.)""")
parser.add_argument('--bert_layer', type=str, default="weighted-sum-last-four", help="""Bert layer to extract embeddings from. Layer can only be last, second-to-last, first, weighted-sum-last-four, all, or concat-last-4 """)
parser.add_argument('--load_model', type=str, default="../models/bert/epoch_39.model", help="Loading the trained model")
parser.add_argument('--save_model_dir', type=str, default=os.path.join(os.getcwd(), "../models/knowledge-transfer/valhalla-berlusconi/weighted-sum-last-four/"), help="""Directory for models to be saved""")
parser.add_argument('-lr', type=float, default=4e-5, help="learning rate")
parser.add_argument('--batch_size', type=int, default=32, help="Batch Size")
parser.add_argument('--nb_epochs', type=int, default=100, help="Number of Epochs")
parser.add_argument("--hidden_units", type=int, default=768, help="Number of hidden units ")
parser.add_argument("--embedding_len", type=int, default=768, help="Embedding size ")
parser.add_argument('--max_seq_len', type=int, default=512, help="Maximum Sequence length the model will support")
parser.add_argument('--n_layers', type=int, default=2, help="Number of Bi-GRU layers")
parser.add_argument('--early_stopping', type=bool, default=True, help="Early stopping to stop training when early_stopping_metric doesn’t improve")
parser.add_argument('--eval_per_steps', type=int, default=2000, help="Evaluate after n number of steps in training data")
parser.add_argument('--delta', type=float, default=0.01, help="Minimum change for Early Stopping")
parser.add_argument('--patience', type=int, default=3, help="Early Stopping patience")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
parser.add_argument('-dropout', type=float, default=0.65, help="Dropout for the linear layer to classify")
parser.add_argument('--version', type=str, default='full', help='Run of small data of full data. can be ("small" or "full").')
parser.add_argument('--setting', type=str, default='low', help="Low for low-resource setting and High for high-resource setting")
parser.add_argument('--n_splits', type=int, default=5, help='Number of trials to perform cross validation')
parser.add_argument('--split_ratio', type=float, default=0.25, help="Splitting ratio for the dataset")
parser.add_argument('--preprocess_flag', action='store_true', help='Preprocess data')
parser.add_argument('--ads_count', type=int, default=20, help="Minimum number of advertisements per vendor")
args = parser.parse_args()


logging.basicConfig(level=logging.ERROR)
# Creating a directory if the path doesn't exist
Path(os.path.join(args.save_model_dir, args.pretrained_model)).mkdir(parents=True, exist_ok=True)

# setting random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

[(train_alpha_dreams, train_dreams_silk, train_alpha_silk, train_alpha_dreams_silk, train_alpha, train_dreams, train_silk), (test_alpha_dreams, test_dreams_silk, test_alpha_silk, test_alpha_dreams_silk, test_alpha, test_dreams, test_silk)] = FetchData(data_df, args.version, args.data_to_load_pretrain_model,  args.split_ratio, args.preprocess_flag, args.setting, args.ads_count,  args.seed).split_data()
[(train_valhalla_traderoute, train_traderoute_berlusconi, train_valhalla_berlusconi, train_traderoute_agora, train_valhalla_traderoute_berlusconi, train_valhalla, train_traderoute, train_berlusconi, train_agora), (test_valhalla_traderoute, test_traderoute_berlusconi, test_valhalla_berlusconi, test_traderoute_agora, test_valhalla_traderoute_berlusconi, test_valhalla, test_traderoute, test_berlusconi, test_agora)] = FetchData(data_df, args.version, args.data_to_train,  args.split_ratio, args.preprocess_flag, "low", args.ads_count,  args.seed).split_data()

# %% Loading pre-trained model
data = pd.concat([train_dreams, test_dreams, train_alpha, test_alpha, train_silk, test_silk])
all_vendors = list(data['vendor'].unique())
vendor_to_idx_dict = {vendor_name:index for index, vendor_name in enumerate(all_vendors)}

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', truncation=True, do_lower_case=True)
bert_model = BertForSequenceClassification.from_pretrained("bert-base-cased",
                                            num_labels=len(vendor_to_idx_dict),
                                            output_attentions=False,
                                            output_hidden_states=True).to(device)

# load model
bert_model.load_state_dict(torch.load(args.load_model))
bert_model.eval()
bert_model.zero_grad()

if args.data_to_train == "traderoute-agora":
    data_df = pd.concat([train_traderoute, test_traderoute, train_agora, test_agora])
    all_vendors = list(data_df['vendor'].unique())
    vendor_to_idx_dict = {vendor_name:index for index, vendor_name in enumerate(all_vendors)}
    
    train_traderoute['vendor'] = train_traderoute['vendor'].replace(vendor_to_idx_dict, regex=True)
    test_traderoute['vendor'] = test_traderoute['vendor'].replace(vendor_to_idx_dict, regex=True)
    train_agora['vendor'] = train_agora['vendor'].replace(vendor_to_idx_dict, regex=True)
    test_agora['vendor'] = test_agora['vendor'].replace(vendor_to_idx_dict, regex=True)

    train_traderoute = merge_and_create_dataframe(train_traderoute)
    test_traderoute = merge_and_create_dataframe(test_traderoute)
    train_agora = merge_and_create_dataframe(train_agora)
    test_agora = merge_and_create_dataframe(test_agora)

    # merging the training and test data
    train_df = pd.concat([train_traderoute, train_agora]).drop_duplicates()
    test_df = pd.concat([test_traderoute, test_agora]).drop_duplicates()

elif args.data_to_train == "valhalla-berlusconi":
    data_df = pd.concat([train_valhalla, test_valhalla, train_berlusconi, test_berlusconi])
    all_vendors = list(data_df['vendor'].unique())
    vendor_to_idx_dict = {vendor_name:index for index, vendor_name in enumerate(all_vendors)}

    train_valhalla['vendor'] = train_valhalla['vendor'].replace(vendor_to_idx_dict, regex=True)
    test_valhalla['vendor'] = test_valhalla['vendor'].replace(vendor_to_idx_dict, regex=True)
    train_berlusconi['vendor'] = train_berlusconi['vendor'].replace(vendor_to_idx_dict, regex=True)
    test_berlusconi['vendor'] = test_berlusconi['vendor'].replace(vendor_to_idx_dict, regex=True)

    train_valhalla = merge_and_create_dataframe(train_valhalla)
    test_valhalla = merge_and_create_dataframe(test_valhalla)
    train_berlusconi = merge_and_create_dataframe(train_berlusconi)
    test_berlusconi = merge_and_create_dataframe(test_berlusconi)

    # merging the training and test data
    train_df = pd.concat([train_valhalla, train_berlusconi]).drop_duplicates()
    test_df = pd.concat([test_valhalla, test_berlusconi]).drop_duplicates()
    
else:
    raise Exception("data_to_train can only be either valhalla-berlusconi or traderoute-agora")

# Splitting the data into train and validation dataset
train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=args.seed, shuffle=True)

# Initializing the GRU model for Bert Embeddings
gru_model = BiGRUBertClassifier(vocab_size=len(tokenizer.get_vocab()), embedding_size=args.embedding_len, hidden_units=args.hidden_units, 
                            max_seq_len=args.max_seq_len, batch_size=args.batch_size, n_layers=args.n_layers, 
                            output_size=len(all_vendors))

num_of_parameters = sum(map(torch.numel, gru_model.parameters()))
print("Number of parameters:", num_of_parameters)

# Encoding the train data through the transformer tokenizer
encoded_data = tokenizer.batch_encode_plus(train_df.text.values, 
                                           add_special_tokens=True, 
                                           return_attention_mask=True, 
                                           pad_to_max_length=True, 
                                           max_length=args.max_seq_len, 
                                           return_tensors='pt')
input_ids = encoded_data['input_ids']
attention_masks = encoded_data['attention_mask']
labels_ = torch.tensor(train_df.labels.values)
dataset = TensorDataset(input_ids, attention_masks, labels_)

# Data Loaders
train_dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=args.batch_size)
# Encoding the val data through the transformer tokenizer
encoded_data = tokenizer.batch_encode_plus(val_df.text.values, 
                                           add_special_tokens=True, 
                                           return_attention_mask=True, 
                                           pad_to_max_length=True, 
                                           max_length=args.max_seq_len, 
                                           return_tensors='pt')
input_ids = encoded_data['input_ids']
attention_masks = encoded_data['attention_mask']
labels_ = torch.tensor(val_df.labels.values)
dataset = TensorDataset(input_ids, attention_masks, labels_)
# Data Loaders
valid_dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=args.batch_size)
# valid_weights = extract_layer_representations_from_bert_last_layer(bert_model, valid_dataloader, attention_masks, vendor_to_idx_dict, device)

# Encoding the test data through the transformer tokenizer
encoded_data = tokenizer.batch_encode_plus(test_df.text.values, 
                                           add_special_tokens=True, 
                                           return_attention_mask=True, 
                                           pad_to_max_length=True, 
                                           max_length=512, 
                                           return_tensors='pt')
input_ids = encoded_data['input_ids']
attention_masks = encoded_data['attention_mask']
labels_ = torch.tensor(test_df.labels.values)
dataset = TensorDataset(input_ids, attention_masks, labels_)
# Data Loaders
test_dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=args.batch_size)
# test_weights = extract_layer_representations_from_bert_last_layer(bert_model, test_dataloader, attention_masks, vendor_to_idx_dict, device)

# %% Training and evaluating the model

# Initializing the optimizer
optimizer = torch.optim.Adam(gru_model.parameters(), lr=args.lr)

# Training the model
start_time = time.time()
trainGRUBERT(model=gru_model, pretrained_model=bert_model, layer=args.bert_layer, optimizer=optimizer, criterion = nn.CrossEntropyLoss(), train_loader = train_dataloader,
            valid_loader = valid_dataloader, num_epochs = args.nb_epochs, max_seq_len = args.max_seq_len,
            batch_size = args.batch_size, eval_every = len(train_dataloader) // 2, file_path = args.save_model_dir, device=device)
end_time = time.time()
print("Total time taken :", end_time - start_time)

# Evaluating the model on test data
print("Evaluation on Test data:")
# evaluate_GRU_BERT_model(model=gru_model, valid_loader=test_dataloader, valid_text_embeddings=test_weights, batch_size=args.batch_size, device=device)
evaluateGRUBERT(model=gru_model, pretrained_model=bert_model, layer=args.bert_layer, valid_loader=test_dataloader, batch_size=args.batch_size, device=device)
