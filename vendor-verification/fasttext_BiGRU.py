# %% Loading the libraries and data
import os, sys
import random
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torchtext.legacy import data

import spacy.cli 
spacy.cli.download("en")

from torchtext.vocab import Vectors
vectors = Vectors(name='../models/fasttext/fasttextmodel.vec', cache='./')

# Loading the custom utility library
sys.path.append('../utilities/')
from load_data import FetchData, LoadDataForBiGRUFasttext
from utils import merge_and_create_dataframe
from traininghelpers import trainBiGRUFasttext, evaluateBiGRUFasttext

# Loading the custom utility library
sys.path.append('../architectures/')
from GRUClassifier import BiGRUFasttextClassifier

# Loading the custom utility library
sys.path.append('../metrics/')
from performance import evaluate_and_generate_classification_report

# %% Initializing the argparser
parser = argparse.ArgumentParser(description="Trainining a 2-layered BiGRU classifier initialized with fasttext embeddings")
parser.add_argument('--save_model_dir', type=str, default=os.path.join(os.getcwd(), "../models/traderoute-agora/fasttext-BiGRU/"), help="""Directory for models to be saved""")
parser.add_argument('--data_dir', type=str, default='../data', help="""Data directory of the pre-processed data (if your data is not pre-processed, we recommend you to go to the statistical model and process it first.)""")
parser.add_argument('--data',  type=str, default='traderoute-agora', help="""Datasets to be evaluated (can be "shared", "alpha" for alphabay, "dreams", "silk" for silk-road, "alpha-dreams", "dreams-silk", "alpha-silk", or "alpha-dreams-silk")""")
parser.add_argument('--batch_size', type=int, default=64, help="Batch Size")
parser.add_argument('--intent', type=str, default="baseline", help="""Can be baseline or benchmark""")
parser.add_argument('--nb_epochs', type=int, default=100, help="Number of Epochs")
parser.add_argument('--max_vocab_size', type=int, default=100000, help="Maximum number of unique features allowed")
parser.add_argument("--hidden_dim", type=int, default=256, help="Number of hidden units ")
parser.add_argument("--embedding_dim", type=int, default=300, help="Embedding size ")
parser.add_argument('--max_seq_len', type=int, default=512, help="Maximum Sequence length the model will support")
parser.add_argument('--n_layers', type=int, default=2, help="Number of Bi-GRU layers")
parser.add_argument('--early_stopping', type=bool, default=True, help="Early stopping to stop training when early_stopping_metric doesn’t improve")
parser.add_argument('--delta', type=float, default=0.01, help="Minimum change for Early Stopping")
parser.add_argument('--patience', type=int, default=3, help="Early Stopping patience")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
parser.add_argument('--dropout', type=float, default=0.65, help="Dropout for the linear layer to classify")
parser.add_argument('--bidirectional', type=bool, default=True, help="Enables Bi-directional GRI")
parser.add_argument('--version', type=str, default='full', help='Run of small data of full data. can be ("small" or "full").')
parser.add_argument('--setting', type=str, default='low', help="Low for low-resource setting and High for high-resource setting")
parser.add_argument('--split_ratio', type=float, default=0.25, help="Splitting ratio for the dataset")
parser.add_argument('--preprocess_flag', action='store_true', help='Preprocess data')
parser.add_argument('--ads_count', type=int, default=20, help="Minimum number of advertisements per vendor")
parser.add_argument('-lr', type=float, default=4e-5, help="learning rate")
args = parser.parse_args()

# setting random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Path(args['save_model_dir']).mkdir(parents=True, exist_ok=True)

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

if args.data == "alpha-dreams" or args.data == "dreams-silk" or args.data == "alpha-silk":
    if args.intent == "baseline":
        data = pd.concat([train_dreams, test_dreams])
    elif args.intent == "benchmark":
        data = pd.concat([train_dreams, test_dreams, train_alpha, test_alpha, train_silk, test_silk])
    else:
        raise Exception("intent argument can only be baseline or benchmark ...")

    all_vendors = list(data['vendor'].unique())
    vendor_to_idx_dict = {vendor_name:index for index, vendor_name in enumerate(all_vendors)}

    if args.intent == "baseline":
        # Vectorizing class labels
        train_dreams['vendor'] = train_dreams['vendor'].replace(vendor_to_idx_dict, regex=True)
        test_dreams['vendor'] = test_dreams['vendor'].replace(vendor_to_idx_dict, regex=True)

        train_df = merge_and_create_dataframe(train_dreams).drop_duplicates()
        test_df = merge_and_create_dataframe(test_dreams).drop_duplicates()
    
    elif args.intent == "benchmark":
        train_dreams['vendor'] = train_dreams['vendor'].replace(vendor_to_idx_dict, regex=True)
        test_dreams['vendor'] = test_dreams['vendor'].replace(vendor_to_idx_dict, regex=True)
        train_alpha['vendor'] = train_alpha['vendor'].replace(vendor_to_idx_dict, regex=True)
        test_alpha['vendor'] = test_alpha['vendor'].replace(vendor_to_idx_dict, regex=True)
        train_silk['vendor'] = train_silk['vendor'].replace(vendor_to_idx_dict, regex=True)
        test_silk['vendor'] = test_silk['vendor'].replace(vendor_to_idx_dict, regex=True)

        df_train = pd.concat([train_dreams, train_alpha, train_silk])
        df_test = pd.concat([test_dreams, test_alpha, test_silk])

        train_df = merge_and_create_dataframe(df_train).drop_duplicates()
        test_df = merge_and_create_dataframe(df_test).drop_duplicates()

    else:
        raise Exception("intent argument can only be baseline or benchmark ...")

elif args.data == "valhalla-berlusconi":
    data = pd.concat([train_valhalla, test_valhalla, train_berlusconi, test_berlusconi])
    all_vendors = list(data['vendor'].unique())
    vendor_to_idx_dict = {vendor_name:index for index, vendor_name in enumerate(all_vendors)}
    
    # Vectorizing class labels
    train_valhalla['vendor'] = train_valhalla['vendor'].replace(vendor_to_idx_dict, regex=True)
    test_valhalla['vendor'] = test_valhalla['vendor'].replace(vendor_to_idx_dict, regex=True)
    train_berlusconi['vendor'] = train_berlusconi['vendor'].replace(vendor_to_idx_dict, regex=True)
    test_berlusconi['vendor'] = test_berlusconi['vendor'].replace(vendor_to_idx_dict, regex=True)

    # merging the training and test data
    df_train = pd.concat([train_valhalla, train_berlusconi])
    df_test = pd.concat([test_valhalla, test_berlusconi])

    train_df = merge_and_create_dataframe(df_train).drop_duplicates()
    test_df = merge_and_create_dataframe(df_test).drop_duplicates()

elif args.data == "traderoute-agora":
    data = pd.concat([train_traderoute, test_traderoute, train_agora, test_agora])
    all_vendors = list(data['vendor'].unique())
    vendor_to_idx_dict = {vendor_name:index for index, vendor_name in enumerate(all_vendors)}
    
    # Vectorizing class labels
    train_traderoute['vendor'] = train_traderoute['vendor'].replace(vendor_to_idx_dict, regex=True)
    test_traderoute['vendor'] = test_traderoute['vendor'].replace(vendor_to_idx_dict, regex=True)
    train_agora['vendor'] = train_agora['vendor'].replace(vendor_to_idx_dict, regex=True)
    test_agora['vendor'] = test_agora['vendor'].replace(vendor_to_idx_dict, regex=True)

    # merging the training and test data
    df_train = pd.concat([train_traderoute, train_agora])
    df_test = pd.concat([test_traderoute, test_agora])
    
    train_df = merge_and_create_dataframe(df_train).drop_duplicates()
    test_df = merge_and_create_dataframe(df_test).drop_duplicates()

else:
    raise Exception("The data argument can only be amongst alpha-dreams, dreams-silk, alpha-silk, traderoute-agora, or valhalla-berlusconi")

# Splitting the data into train and validation dataset
train_df, valid_df = train_test_split(train_df, test_size=0.05, random_state= random.seed(args.seed))

print("Shape of the training data:", train_df.shape)
print("Shape of the test data:", test_df.shape)
print("Shape of the validation data:", valid_df.shape)

# Processing, loading, and splitting data
TEXT = data.Field(tokenize = 'spacy', batch_first=True, include_lengths = True)
LABEL = data.LabelField(dtype = torch.float, batch_first=True)
    
fields = [('text',TEXT), ('label',LABEL)]

train_ds, val_ds = LoadDataForBiGRUFasttext.splits(fields, train_df=train_df, val_df=valid_df)
train_ds, test_ds = LoadDataForBiGRUFasttext.splits(fields, train_df=train_df, val_df=test_df)

# Importing fasttext vectors
TEXT.build_vocab(train_ds, val_ds, test_ds, max_size = args.max_vocab_size, vectors = vectors, unk_init = torch.Tensor.zero_)
LABEL.build_vocab(train_ds, val_ds, test_ds)

#No. of unique tokens in text
print("Size of TEXT vocabulary:",len(TEXT.vocab))

#No. of unique tokens in label
print("Size of LABEL vocabulary:",len(LABEL.vocab))

# Commonly used words
# print(TEXT.vocab.freqs.most_common(10))  

train_iterator, valid_iterator = data.BucketIterator.splits((train_ds, val_ds), batch_size = args.batch_size, sort_within_batch = True, device = device)
data_iterator = data.BucketIterator.splits((train_ds, val_ds), batch_size = args.batch_size, sort_within_batch = True, device = device)

# Building model

INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(vendor_to_idx_dict)
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] # padding
    
#creating instance of our BiGRUFasttextClassifier class

model = BiGRUFasttextClassifier(INPUT_DIM, args.embedding_dim, args.hidden_dim, OUTPUT_DIM, args.n_layers, args.bidirectional, args.dropout, PAD_IDX)

# No. of trianable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
print(f'The model has {count_parameters(model):,} trainable parameters')

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

#  to initiaise padded to zeros
model.embedding.weight.data[PAD_IDX] = torch.zeros(args.embedding_dim)

t = time.time()

best_valid_loss = float('inf')
model.to(device) # GRU to GPU

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.nb_epochs):
    
    train_loss = trainBiGRUFasttext(model, train_iterator, optimizer, criterion)
    valid_loss = evaluateBiGRUFasttext(model, valid_iterator, criterion)

    print(f'\t Epoch: {epoch}| Train Loss: {train_loss:.4f} | Val. Loss: {valid_loss:.4f}')

    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), os.path.join(args['save_model_dir'], 'best_model.pt'))
    
print(f'time:{time.time()-t:.3f}')

train_iterator, test_iterator = data.BucketIterator.splits((train_ds, test_ds), batch_size = args.batch_size, sort_within_batch = True, device = device)

print("Test perfomance ...")
evaluate_and_generate_classification_report(model, test_iterator)

