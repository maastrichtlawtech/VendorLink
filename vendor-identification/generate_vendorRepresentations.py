"""
Python version : 3.8
Description : Generates sentence embedding extracted from the trained classifier for all the advertisements in the dataset.
Note : This file some time to run.
"""

# %% Importing libraries
import os, sys
import random
import argparse
from pathlib import Path
import logging
from tqdm import tqdm 
import pickle

import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from transformers import BertTokenizer, BertForSequenceClassification

# Loading the custom library
sys.path.append('../utilities/')
from load_data import FetchData
from utils import merge_and_create_dataframe
from trainingHelpers import extract_layer_representations_from_bert_layer

import warnings
warnings.filterwarnings('ignore')

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Generates sentence embedding extracted from the trained classifier for all the advertisement in the dataset.")
parser.add_argument('--model', type=str, default="bert", help="Can be roberta or gpt2")
parser.add_argument('--mode', type=str, default="evaluate", help="Can be train or evaluate")
parser.add_argument('--data',  type=str, default='alpha-dreams', help="""Datasets to be evaluated (can be "shared", "alpha" for alphabay, "dreams", "silk" for silk-road, "alpha-dreams", "dreams-silk", "alpha-silk", or "alpha-dreams-silk")""")
parser.add_argument('--data_dir', type=str, default='../data', help="""Data directory of the pre-processed data (if your data is not pre-processed, we recommend you to go to the statistical model and process it first.)""")
parser.add_argument('--model_dir', type=str, default=os.path.join(os.getcwd(), "../models/bert/"), help="""Directory of models saved""")
parser.add_argument('--pickle_dir', type=str, default=os.path.join(os.getcwd(), "../pickled/"), help="""Directory to save the pickled file with sentence representations""")
parser.add_argument('--load_model', type=str, default="epoch_39.model", help="Loading the trained model")
parser.add_argument('--layer', type=str, default="weighted-sum-last-four", help="Layer/'s of the trained model representations are extracted from. Layers can only be last, second-to-last, first, weighted-sum-last-four, all, or concat-last-4")
parser.add_argument('-lr', type=float, default=4e-5, help="learning rate")
parser.add_argument('--batch_size', type=int, default=64, help="Batch Size")
parser.add_argument('--nb_epochs', type=int, default=100, help="Number of Epochs")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
parser.add_argument('-dropout', type=float, default=0.65, help="Dropout for the linear layer to classify")
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
Path(os.path.join(args.pickle_dir)).mkdir(parents=True, exist_ok=True)
Path(os.path.join(args.model_dir, args.model)).mkdir(parents=True, exist_ok=True)
Path(os.path.join(args.model_dir, args.model, "best_model")).mkdir(parents=True, exist_ok=True)

# setting random seed
# pl.seed_everything(args.seed)
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

# %% Loading data
[(train_alpha_dreams, train_dreams_silk, train_alpha_silk, train_alpha_dreams_silk, train_alpha, train_dreams, train_silk), (test_alpha_dreams, test_dreams_silk, test_alpha_silk, test_alpha_dreams_silk, test_alpha, test_dreams, test_silk)] = FetchData(data_df, args.version, args.data,  args.split_ratio, args.preprocess_flag, args.setting, args.ads_count,  args.seed).split_data()

[(train_valhalla_traderoute, train_traderoute_berlusconi, train_valhalla_berlusconi, train_traderoute_agora, train_valhalla_traderoute_berlusconi, train_valhalla, train_traderoute, train_berlusconi, train_agora),
                    (test_valhalla_traderoute, test_traderoute_berlusconi, test_valhalla_berlusconi, test_traderoute_agora, test_valhalla_traderoute_berlusconi, test_valhalla, test_traderoute, test_berlusconi, test_agora)] = FetchData(data_df, args.version, "traderoute-agora",  args.split_ratio, args.preprocess_flag, "low", args.ads_count,  args.seed).split_data()

# Merging data from different datasets
data = pd.concat([train_dreams, test_dreams, train_alpha, test_alpha, train_silk, test_silk])
# data = pd.concat([train_dreams, test_dreams, train_alpha, test_alpha, train_silk, test_silk, train_valhalla, test_valhalla, train_berlusconi, test_berlusconi, train_traderoute, test_traderoute, train_agora, test_agora])

# Vectorizing the data
all_vendors = list(data['vendor'].unique())
vendor_to_idx_dict = {vendor_name:index for index, vendor_name in enumerate(all_vendors)}
data['vendor'] = data['vendor'].replace(vendor_to_idx_dict, regex=True)
# Merging the data
data = merge_and_create_dataframe(data)

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', truncation=True, do_lower_case=True)
model = BertForSequenceClassification.from_pretrained("bert-base-cased",
                                            num_labels=len(vendor_to_idx_dict),
                                            output_attentions=False,
                                            output_hidden_states=True).to(device)

# load model
model.load_state_dict(torch.load(os.path.join(args['model_dir'], args['model'], args['load_model'])))
model.eval()
model.zero_grad()

unique_vendor_id = list(data['labels'].unique())
unique_vendor_id.remove(vendor_to_idx_dict['others'])

pbar = tqdm(total=len(unique_vendor_id))
vendor_representation_dict = {}

for vendor in unique_vendor_id:
    sample_df = data[data['labels']==vendor]
    vendor_representation_list = []

    # Encoding the data through the transformer tokenizer
    encoded_data = tokenizer.batch_encode_plus(sample_df.text.values, 
                                               add_special_tokens=True, 
                                               return_attention_mask=True, 
                                               pad_to_max_length=True, 
                                               max_length=512, 
                                               return_tensors='pt')
    
    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    labels_ = torch.tensor(sample_df.labels.values)

    dataset = TensorDataset(input_ids, attention_masks, labels_)
    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=args.batch_size)
    for idx, batch in enumerate(dataloader):
        representations = extract_layer_representations_from_bert_layer(model, args.layer, batch, device)
        vendor_representation_list.append(representations)
    representations = torch.cat(vendor_representation_list, dim=0)

    vendor_name = list(vendor_to_idx_dict.keys())[list(vendor_to_idx_dict.values()).index(vendor)]
    vendor_representation_dict[vendor_name] = representations
    pbar.update(1)

pbar.close()

with open(os.path.join(args.pickle_dir, "sentence_representations_for_vendors.pickle"), 'wb') as handle:
    pickle.dump(vendor_representation_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)