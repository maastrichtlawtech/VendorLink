"""
    Python version : 3.8
    Description : Involves training of transformers-based contextualized models
"""

# %% Importing Libraries
import os, sys, time
import random
import argparse
from pathlib import Path
import logging
from torch import cuda
from tqdm import tqdm 

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup

# Loading the custom library
sys.path.append('../utilities/')
from load_data import FetchData
from utils import merge_and_create_dataframe
from trainingHelpers import trainTransformers, evaluateTransformers

from sklearn.metrics import classification_report

sys.path.append('../architectures/')
from contextualized import LoadHuggingFaceCheckpoints

sys.path.append('../metrics/')
from performance import f1_score_func

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Training transformers-based contextualized models")
parser.add_argument('--model', type=str, default="bert", help="Can be bert, roberta, electra, distill, or gpt2")
parser.add_argument('--mode', type=str, default="train", help="Can be train or evaluate")
parser.add_argument('--data',  type=str, default='alpha-dreams', help="""Datasets to be evaluated (can be "shared", "alpha" for alphabay, "dreams", "silk" for silk-road, "alpha-dreams", "dreams-silk", "alpha-silk", or "alpha-dreams-silk")""")
parser.add_argument('--data_dir', type=str, default='../data', help="""Data directory of the pre-processed data (if your data is not pre-processed, we recommend you to go to the statistical model and process it first.)""")
parser.add_argument('--save_dir', type=str, default=os.path.join(os.getcwd(), "../models/"), help="""Directory for models to be saved""")
parser.add_argument('--intent', type=str, default="baseline", help="""Can be baseline or benchmark""")
parser.add_argument('--load_model', type=str, default="epoch_39.model", help="Loading the trained model")
parser.add_argument('-lr', type=float, default=4e-5, help="learning rate")
parser.add_argument('--batch_size', type=int, default=32, help="Batch Size")
parser.add_argument('--nb_epochs', type=int, default=100, help="Number of Epochs")
parser.add_argument("--hidden_states", type=int, default=512, help="Number of hidden states ")
parser.add_argument('--max_seq_len', type=int, default=512, help="Maximum Sequence length the model will support")
parser.add_argument('--early_stopping', type=bool, default=True, help="Early stopping to stop training when early_stopping_metric doesn’t improve")
parser.add_argument('--eval_per_steps', type=int, default=2000, help="Evaluate after n number of steps in training data")
parser.add_argument('--delta', type=float, default=0.01, help="Minimum change for Early Stopping")
parser.add_argument('--patience', type=int, default=3, help="Early Stopping patience")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
parser.add_argument('-dropout', type=float, default=0.65, help="Dropout for the linear layer to classify")
parser.add_argument('--version', type=str, default='full', help='Run of small data of full data. can be ("small" or "full").')
parser.add_argument('--setting', type=str, default='high', help="Low for low-resource setting and High for high-resource setting")
parser.add_argument('--n_splits', type=int, default=5, help='Number of trials to perform cross validation')
parser.add_argument('--split_ratio', type=float, default=0.25, help="Splitting ratio for the dataset")
parser.add_argument('--preprocess_flag', action='store_true', help='Preprocess data')
parser.add_argument('--ads_count', type=int, default=20, help="Minimum number of advertisements per vendor")
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

# %% Helper function to load and pre-process individual test datasets
# Is only called if the mode parameter is selected to evaluate
def create_data_for_evaluation(test_data):
    vendors = [vendor if vendor in vendor_to_idx_dict.keys() else 'others' for vendor in test_data['vendor']]
    test_data['vendor'] = vendors
    test_data['vendor'] = test_data['vendor'].replace(vendor_to_idx_dict, regex=True)
    df_test = merge_and_create_dataframe(test_data)
    
    encoded_data_test = tokenizer.batch_encode_plus(df_test.text.values, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True, 
                                            max_length=args.max_seq_len, return_tensors='pt')
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(list(df_test.labels.values))

    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    dataloader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=args.batch_size)
    
    return dataloader_test

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

# %% Loading the model
tokenizer, model = LoadHuggingFaceCheckpoints(args.model, vendor_to_idx_dict).load_model()

# Converting the data into the DataLoader format
# Splitting the training data into training and validation data
train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=args.seed, shuffle=True)

# Encoding the data through the transformer tokenizer
encoded_data_train = tokenizer.batch_encode_plus(train_df.text.values, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True, 
                                                max_length=args.max_seq_len, return_tensors='pt')
input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(train_df.labels.values)

encoded_data_test = tokenizer.batch_encode_plus(test_df.text.values, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True, 
                                                max_length=args.max_seq_len, return_tensors='pt')
input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(test_df.labels.values)
                                    
encoded_data_val = tokenizer.batch_encode_plus(val_df.text.values, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True, 
                                                max_length=args.max_seq_len, return_tensors='pt')
input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(val_df.labels.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

# Data Loaders
# We use RandomSampler for training and SequentialSampler for testing and validation
dataloader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=args.batch_size)
dataloader_validation = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=args.batch_size)        
dataloader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=args.batch_size)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train)*args.nb_epochs)

# %% Training model
if args.mode == "train":
    start_time = time.time()
    trainTransformers(model, vendor_to_idx_dict, dataloader_train, dataloader_validation, args.nb_epochs, optimizer, scheduler, device, os.path.join(args.save_dir, args.model))
    end_time = time.time()
    print("Total time taken : ", end_time - start_time)

# %% Evaluating model
elif args.mode == "evaluate":
    model.load_state_dict(torch.load(os.path.join(args.save_dir, args.model, args.load_model), map_location=device))
    # Evaluating on the test dataset
    _, predictions, true_vals = evaluateTransformers(model, dataloader_test, device)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'F1 Score for test dataset (Weighted): {val_f1}')
    print(classification_report(predictions, true_vals, digits=4))
        
    if args.intent == "benchmark":
        # Evaluating on the Dreams dataset
        dataloader_test = create_data_for_evaluation(test_dreams)
        _, predictions, true_vals = evaluateTransformers(model, dataloader_test, device)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'F1 Score for Dreams dataset (Weighted): {val_f1}')
        print(classification_report(predictions, true_vals, digits=4))
    
        # Evaluating on the Alphabay dataset
        dataloader_test = create_data_for_evaluation(test_alpha)
        _, predictions, true_vals = evaluateTransformers(model, dataloader_test, device)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'F1 Score for Alphabay dataset (Weighted): {val_f1}')
        print(classification_report(predictions, true_vals, digits=4))

        # Evaluation on the Silk-Road dataset
        dataloader_test = create_data_for_evaluation(test_silk)
        _, predictions, true_vals = evaluateTransformers(model, dataloader_test, device)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'F1 Score for Silk-Road dataset (Weighted): {val_f1}')
        print(classification_report(predictions, true_vals, digits=4))

    if args.intent == "traderoute-agora":
        # Evaluating on the traderoute dataset
        dataloader_test = create_data_for_evaluation(test_traderoute)
        _, predictions, true_vals = evaluateTransformers(model, dataloader_test, device)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'F1 Score for Traderoute dataset (Weighted): {val_f1}')
        print(classification_report(predictions, true_vals, digits=4))

        # Evaluating on the agora dataset
        dataloader_test = create_data_for_evaluation(test_agora)
        _, predictions, true_vals = evaluateTransformers(model, dataloader_test, device)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'F1 Score for Traderoute dataset (Weighted): {val_f1}')
        print(classification_report(predictions, true_vals, digits=4))

    if args.intent == "valhalla-berlusconi":
        # Evaluating on the Valhalla dataset
        dataloader_test = create_data_for_evaluation(test_valhalla)
        _, predictions, true_vals = evaluateTransformers(model, dataloader_test, device)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'F1 Score for Traderoute dataset (Weighted): {val_f1}')
        print(classification_report(predictions, true_vals, digits=4))

        # Evaluating on the Berlusconi dataset
        dataloader_test = create_data_for_evaluation(test_berlusconi)
        _, predictions, true_vals = evaluateTransformers(model, dataloader_test, device)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'F1 Score for Traderoute dataset (Weighted): {val_f1}')
        print(classification_report(predictions, true_vals, digits=4))

else:
    raise Exception("mode parameter can only be between train or evaluate ...")