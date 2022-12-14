"""
Python version : 3.8
Description : Trains TF-IDF based statistical models to compare different baselines.
"""


# %% Importing Libraries
import os, time
import sys
from pathlib import Path

import random
import argparse
import pickle
from timeit import main
from tqdm import tqdm

import itertools
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import plotly
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')

# Custom imports
sys.path.append('../utilities/')
from load_data import FetchData

sys.path.append('../architectures/')
from tfidf import TFIDF, evaluateTFIDF, evaluateTFIDF_and_print_classification_report

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Training TF-IDF based statistical models")
parser.add_argument('--model', type=str, default="tf-idf", help="Model to be trained (Can be tf-idf-simple, tf-idf-transformer, or bow)")
parser.add_argument('--stats_model_type', type=str, default="SVC", help="can only be one amongst MultinomialNB, MLPClassifier, LogisticRegression, RandomForestClassifier, and SVC")
parser.add_argument('--data', type=str, default='alpha-dreams', help="""Datasets to be evaluated (can be "shared", "alpha" for alphabay, "dreams", "silk" for silk-road, "alpha-dreams", "dreams-silk", "alpha-silk", or "alpha-dreams-silk")""")
parser.add_argument('--data_dir', type=str, default='../data', help="""Data directory of the pre-processed data (if your data is not pre-processed, we recommend you to go to the statistical model and process it first.)""")
parser.add_argument('--mode', type=str, default='train', help='train or evaluate')
parser.add_argument('--save_dir', type=str, default=os.path.join(os.getcwd(), "../models/statistical/"), help="""Directory for models to be saved""")
parser.add_argument('--setting', type=str, default='high', help="Low for low-resource setting and High for high-resource setting")
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--version', type=str, default='full', help='Run of small data of full data. can be ("small" or "full").')
parser.add_argument('--split_ratio', type=float, default=0.25, help="Splitting ratio for the dataset")
parser.add_argument('--n_splits', type=int, default=5, help='Number of trials to perform cross validation')
parser.add_argument('--ads_count', type=int, default=20, help="Minimum number of advertisements per vendor")
parser.add_argument('--preprocess_flag', action='store_true', help='Preprocess data')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
args = parser.parse_args()

# Creating a directory if the path doesn't exist
Path(args.save_dir).mkdir(parents=True, exist_ok=True)

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

# setting random seed
random.seed(args.seed)
np.random.seed(args.seed)

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

if args.mode == "train":
    # %% Training the shared model
    if args.data == "shared":
        train_data = pd.concat([train_alpha, train_dreams, train_silk])
        test_data = pd.concat([test_alpha, test_dreams, test_silk])

        train_data = train_data[['marketplace','vendor', 'title', 'description']].drop_duplicates()
        test_data = test_data[['marketplace','vendor', 'title', 'description']].drop_duplicates()

        # Vectorizing class labels
        data = pd.concat([train_data, test_data])
        all_vendors = list(data['vendor'].unique())
        vendor_to_idx_dict = {vendor_name:index for index, vendor_name in enumerate(all_vendors)}
        train_data['vendor'] = train_data['vendor'].replace(vendor_to_idx_dict, regex=True)
        test_data['vendor'] = test_data['vendor'].replace(vendor_to_idx_dict, regex=True)
        
        if args.model == "tf-idf":
            start_time = time.time()
            tfidf_transformer = TFIDF(train_data, test_data, args.stats_model_type, args.n_splits, args.save_dir).train_models()
            end_time = time.time()
            
            print("Time taken : ", end_time - start_time)
                        
            with open(os.path.join(args.save_dir, 'tf-idf-transformer_' + args.stats_model_type + '.pickle'), 'wb') as handle:
                pickle.dump(tfidf_transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise Exception("args.model can only be one amongst the tf-idf")

    elif args.data == "alpha-dreams":
        # Vectorizing class labels
        data = pd.concat([train_dreams, test_dreams])
        all_vendors = list(data['vendor'].unique())
        vendor_to_idx_dict = {vendor_name:index for index, vendor_name in enumerate(all_vendors)}
        train_dreams['vendor'] = train_dreams['vendor'].replace(vendor_to_idx_dict, regex=True)
        test_dreams['vendor'] = test_dreams['vendor'].replace(vendor_to_idx_dict, regex=True)

        if args.model == "tf-idf":
            start_time = time.time()
            tfidf_transformer = TFIDF(train_dreams, test_dreams, args.stats_model_type, args.n_splits, args.save_dir).train_models()
            end_time = time.time()
            print("Time taken : ", end_time - start_time)
            evaluateTFIDF_and_print_classification_report(tfidf_transformer['SVM']["model"], test_dreams, vendor_to_idx_dict)
            
            with open(os.path.join(args.save_dir, 'tf-idf-transformer-dreams_' + args.stats_model_type + '.pickle'), 'wb') as handle:
                pickle.dump(tfidf_transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise Exception("args.model can only be one amongst the tf-idf")
    # %%

elif args.mode == "evaluate":
    if args.data == "alpha-dreams":
        # Vectorizing class labels
        data = pd.concat([train_dreams, test_dreams])
        all_vendors = list(data['vendor'].unique())
        vendor_to_idx_dict = {vendor_name:index for index, vendor_name in enumerate(all_vendors)}
        
        if args.model == "tf-idf":
            if args.stats_model_type == "MultinomialNB":
                # Evaluating trained MNB
                with open("../models/tfidf_trans_mnb.pickle", "rb") as file:
                    trained_model = pickle.load(file)

            elif args.stats_model_type == "RandomForestClassifier":
                # Evaluating trained Random Forest
                with open("../models/tfidf_trans_rf.pickle", "rb") as file:
                    trained_model = pickle.load(file)

            elif args.stats_model_type == "MLPClassifier":
                # Evaluating trained Multilayer Perceptron
                with open("../models/tfidf_trans_mlp.pickle", "rb") as file:
                    trained_model = pickle.load(file)
            else:
                raise Exception("stats_model_type can only be between MultinomialNB or RandomForestClassifier")

            # Evaluating on alphabay dataset
            precision, recall, fscore = evaluateTFIDF(trained_model, test_alpha, vendor_to_idx_dict)
            print("F1 score on Alphabay Dataset", np.mean(fscore))
            # Evaluating on Silk dataset
            precision, recall, fscore = evaluateTFIDF(trained_model, test_silk, vendor_to_idx_dict)
            print("F1 score on Silk Dataset", np.mean(fscore))
            # Evaluating on Alpha-Dreams dataset
            precision, recall, fscore = evaluateTFIDF(trained_model, test_alpha_dreams, vendor_to_idx_dict)
            print("F1 score on Alpha-Dreams Dataset", np.mean(fscore))
            # Evaluating on Dreams-Silk dataset
            precision, recall, fscore = evaluateTFIDF(trained_model, test_dreams_silk, vendor_to_idx_dict)
            print("F1 score on Dreams-Silk Dataset", np.mean(fscore))
            # Evaluating on Alpha-Silk dataset
            precision, recall, fscore = evaluateTFIDF(trained_model, test_dreams_silk, vendor_to_idx_dict)
            print("F1 score on Alpha-Silk Dataset", np.mean(fscore))
            # Evaluating on Alpha-Dream-Silk dataset
            precision, recall, fscore = evaluateTFIDF(trained_model, test_alpha_dreams_silk, vendor_to_idx_dict)
            print("F1 score on Alpha-Dreams-Silk Dataset", np.mean(fscore))

else:
    raise Exception("mode can only be between train or evaluate")