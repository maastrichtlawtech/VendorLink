"""
Python version : 3.8
Description : Finetunes pre-trained BERT-base checkpoint for Language Task on Dark-Web data. The prefix dark represents that the model
              has been pre-trained on the top of available checkpoint for the language task on Darknet market advertisements. 
"""

# %% Importing Libraries
import os, sys
import random, math
import warnings
import argparse
from pathlib import Path
import logging

import numpy as np
import pandas as pd

import torch

from transformers import (TrainingArguments, Trainer, set_seed)

# Loading the custom library
sys.path.append('../utilities/')
from load_data import FetchData
from utils import save_data_to_text_file

sys.path.append('../architectures/')
from langModel import ModelDataArguments, get_model_config, get_tokenizer, get_model, get_dataset, get_collator

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Finetuning BERT checkpoint for Language Task on Dark-Web data")
parser.add_argument('--mode', type=str, default="train", help="Can be train or evaluate")
parser.add_argument('--model', type=str, default="darkBERT", help="name of the model to be trained")
parser.add_argument('--data',  type=str, default='alpha-dreams', help="""Datasets to be evaluated (can be "shared", "alpha" for alphabay, "dreams", "silk" for silk-road, "alpha-dreams", "dreams-silk", "alpha-silk", or "alpha-dreams-silk")""")
parser.add_argument('--data_dir', type=str, default='../data', help="""Data directory of the pre-processed data (if your data is not pre-processed, we recommend you to go to the statistical model and process it first.)""")
parser.add_argument('--save_dir', type=str, default=os.path.join(os.getcwd(), "../models/"), help="""Directory for models to be saved""")
parser.add_argument('-lr', type=float, default=4e-5, help="learning rate")
parser.add_argument('--batch_size', type=int, default=64, help="Batch Size")
parser.add_argument('--nb_epochs', type=int, default=5, help="Number of Epochs")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
parser.add_argument('--masking', type=float, default=0.15, help="Probabilities to be masked from the input array")
parser.add_argument('--permutation', type=float, default=float(1/6), help="The ratio of length of a span of masked tokens to surrounding context length for permutation language modeling.")
parser.add_argument('-dropout', type=float, default=0.65, help="Dropout for the model")
parser.add_argument('--version', type=str, default='full', help='Run of small data of full data. can be ("small" or "full").')
parser.add_argument('--setting', type=str, default='high', help="Low for low-resource setting and High for high-resource setting")
parser.add_argument("--min_occurence", type=int, default=10, help="Minimum number of tokens present in the data to add it to the BERT toknizer")
parser.add_argument('--split_ratio', type=float, default=0.25, help="Splitting ratio for the dataset")
parser.add_argument('--preprocess_flag', action='store_true', help='Preprocess data')
parser.add_argument('--data_to_text_flag', action='store_true', help="Converts processed data into a text file")
parser.add_argument('--ads_count', type=int, default=40, help="Minimum number of advertisements per vendor")
args_parser = parser.parse_args()

logging.basicConfig(level=logging.ERROR)
# Creating a directory if the path doesn't exist
Path(os.path.join(args_parser.save_dir, args_parser.model)).mkdir(parents=True, exist_ok=True)

# setting random seed
# pl.seed_everything(args.seed)
random.seed(args_parser.seed)
np.random.seed(args_parser.seed)
torch.manual_seed(args_parser.seed)
torch.cuda.manual_seed_all(args_parser.seed)
set_seed(args_parser.seed)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args_parser.data_to_text_flag == True:
    # %% Loading the datasets
    alpha_df = pd.read_csv(os.path.join(args_parser.data_dir, "preprocessed_alpha_new.csv"), error_bad_lines=False, 
                                lineterminator='\n', usecols=['marketplace', 'title', 'vendor', 'prediction', 'ships_to', 'ships_from', 'description']).drop_duplicates()
    dreams_df = pd.read_csv(os.path.join(args_parser.data_dir, "preprocessed_dreams_new.csv"), error_bad_lines=False, 
                                lineterminator='\n', usecols=['marketplace', 'title', 'vendor', 'prediction', 'ships_to', 'ships_from', 'description']).drop_duplicates()
    silk_df = pd.read_csv(os.path.join(args_parser.data_dir, "preprocessed_silk_new.csv"), error_bad_lines=False, 
                                lineterminator='\n', usecols=['marketplace', 'title', 'vendor', 'prediction', 'ships_to', 'ships_from', 'description']).drop_duplicates()
    data_df = {"alpha":alpha_df, "dreams":dreams_df, "silk":silk_df}

    # Splitting the data
    # %% Loading data
    if args_parser.data == "shared":
        if args_parser.setting == "high":
            [(train_alpha, train_dreams, train_silk), (test_alpha, test_dreams, test_silk)] = FetchData(data_df, args_parser.version, args_parser.data, args_parser.split_ratio, args_parser.preprocess_flag, args_parser.setting, args_parser.ads_count,  args_parser.seed).split_data()
        else:
            [(train_valhalla, train_traderoute, train_berlusconi), (test_valhalla, test_traderoute, test_berlusconi)] = FetchData(data_df, args_parser.version, args_parser.data, args_parser.split_ratio, args_parser.preprocess_flag, args_parser.setting, args_parser.ads_count,  args_parser.seed).split_data()
    elif args_parser.data == "alpha" or args_parser.data == "dreams" or args_parser.data == "silk":
        [(train_data, train_alpha, train_dreams, train_silk), (test_data, test_alpha, test_dreams, test_silk)] = FetchData(data_df, args_parser.version, args_parser.data, args_parser.split_ratio, args_parser.preprocess_flag, args_parser.setting, args_parser.ads_count, args_parser.seed).split_data()
    elif args_parser.data == "valhalla" or args_parser.data == "traderoute" or args_parser.data == "berlusconi":
        [(train_data, train_valhalla, train_traderoute, train_berlusconi), (test_data, test_valhalla, test_traderoute, test_berlusconi)]  = FetchData(data_df, args_parser.version, args_parser.data, args_parser.split_ratio, args_parser.preprocess_flag, args_parser.setting, args_parser.ads_count, args_parser.seed).split_data()

    elif args_parser.data == "alpha-dreams" or args_parser.data == "dreams-silk" or args_parser.data == "alpha-silk":
        [(train_alpha_dreams, train_dreams_silk, train_alpha_silk, train_alpha_dreams_silk, train_alpha, train_dreams, train_silk), (test_alpha_dreams, test_dreams_silk, test_alpha_silk, test_alpha_dreams_silk, test_alpha, test_dreams, test_silk)] = FetchData(data_df, args_parser.version, args_parser.data,  args_parser.split_ratio, args_parser.preprocess_flag, args_parser.setting, args_parser.ads_count,  args_parser.seed).split_data()
    elif args_parser.data == "valhalla-traderoute" or args_parser.data == "traderoute-berlusconi" or args_parser.data == "valhalla-berlusconi":
        [(train_valhalla_traderoute, train_traderoute_berlusconi, train_valhalla_berlusconi, train_valhalla_traderoute_berlusconi, train_valhalla, train_traderoute, train_berlusconi), (test_valhalla_traderoute, test_traderoute_berlusconi, test_valhalla_berlusconi, test_valhalla_traderoute_berlusconi, test_valhalla, test_traderoute, test_berlusconi)] = FetchData(data_df, args_parser.version, args_parser.data,  args_parser.split_ratio, args_parser.preprocess_flag, args_parser.setting, args_parser.ads_count,  args_parser.seed).split_data()

    else:
        raise Exception("""Datasets to be evaluated (can be "shared" for shared vendors across different markets, "alpha" for alphabay, "dreams", "silk" for silk-road, "alpha-dreams", "traderoute-berlusconi", "valhalla-traderoute", "valhalla-berlusconi", "dreams-silk", "alpha-silk", or "alpha-dreams-silk")""")

    if args_parser.data == "alpha-dreams":
        # Concatinating the data
        train_data = pd.concat([train_dreams, train_alpha, train_silk])
        test_data = pd.concat([test_dreams, test_alpha, test_silk])
        save_data_to_text_file(train_data, test_data, args_parser.data_dir)
    else:
        raise Exception("Pipeline not Implemented ....")

# %% Parameters Setup
# Define arguments for data, tokenizer and model arguments.
# See comments in `ModelDataArguments` class.
model_data_args = ModelDataArguments(
                                    train_data_file=os.path.join(args_parser.data_dir, 'dw_train_lm.txt'), 
                                    eval_data_file=os.path.join(args_parser.data_dir, 'dw_test_lm.txt'), 
                                    line_by_line=True, 
                                    mlm=True,
                                    whole_word_mask=True,
                                    mlm_probability=args_parser.masking,
                                    plm_probability=args_parser.permutation, 
                                    max_span_length=5,
                                    block_size=-1, 
                                    overwrite_cache=False, 
                                    model_type='bert', 
                                    model_config_name='bert-base-cased', 
                                    tokenizer_name='bert-base-cased', 
                                    model_name_or_path='bert-base-cased', 
                                    model_cache_dir=None,
                                    )

# Define arguments for training
# Note: I only used the arguments I care about. `TrainingArguments` contains
# a lot more arguments. For more details check the awesome documentation:
# https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
training_args = TrainingArguments(
                          # The output directory where the model predictions 
                          # and checkpoints will be written.
                          output_dir=os.path.join(args_parser.save_dir, args_parser.model),
 
                          # Overwrite the content of the output directory.
                          overwrite_output_dir=True,
 
                          # Whether to run training or not.
                          do_train=True, 
                           
                          # Whether to run evaluation on the dev or not.
                          do_eval=True,
                           
                          # Batch size GPU/TPU core/CPU training.
                          per_device_train_batch_size=args_parser.batch_size,
                           
                          # Batch size  GPU/TPU core/CPU for evaluation.
                          per_device_eval_batch_size=8,
 
                          # evaluation strategy to adopt during training
                          # `no`: No evaluation during training.
                          # `steps`: Evaluate every `eval_steps`.
                          # `epoch`: Evaluate every end of epoch.
                          evaluation_strategy='steps',
 
                          # How often to show logs. I will se this to 
                          # plot history loss and calculate perplexity.
                          logging_steps=10000,
 
                          # Number of update steps between two 
                          # evaluations if evaluation_strategy="steps".
                          # Will default to the same value as l
                          # logging_steps if not set.
                          eval_steps = None,
                           
                          # Set prediction loss to `True` in order to 
                          # return loss for perplexity calculation.
                          prediction_loss_only=True,
 
                          # The initial learning rate for Adam. 
                          # Defaults to 5e-5.
                          learning_rate = args_parser.lr,
 
                          # The weight decay to apply (if not zero).
                          weight_decay=0,
 
                          # Epsilon for the Adam optimizer. 
                          # Defaults to 1e-8
                          adam_epsilon = 1e-8,
 
                          # Maximum gradient norm (for gradient 
                          # clipping). Defaults to 0.
                          max_grad_norm = 1.0,
                          # Total number of training epochs to perform 
                          # (if not an integer, will perform the 
                          # decimal part percents of
                          # the last epoch before stopping training).
                          num_train_epochs = args_parser.nb_epochs,
 
                          # Number of updates steps before two checkpoint saves. 
                          # Defaults to 500
                          save_steps = -1,
                          )


# %% Load Configuration, Tokenizer and Model
# Load model configuration.
print('Loading model configuration...')
config = get_model_config(model_data_args)
 
# Load model tokenizer.
print('Loading model`s tokenizer...')
tokenizer = get_tokenizer(model_data_args)
# tokenizer = add_tokens_to_vocabulary(tokenizer, os.path.join(args_parser.data_dir, 'dw_train_lm.txt'), os.path.join(args_parser.data_dir, 'dw_test_lm.txt'), args_parser.min_occurence)
# tokenizer = Tokenizer(BPE())
# trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
# tokenizer.train(files=[os.path.join("../data", 'dw_train_lm.txt'), os.path.join("../data", 'dw_test_lm.txt')], trainer=trainer)

# Loading model.
print('Loading actual model...')
model = get_model(model_data_args, config)
    
# Resize model to fit all tokens in tokenizer.
model.resize_token_embeddings(len(tokenizer.get_vocab()))
model.to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total parameters :", total_params)

print("Model Description : ", model)

# %% Dataset and Collator
# Setup train dataset if `do_train` is set.
print('Creating train dataset...')
train_dataset = get_dataset(model_data_args, tokenizer=tokenizer, evaluate=False) if training_args.do_train else None
 
# Setup evaluation dataset if `do_eval` is set.
print('Creating evaluate dataset...')
eval_dataset = get_dataset(model_data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
 
# Get data collator to modify data format depending on type of model used.
data_collator = get_collator(model_data_args, config, tokenizer)
 
# Check how many logging prints you'll have. This is to avoid overflowing the 
# notebook with a lot of prints. Display warning to user if the logging steps 
# that will be displayed is larger than 100.
if (len(train_dataset) // training_args.per_device_train_batch_size // training_args.logging_steps * training_args.num_train_epochs) > 100:
    # Display warning.
    warnings.warn('Your `logging_steps` value will will do a lot of printing!' \
                ' Consider increasing `logging_steps` to avoid overflowing' \
                ' the notebook with a lot of prints!')

# %% Train model
# Initialize Trainer.
print('Loading `trainer`...')
trainer = Trainer(model=model,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=train_dataset,
                  eval_dataset=eval_dataset,
                  )
 
 
# Check model path to save.
if training_args.do_train:
    print('Start training...')

    # Setup model path if the model to train loaded from a local path.
    model_path = (model_data_args.model_name_or_path 
                if model_data_args.model_name_or_path is not None and
                os.path.isdir(model_data_args.model_name_or_path) 
                else None
                )
    # Run training.
    trainer.train(model_path=model_path)
    # Save model.
    trainer.save_model()

    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =).
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


# %% Evaluate model
# check if `do_eval` flag is set.
if training_args.do_eval:
    # capture output if trainer evaluate.
    eval_output = trainer.evaluate()
    # compute perplexity from model loss.
    perplexity = math.exp(eval_output["eval_loss"])
    print('\nEvaluate Perplexity: {:10,.2f}'.format(perplexity))
else:
    print('No evaluation needed. No evaluation data provided, `do_eval=False`!')