"""
Python version: 3.8.12
Description : calculates similarity in advertisements using traditional stylometric approaches
"""


# %% Import Libraries
from pathlib import Path

import os
import sys
import pickle
from tqdm import tqdm
import itertools
import argparse
import collections
from collections import Counter, defaultdict
from multiprocessing import Pool

import textdistance

import numpy as np
import pandas as pd

import plotly
import plotly.express as px
import plotly.graph_objects as go

# %% Loading data
print("Loading Data ....")
alpha_items_df = pd.read_csv("../data/non-anonymous/alphabay/items.csv", error_bad_lines=False, 
                      lineterminator='\n', usecols=['marketplace', 'title', 'vendor', 'prediction', 'ships_to', 'ships_from', 'description'])
alpha_feedback_df = pd.read_csv("../data/non-anonymous/alphabay/feedbacks.csv", error_bad_lines=False, 
                      lineterminator='\n', usecols=['reciever', 'order_title', 'order_amount_usd'])
alpha_feedback_df.columns = ['vendor', 'title', 'order_amount_usd']
alpha_df = alpha_items_df.merge(alpha_feedback_df, how = 'inner', on = ['title', 'vendor']).drop_duplicates()

dreams_items_df = pd.read_csv("../data/non-anonymous/dream/items.csv", error_bad_lines=False, 
                      lineterminator='\n', usecols=['marketplace', 'title', 'vendor', 'prediction', 'ships_to', 'ships_from', 'description'])
dreams_feedback_df = pd.read_csv("../data/non-anonymous/dream/feedbacks.csv", error_bad_lines=False, 
                      lineterminator='\n', usecols=['reciever', 'order_title', 'order_amount_usd'])
dreams_feedback_df.columns = ['vendor', 'title', 'order_amount_usd']
dreams_df = dreams_items_df.merge(dreams_feedback_df, how = 'inner', on = ['title', 'vendor']).drop_duplicates()
silk_df = pd.read_csv("../data/non-anonymous/silk-road/items.csv", error_bad_lines=False, 
                      lineterminator='\n', usecols=['marketplace', 'title', 'seller_id', 'category', 'ship_to', 'ship_from', 'listing_description', 'price_btc']).drop_duplicates()
silk_df.columns = ['marketplace' ,'title', 'prediction', 'order_amount_usd', 'ships_to', 'ships_from', 'vendor', 'description']
silk_df['order_amount_usd'] = silk_df['order_amount_usd'].apply(lambda x: x*54.46)

df_dict = {"alpha":alpha_df, "silk":silk_df, "dreams":dreams_df}

# %% Initializing Argparser
parser = argparse.ArgumentParser(description="Computing the distance")
parser.add_argument('--market', type=str, default="alpha", help="Market can only be alpha (for Alphabay), dreams, or silk (for Silk-road)")
parser.add_argument('--cuda', action='store_true', help='use CUDA')
args = parser.parse_args()

# %% Finding unique users
print("Finding unique vendors ....")
silk_df.vendor = silk_df.vendor.apply(lambda x : str(x).lower())
alpha_df.vendor = alpha_df.vendor.apply(lambda x : str(x).lower())
dreams_df.vendor = dreams_df.vendor.apply(lambda x : str(x).lower())

silk_vendors = list(silk_df.vendor.unique())
alpha_vendors = list(alpha_df.vendor.unique())
dreams_vendors = list(dreams_df.vendor.unique())
all_vendors = set(silk_vendors + dreams_vendors + alpha_vendors)

# %% Helper functions
def seq_distance(sequence_comb):
    sequence1, sequence2 = sequence_comb
    sequence1 = str(sequence1)
    sequence2 = str(sequence2)
    # Longest common subsequence similarity
    dist1 = textdistance.lcsseq.normalized_similarity(sequence1, sequence2)
    # Longest common substring similarity
    dist2 = textdistance.lcsstr.normalized_similarity(sequence1, sequence2)
    # Ratcliff-Obershelp similarity
    dist3 = textdistance.ratcliff_obershelp.normalized_similarity(sequence1, sequence2)
    return round((dist1 + dist2 + dist3)/3, 4)


def calculate_distance_between_shared_vendors(dir):
    alpha_df['vendor'] = alpha_df['vendor'].apply(lambda x : str(x).lower())
    dreams_df['vendor'] = dreams_df['vendor'].apply(lambda x : str(x).lower())
    silk_df['vendor'] = silk_df['vendor'].apply(lambda x : str(x).lower())

    alpha_vendors = list(alpha_df.vendor.unique())
    dreams_vendors = list(dreams_df.vendor.unique())
    silk_vendors = list(silk_df.vendor.unique())
    shared_vendors = set(alpha_vendors) & set(dreams_vendors) & set(silk_vendors)

    alpha_shared = alpha_df[alpha_df['vendor'].isin(shared_vendors)]
    dreams_shared = dreams_df[dreams_df['vendor'].isin(shared_vendors)]
    silk_shared = silk_df[silk_df['vendor'].isin(shared_vendors)]

    df = pd.concat([alpha_shared, dreams_shared, silk_shared])
    df['marketplace'] = df['marketplace'].apply(lambda x : str(x).lower())
    
    alpha_vendors = list(df[df['marketplace']=='alphabay']['vendor'].unique())
    valhalla_vendors = list(df[df['marketplace']=='valhalla']['vendor'].unique())
    dreams_vendors = list(df[df['marketplace']=='dream']['vendor'].unique())
    berlusconi_vendors = list(df[df['marketplace']=='berlusconi']['vendor'].unique())
    traderoute_vendors = list(df[df['marketplace']=='traderoute']['vendor'].unique())
    silk_vendors = list(df[df['marketplace']=='silk road 1']['vendor'].unique())

    vendor_market = {}
    adv_count_dict = dict(Counter(df['vendor']))
    adv_count_dict = {k:v for k,v in adv_count_dict.items() if v<=50}
    
    for vendor in adv_count_dict:
        temp_list = [] 
        if vendor in alpha_vendors:
            temp_list.append('alphabay')
        if vendor in valhalla_vendors:
            temp_list.append('valhalla')
        if vendor in dreams_vendors:
            temp_list.append('dreams')
        if vendor in berlusconi_vendors:
            temp_list.append('berlusconi')
        if vendor in traderoute_vendors:
            temp_list.append('traderoute')
        if vendor in silk_vendors:
            temp_list.append('silk road 1')
        else:
            pass
        vendor_market[vendor] = temp_list
        
    for vendor, markets in vendor_market.items():
        all_comb = list(itertools.combinations(markets, 2)) + [(markets[i], markets[i]) for i in range(len(markets))]
        vendor_dict = {}
        for market1, market2 in all_comb:
            temp_df = df[df['vendor']==vendor]
            temp_df1 = temp_df[temp_df['marketplace']==market1]
            temp_df2 = temp_df[temp_df['marketplace']==market2]
            if temp_df1.shape[0] + temp_df2.shape[0] > 2:
                text_list1 = extract_title_description(temp_df1)
                text_list2 = extract_title_description(temp_df2)
                if market1 != market2:
                    sequence_distance = compute_distance_within_market(text_list1, text_list2, mode='between')
                else:
                    sequence_distance = compute_distance_within_market(text_list1, text_list1)
            else:
                sequence_distance = np.array([-1.0])
                
            vendor_dict[(vendor, market1, market2)] = np.mean(sequence_distance)
        with open(os.path.join(dir, vendor + '.pickle'), 'wb') as handle:
            pickle.dump(vendor_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def edit_distance(vendor1, vendor2):
    vendor1 = str(vendor1)
    vendor2 = str(vendor2)
    # Hamming Distance
    dist1 = textdistance.hamming.normalized_similarity(vendor1, vendor2)
    # Mlipns Distance
    # dist2 = textdistance.mlipns.normalized_similarity(vendor1, vendor2)
    # Levenshtein
    dist3 = textdistance.levenshtein.normalized_similarity(vendor1, vendor2)
    # Damerau Levenshtein
    dist4 = textdistance.damerau_levenshtein.normalized_similarity(vendor1, vendor2)
    # Jaro Winkler
    dist5 = textdistance.jaro_winkler.normalized_similarity(vendor1, vendor2)
    # Strcmp95
    dist6 = textdistance.strcmp95.normalized_similarity(vendor1, vendor2)
    # Needleman-Wunsch
    dist7 = textdistance.needleman_wunsch.normalized_similarity(vendor1, vendor2)
    # Goto
    dist9 = textdistance.smith_waterman.normalized_similarity(vendor1, vendor2)

    return round((dist1 + dist3 + dist4 + dist5 + dist6 + dist7 + dist9)/7, 4)

def calculate_edit_distance(path):
    Path(path).mkdir(parents=True, exist_ok=True)

    vendor_distance = np.zeros((len(all_vendors), len(all_vendors)))
    pbar = tqdm(total=len(all_vendors))
    for i, vendor1 in enumerate(all_vendors):
        j = i
        for vendor2 in list(all_vendors)[i:]:
            vendor_distance[i][j] = edit_distance(vendor1, vendor2)
            vendor_distance[j][i] = vendor_distance[i][j]
            j += 1
        pbar.update(1)
    pbar.close()


    df = pd.DataFrame(vendor_distance, columns=all_vendors, index=all_vendors)
    df.to_csv(os.path.join(path, "vendor_distance.csv"))

def calculate_sequence_distance_within_market(path, market='alpha'):
    Path(path).mkdir(parents=True, exist_ok=True)
    
    df = df_dict[market]
    print("Calculating distance between vendors on " + market + " market")
    pbar = tqdm(total=df.vendor.nunique())
    vendor_dict = {}

    for vendor in df.vendor.unique():
        temp_df = df[df['vendor'] == vendor]
        if temp_df.shape[0] > 1:
            text_list = extract_title_description(temp_df)
            sequence_distance = compute_distance_within_market(text_list, text_list)
        else:
            sequence_distance = np.array([-1.0])
        
        vendor_dict[vendor] = np.mean(sequence_distance)
        pbar.update(1)
    pbar.close()
    
    df = pd.DataFrame(list(vendor_dict.values()),columns = [market], index=list(df.vendor.unique()))
    df.to_csv(os.path.join(path, "vendor_" + market + "_distance.csv"))
    
def compute_distance_within_market(text_list1, text_list2, mode='within'):
    if mode == 'between':
        all_combinations = [list(zip(each_permutation, text_list2)) for each_permutation in itertools.permutations(text_list1, len(text_list2))]
        all_combinations = [item for sublist in all_combinations for item in sublist]
        all_combinations = set([value for value in all_combinations if value[0]!=value[1]])
    elif mode == 'within':
        all_combinations = [list(x) for x in itertools.combinations(text_list1, 2)]
    p = Pool()
    sequence_distance = p.map(seq_distance, all_combinations)
    p.close()
    return sequence_distance

def extract_title_description(df):
    title_text = list(df['title'])
    title_text = [str(title) for title in title_text]
    description_text = list(df['description'])
    description_text = [str(description) for description in description_text]
    return [title_text[i] + ' [SEP] ' + description_text[i] for i in range(df.shape[0])]

# %%
if __name__ == "__main__":
    Path(os.path.join(os.getcwd(), "../data/processed_data/distance/")).mkdir(parents=True, exist_ok=True)
    dir = os.path.join(os.getcwd(), "../data/processed_data/distance/")
    calculate_distance_between_shared_vendors(dir)
    calculate_sequence_distance_within_market(dir, args.market)
    calculate_edit_distance(dir)