################################################################################
########################## Importing libraries ##################################
print("Importing libraries...")
import os
import argparse
import re
import numpy as np
import pandas as pd
from pandas import read_excel
from tqdm import tqdm
import csv
import pickle
import collections
from collections import Counter

import emoji
import itertools

import spacy
from spacy_cld import LanguageDetector
import contextualSpellCheck

import matplotlib.pyplot as plt

import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings('ignore')

nlp = spacy.load('en_core_web_sm')
# language_detector = LanguageDetector()
# nlp.add_pipe(language_detector)
################################################################################


################################################################################
############################# Initializing the parser ##########################
print("Initializing the parser...")
parser = argparse.ArgumentParser(description="Generate POS and NER distribution")

parser.add_argument('--mode', type=str, default="alphabay-dreams", 
                    help="Can be alphabay-dreams, dreams-silk, or alphabay-silk")
args = parser.parse_args()
################################################################################


################################################################################
################################### Loading the datasets #######################
print("Loading the datasets...")
alpha_listing_df = pd.read_csv("../../../sybil-identification/data/non-anonymous/alphabay/items.csv", error_bad_lines=False, 
                      lineterminator='\n', usecols=['title', 'vendor', 'first_observed', 'last_observed', 'prediction', 'total_sales', 'ships_to', 'ships_from', 'description'])

alpha_feedback_df = pd.read_csv("../../../sybil-identification/data/non-anonymous/alphabay/feedbacks.csv", error_bad_lines=False, 
                      lineterminator='\n', usecols=['reciever', 'message', 'order_title', 'order_amount_usd'])
alpha_feedback_df.columns = ['vendor', 'message', 'title', 'order_amount_usd']

alphabay_df = alpha_listing_df.merge(alpha_feedback_df, how = 'inner', on = ['title', 'vendor'])

dreams_listing_df = pd.read_csv("../../../sybil-identification/data/non-anonymous/dream/items.csv", error_bad_lines=False, 
                      lineterminator='\n', usecols=['title', 'vendor', 'first_observed', 'last_observed', 'prediction', 'total_sales', 'ships_to', 'ships_from', 'description'])

dreams_feedback_df = pd.read_csv("../../../sybil-identification/data/non-anonymous/dream/feedbacks.csv", error_bad_lines=False, 
                      lineterminator='\n', usecols=['reciever', 'message', 'order_title', 'order_amount_usd'])
dreams_feedback_df.columns = ['vendor', 'message', 'title', 'order_amount_usd']

dreams_df = dreams_feedback_df.merge(dreams_listing_df, how = 'inner', on = ['title', 'vendor'])

silk_listing_df = pd.read_csv("../../../sybil-identification/data/non-anonymous/silk-road/items.csv", error_bad_lines=False, 
                      lineterminator='\n', usecols=['title', 'seller_id', 'category', 'ship_to', 'ship_from', 'listing_description', 'price_btc'])
# silk_listing_df.columns = ['title', 'vendor', 'prediction', 'ships_to', 'ships_from', 'description', 'price_btc']
silk_listing_df.columns = ['title', 'prediction', 'price_btc', 'ships_to', 'ships_from', 'vendor', 'description']
#################################################################################


################################################################################
################################### Merging dataset ############################
print("Merging dataset...")
alphabay_df_temp = alphabay_df.copy()
alphabay_df_temp.drop(columns=['first_observed', 'last_observed', 'message', 'total_sales'], inplace=True)
# alphabay_df_temp.drop_duplicates(inplace=True)

silk_listing_df['price_btc'] = silk_listing_df['price_btc'].apply(lambda x : float(x*57.95))
silk_listing_df.rename(columns={"price_btc": "order_amount_usd"}, inplace=True)
silk_listing_df.drop_duplicates(inplace=True)

alpha_silk = pd.concat([alphabay_df_temp, silk_listing_df], axis=0)

dreams_df_temp = dreams_df.copy()
dreams_df_temp.drop(columns=['first_observed', 'last_observed', 'message', 'total_sales'], inplace=True)

dreams_silk = pd.concat([dreams_df_temp, silk_listing_df], axis=0)
################################################################################

def clean_data(text):
    text = text.replace("♕","kingPieceEmoji ").replace("★","starEmoji ")
    text = text.replace("\r", " \r ").replace("\n", " \n ")
    return text

################################################################################
#################### Finding Sybils in Alphabay and Dreams dataset ##############
if args.mode == "alphabay-dreams":
    print("Finding Sybils in Alphabay and Dreams dataset...")
    alphabay_df = alphabay_df_temp.copy()
    dreams_df = dreams_df_temp.copy()
    ad_sybils = set(alphabay_df['vendor'].unique()).intersection(set(dreams_df['vendor'].unique()))
    ad_sybils = [str(sybil).lower() for sybil in ad_sybils]
    alpha_dreams = pd.concat([alphabay_df, dreams_df], axis=0)
    alpha_dreams.drop_duplicates(inplace=True)

    vendor_list = list(alpha_dreams['vendor'])
    vendor_list = [str(vendor).lower() for vendor in vendor_list]

    for index, vendor in enumerate(vendor_list):
        if vendor in ad_sybils:
            pass
        else:
            vendor_list[index] = 'others'

    alpha_dreams_stats = dict(Counter(vendor_list))
    del alpha_dreams_stats['others']
    alpha_dreams_stats = {k: v for k, v in sorted(alpha_dreams_stats.items(), key=lambda item: item[1], reverse=True)}

    most_active_user = Counter(alpha_dreams_stats).most_common(1)[0][0]
    least_active_user = Counter(alpha_dreams_stats).most_common()[-1][0]
    moderately_active_user = 'thecheekygirls1'

    #------------------------------ Most Active user -------------------------------#
    # Most active Alphabay user
    alphabay_df['vendor'] = alphabay_df['vendor'].apply(lambda x : str(x).lower())
    temp_alphabay_df = alphabay_df[alphabay_df['vendor']==most_active_user]
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : str(x).lower())
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : str(x).lower())
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : emoji.demojize(x))
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : emoji.demojize(x))
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : clean_data(x))
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : clean_data(x))

    # Most active Dreams user
    dreams_df['vendor'] = dreams_df['vendor'].apply(lambda x : str(x).lower())
    temp_dreams_df = dreams_df[dreams_df['vendor']==most_active_user]
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : str(x).lower())
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : str(x).lower())
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : emoji.demojize(x))
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : emoji.demojize(x))
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : clean_data(x))
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : clean_data(x))

    alpha_title = list(temp_alphabay_df['title'])
    dreams_title = list(temp_dreams_df['title'])
    alpha_description = list(temp_alphabay_df['description'])
    dreams_description = list(temp_dreams_df['description'])

    # Generating the title data
    title_doc = [nlp(title) for title in alpha_title]
    data1_title_text = [[token.text for token in title] for title in title_doc]
    data1_title_text = [item for sublist in data1_title_text for item in sublist]
    data1_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_pos = [item for sublist in data1_title_pos for item in sublist]
    data1_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_dep = [item for sublist in data1_title_dep for item in sublist]
    data1_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data1_title_stop = [item for sublist in data1_title_stop for item in sublist]
    title_doc = [nlp(title) for title in dreams_title]
    data2_title_text = [[token.text for token in title] for title in title_doc]
    data2_title_text = [item for sublist in data2_title_text for item in sublist]
    data2_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_pos = [item for sublist in data2_title_pos for item in sublist]
    data2_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_dep = [item for sublist in data2_title_dep for item in sublist]
    data2_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data2_title_stop = [item for sublist in data2_title_stop for item in sublist]

    # Generating the description data
    description_doc = [nlp(description) for description in alpha_description]
    data1_description_text = [[token.text for token in description] for description in description_doc]
    data1_description_text = [item for sublist in data1_description_text for item in sublist]
    data1_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_pos = [item for sublist in data1_description_pos for item in sublist]
    data1_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_dep = [item for sublist in data1_description_dep for item in sublist]
    data1_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data1_description_stop = [item for sublist in data1_description_stop for item in sublist]
    description_doc = [nlp(description) for description in dreams_description]
    data2_description_text = [[token.text for token in description] for description in description_doc]
    data2_description_text = [item for sublist in data2_description_text for item in sublist]
    data2_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_pos = [item for sublist in data2_description_pos for item in sublist]
    data2_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_dep = [item for sublist in data2_description_dep for item in sublist]
    data2_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data2_description_stop = [item for sublist in data2_description_stop for item in sublist]

    # Creating an empty dataset
    title_df1 = pd.DataFrame(columns=['text','pos','dep','is_stop'])
    title_df2 = title_df1.copy()
    description_df1 = title_df1.copy()
    description_df2 = title_df1.copy()

    title_df1['text'] = data1_title_text
    title_df1['pos'] = data1_title_pos
    title_df1['dep'] = data1_title_dep
    title_df1['is_stop'] = data1_title_stop
    title_df2['text'] = data2_title_text
    title_df2['pos'] = data2_title_pos
    title_df2['dep'] = data2_title_dep
    title_df2['is_stop'] = data2_title_stop

    description_df1['text'] = data1_description_text
    description_df1['pos'] = data1_description_pos
    description_df1['dep'] = data1_description_dep
    description_df1['is_stop'] = data1_description_stop
    description_df2['text'] = data2_description_text
    description_df2['pos'] = data2_description_pos
    description_df2['dep'] = data2_description_dep
    description_df2['is_stop'] = data2_description_stop

    title_df1.to_csv("../../../sybil-identification/data/non-anonymous/alphabay/ad_alphabay_title_most_active_stats.csv")
    title_df2.to_csv("../../../sybil-identification/data/non-anonymous/dream/ad_dreams_title_most_active_stats.csv")
    description_df1.to_csv("../../../sybil-identification/data/non-anonymous/alphabay/ad_alphabay_description_most_active_stats.csv")
    description_df2.to_csv("../../../sybil-identification/data/non-anonymous/dream/ad_dreams_description_most_active_stats.csv")
    # ------------------------------------------------------------------------------#

    #------------------------------ moderately Active user -------------------------------#
    # moderately active Alphabay user
    temp_alphabay_df = alphabay_df[alphabay_df['vendor']==moderately_active_user]
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : str(x).lower())
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : str(x).lower())
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : emoji.demojize(x))
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : emoji.demojize(x))
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : clean_data(x))
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : clean_data(x))

    # moderately active Dreams user
    temp_dreams_df = dreams_df[dreams_df['vendor']==moderately_active_user]
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : str(x).lower())
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : str(x).lower())
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : emoji.demojize(x))
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : emoji.demojize(x))
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : clean_data(x))
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : clean_data(x))

    alpha_title = list(temp_alphabay_df['title'])
    dreams_title = list(temp_dreams_df['title'])
    alpha_description = list(temp_alphabay_df['description'])
    dreams_description = list(temp_dreams_df['description'])

    # Generating the title data
    title_doc = [nlp(title) for title in alpha_title]
    data1_title_text = [[token.text for token in title] for title in title_doc]
    data1_title_text = [item for sublist in data1_title_text for item in sublist]
    data1_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_pos = [item for sublist in data1_title_pos for item in sublist]
    data1_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_dep = [item for sublist in data1_title_dep for item in sublist]
    data1_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data1_title_stop = [item for sublist in data1_title_stop for item in sublist]
    title_doc = [nlp(title) for title in dreams_title]
    data2_title_text = [[token.text for token in title] for title in title_doc]
    data2_title_text = [item for sublist in data2_title_text for item in sublist]
    data2_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_pos = [item for sublist in data2_title_pos for item in sublist]
    data2_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_dep = [item for sublist in data2_title_dep for item in sublist]
    data2_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data2_title_stop = [item for sublist in data2_title_stop for item in sublist]

    # Generating the description data
    description_doc = [nlp(description) for description in alpha_description]
    data1_description_text = [[token.text for token in description] for description in description_doc]
    data1_description_text = [item for sublist in data1_description_text for item in sublist]
    data1_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_pos = [item for sublist in data1_description_pos for item in sublist]
    data1_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_dep = [item for sublist in data1_description_dep for item in sublist]
    data1_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data1_description_stop = [item for sublist in data1_description_stop for item in sublist]
    description_doc = [nlp(description) for description in dreams_description]
    data2_description_text = [[token.text for token in description] for description in description_doc]
    data2_description_text = [item for sublist in data2_description_text for item in sublist]
    data2_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_pos = [item for sublist in data2_description_pos for item in sublist]
    data2_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_dep = [item for sublist in data2_description_dep for item in sublist]
    data2_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data2_description_stop = [item for sublist in data2_description_stop for item in sublist]

    # Creating an empty dataset
    title_df1 = pd.DataFrame(columns=['text','pos','dep','is_stop'])
    title_df2 = title_df1.copy()
    description_df1 = title_df1.copy()
    description_df2 = title_df1.copy()

    title_df1['text'] = data1_title_text
    title_df1['pos'] = data1_title_pos
    title_df1['dep'] = data1_title_dep
    title_df1['is_stop'] = data1_title_stop
    title_df2['text'] = data2_title_text
    title_df2['pos'] = data2_title_pos
    title_df2['dep'] = data2_title_dep
    title_df2['is_stop'] = data2_title_stop

    description_df1['text'] = data1_description_text
    description_df1['pos'] = data1_description_pos
    description_df1['dep'] = data1_description_dep
    description_df1['is_stop'] = data1_description_stop
    description_df2['text'] = data2_description_text
    description_df2['pos'] = data2_description_pos
    description_df2['dep'] = data2_description_dep
    description_df2['is_stop'] = data2_description_stop

    title_df1.to_csv("../../../sybil-identification/data/non-anonymous/alphabay/ad_alphabay_title_moderately_active_stats.csv")
    title_df2.to_csv("../../../sybil-identification/data/non-anonymous/dream/ad_dreams_title_moderately_active_stats.csv")
    description_df1.to_csv("../../../sybil-identification/data/non-anonymous/alphabay/ad_alphabay_description_moderately_active_stats.csv")
    description_df2.to_csv("../../../sybil-identification/data/non-anonymous/dream/ad_dreams_description_moderately_active_stats.csv")
    # ------------------------------------------------------------------------------#

    #------------------------------ least Active user -------------------------------#
    # least active Alphabay user
    temp_alphabay_df = alphabay_df[alphabay_df['vendor']==least_active_user]
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : str(x).lower())
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : str(x).lower())
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : emoji.demojize(x))
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : emoji.demojize(x))
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : clean_data(x))
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : clean_data(x))

    # least active Dreams user
    temp_dreams_df = dreams_df[dreams_df['vendor']==least_active_user]
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : str(x).lower())
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : str(x).lower())
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : emoji.demojize(x))
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : emoji.demojize(x))
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : clean_data(x))
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : clean_data(x))

    alpha_title = list(temp_alphabay_df['title'])
    dreams_title = list(temp_dreams_df['title'])
    alpha_description = list(temp_alphabay_df['description'])
    dreams_description = list(temp_dreams_df['description'])

    # Generating the title data
    title_doc = [nlp(title) for title in alpha_title]
    data1_title_text = [[token.text for token in title] for title in title_doc]
    data1_title_text = [item for sublist in data1_title_text for item in sublist]
    data1_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_pos = [item for sublist in data1_title_pos for item in sublist]
    data1_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_dep = [item for sublist in data1_title_dep for item in sublist]
    data1_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data1_title_stop = [item for sublist in data1_title_stop for item in sublist]
    title_doc = [nlp(title) for title in dreams_title]
    data2_title_text = [[token.text for token in title] for title in title_doc]
    data2_title_text = [item for sublist in data2_title_text for item in sublist]
    data2_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_pos = [item for sublist in data2_title_pos for item in sublist]
    data2_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_dep = [item for sublist in data2_title_dep for item in sublist]
    data2_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data2_title_stop = [item for sublist in data2_title_stop for item in sublist]

    # Generating the description data
    description_doc = [nlp(description) for description in alpha_description]
    data1_description_text = [[token.text for token in description] for description in description_doc]
    data1_description_text = [item for sublist in data1_description_text for item in sublist]
    data1_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_pos = [item for sublist in data1_description_pos for item in sublist]
    data1_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_dep = [item for sublist in data1_description_dep for item in sublist]
    data1_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data1_description_stop = [item for sublist in data1_description_stop for item in sublist]
    description_doc = [nlp(description) for description in dreams_description]
    data2_description_text = [[token.text for token in description] for description in description_doc]
    data2_description_text = [item for sublist in data2_description_text for item in sublist]
    data2_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_pos = [item for sublist in data2_description_pos for item in sublist]
    data2_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_dep = [item for sublist in data2_description_dep for item in sublist]
    data2_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data2_description_stop = [item for sublist in data2_description_stop for item in sublist]

    # Creating an empty dataset
    title_df1 = pd.DataFrame(columns=['text','pos','dep','is_stop'])
    title_df2 = title_df1.copy()
    description_df1 = title_df1.copy()
    description_df2 = title_df1.copy()

    title_df1['text'] = data1_title_text
    title_df1['pos'] = data1_title_pos
    title_df1['dep'] = data1_title_dep
    title_df1['is_stop'] = data1_title_stop
    title_df2['text'] = data2_title_text
    title_df2['pos'] = data2_title_pos
    title_df2['dep'] = data2_title_dep
    title_df2['is_stop'] = data2_title_stop

    description_df1['text'] = data1_description_text
    description_df1['pos'] = data1_description_pos
    description_df1['dep'] = data1_description_dep
    description_df1['is_stop'] = data1_description_stop
    description_df2['text'] = data2_description_text
    description_df2['pos'] = data2_description_pos
    description_df2['dep'] = data2_description_dep
    description_df2['is_stop'] = data2_description_stop

    title_df1.to_csv("../../../sybil-identification/data/non-anonymous/alphabay/ad_alphabay_title_least_active_stats.csv")
    title_df2.to_csv("../../../sybil-identification/data/non-anonymous/dream/ad_dreams_title_least_active_stats.csv")
    description_df1.to_csv("../../../sybil-identification/data/non-anonymous/alphabay/ad_alphabay_description_least_active_stats.csv")
    description_df2.to_csv("../../../sybil-identification/data/non-anonymous/dream/ad_dreams_description_least_active_stats.csv")
    # ------------------------------------------------------------------------------#
#################################################################################


################################################################################
#################### Finding Sybils in Dreams and Silk-Road dataset ##############

elif args.mode == "dreams-silk":
    print("Finding Sybils in Dreams and Silk-Road dataset...")
    silk_df = silk_listing_df.copy()
    dreams_df = dreams_df_temp.copy()
    ds_sybils = set(silk_df['vendor'].unique()).intersection(set(dreams_df['vendor'].unique()))
    ds_sybils = [str(sybil).lower() for sybil in ds_sybils]

    vendor_list = list(dreams_silk['vendor'])
    vendor_list = [str(vendor).lower() for vendor in vendor_list]

    for index, vendor in enumerate(vendor_list):
        if vendor in ds_sybils:
            pass
        else:
            vendor_list[index] = 'others'

    dream_silk_stats = dict(Counter(vendor_list))
    del dream_silk_stats['others']
    dream_silk_stats = {k: v for k, v in sorted(dream_silk_stats.items(), key=lambda item: item[1], reverse=True)}

    most_active_user = Counter(dream_silk_stats).most_common(1)[0][0]
    least_active_user = Counter(dream_silk_stats).most_common()[-1][0]
    moderately_active_user = Counter(dream_silk_stats).most_common()[int(len(Counter(dream_silk_stats).most_common())/2)][0]

    #------------------------------ Most Active user -------------------------------#
    # Most active silk user
    silk_df['vendor'] = silk_df['vendor'].apply(lambda x : str(x).lower())
    temp_silk_df = silk_df[silk_df['vendor']==most_active_user]
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : str(x).lower())
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : str(x).lower())
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : emoji.demojize(x))
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : emoji.demojize(x))
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : clean_data(x))
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : clean_data(x))

    # Most active Dreams user
    dreams_df['vendor'] = dreams_df['vendor'].apply(lambda x : str(x).lower())
    temp_dreams_df = dreams_df[dreams_df['vendor']==most_active_user]
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : str(x).lower())
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : str(x).lower())
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : emoji.demojize(x))
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : emoji.demojize(x))
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : clean_data(x))
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : clean_data(x))

    silk_title = list(temp_silk_df['title'])
    dreams_title = list(temp_dreams_df['title'])
    silk_description = list(temp_silk_df['description'])
    dreams_description = list(temp_dreams_df['description'])

    # Generating the title data
    title_doc = [nlp(title) for title in dreams_title]
    data1_title_text = [[token.text for token in title] for title in title_doc]
    data1_title_text = [item for sublist in data1_title_text for item in sublist]
    data1_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_pos = [item for sublist in data1_title_pos for item in sublist]
    data1_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_dep = [item for sublist in data1_title_dep for item in sublist]
    data1_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data1_title_stop = [item for sublist in data1_title_stop for item in sublist]
    title_doc = [nlp(title) for title in silk_title]
    data2_title_text = [[token.text for token in title] for title in title_doc]
    data2_title_text = [item for sublist in data2_title_text for item in sublist]
    data2_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_pos = [item for sublist in data2_title_pos for item in sublist]
    data2_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_dep = [item for sublist in data2_title_dep for item in sublist]
    data2_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data2_title_stop = [item for sublist in data2_title_stop for item in sublist]

    # Generating the description data
    description_doc = [nlp(description) for description in dreams_description]
    data1_description_text = [[token.text for token in description] for description in description_doc]
    data1_description_text = [item for sublist in data1_description_text for item in sublist]
    data1_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_pos = [item for sublist in data1_description_pos for item in sublist]
    data1_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_dep = [item for sublist in data1_description_dep for item in sublist]
    data1_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data1_description_stop = [item for sublist in data1_description_stop for item in sublist]
    description_doc = [nlp(description) for description in silk_description]
    data2_description_text = [[token.text for token in description] for description in description_doc]
    data2_description_text = [item for sublist in data2_description_text for item in sublist]
    data2_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_pos = [item for sublist in data2_description_pos for item in sublist]
    data2_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_dep = [item for sublist in data2_description_dep for item in sublist]
    data2_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data2_description_stop = [item for sublist in data2_description_stop for item in sublist]

    # Creating an empty dataset
    title_df1 = pd.DataFrame(columns=['text','pos','dep','is_stop'])
    title_df2 = title_df1.copy()
    description_df1 = title_df1.copy()
    description_df2 = title_df1.copy()

    title_df1['text'] = data1_title_text
    title_df1['pos'] = data1_title_pos
    title_df1['dep'] = data1_title_dep
    title_df1['is_stop'] = data1_title_stop
    title_df2['text'] = data2_title_text
    title_df2['pos'] = data2_title_pos
    title_df2['dep'] = data2_title_dep
    title_df2['is_stop'] = data2_title_stop

    description_df1['text'] = data1_description_text
    description_df1['pos'] = data1_description_pos
    description_df1['dep'] = data1_description_dep
    description_df1['is_stop'] = data1_description_stop
    description_df2['text'] = data2_description_text
    description_df2['pos'] = data2_description_pos
    description_df2['dep'] = data2_description_dep
    description_df2['is_stop'] = data2_description_stop

    title_df2.to_csv("../../../sybil-identification/data/non-anonymous/silk-road/ds_silk_title_most_active_stats.csv")
    title_df1.to_csv("../../../sybil-identification/data/non-anonymous/dream/ds_dreams_title_most_active_stats.csv")
    description_df2.to_csv("../../../sybil-identification/data/non-anonymous/silk-road/ds_silk_description_most_active_stats.csv")
    description_df1.to_csv("../../../sybil-identification/data/non-anonymous/dream/ds_dreams_description_most_active_stats.csv")
    # ------------------------------------------------------------------------------#

    #------------------------------ moderately Active user -------------------------------#
    # moderately active silk user
    temp_silk_df = silk_df[silk_df['vendor']==moderately_active_user]
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : str(x).lower())
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : str(x).lower())
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : emoji.demojize(x))
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : emoji.demojize(x))
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : clean_data(x))
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : clean_data(x))

    # moderately active Dreams user
    temp_dreams_df = dreams_df[dreams_df['vendor']==moderately_active_user]
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : str(x).lower())
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : str(x).lower())
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : emoji.demojize(x))
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : emoji.demojize(x))
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : clean_data(x))
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : clean_data(x))

    silk_title = list(temp_silk_df['title'])
    dreams_title = list(temp_dreams_df['title'])
    silk_description = list(temp_silk_df['description'])
    dreams_description = list(temp_dreams_df['description'])

    # Generating the title data
    title_doc = [nlp(title) for title in silk_title]
    data1_title_text = [[token.text for token in title] for title in title_doc]
    data1_title_text = [item for sublist in data1_title_text for item in sublist]
    data1_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_pos = [item for sublist in data1_title_pos for item in sublist]
    data1_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_dep = [item for sublist in data1_title_dep for item in sublist]
    data1_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data1_title_stop = [item for sublist in data1_title_stop for item in sublist]
    title_doc = [nlp(title) for title in dreams_title]
    data2_title_text = [[token.text for token in title] for title in title_doc]
    data2_title_text = [item for sublist in data2_title_text for item in sublist]
    data2_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_pos = [item for sublist in data2_title_pos for item in sublist]
    data2_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_dep = [item for sublist in data2_title_dep for item in sublist]
    data2_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data2_title_stop = [item for sublist in data2_title_stop for item in sublist]

    # Generating the description data
    description_doc = [nlp(description) for description in silk_description]
    data1_description_text = [[token.text for token in description] for description in description_doc]
    data1_description_text = [item for sublist in data1_description_text for item in sublist]
    data1_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_pos = [item for sublist in data1_description_pos for item in sublist]
    data1_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_dep = [item for sublist in data1_description_dep for item in sublist]
    data1_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data1_description_stop = [item for sublist in data1_description_stop for item in sublist]
    description_doc = [nlp(description) for description in dreams_description]
    data2_description_text = [[token.text for token in description] for description in description_doc]
    data2_description_text = [item for sublist in data2_description_text for item in sublist]
    data2_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_pos = [item for sublist in data2_description_pos for item in sublist]
    data2_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_dep = [item for sublist in data2_description_dep for item in sublist]
    data2_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data2_description_stop = [item for sublist in data2_description_stop for item in sublist]

    # Creating an empty dataset
    title_df1 = pd.DataFrame(columns=['text','pos','dep','is_stop'])
    title_df2 = title_df1.copy()
    description_df1 = title_df1.copy()
    description_df2 = title_df1.copy()

    title_df1['text'] = data1_title_text
    title_df1['pos'] = data1_title_pos
    title_df1['dep'] = data1_title_dep
    title_df1['is_stop'] = data1_title_stop
    title_df2['text'] = data2_title_text
    title_df2['pos'] = data2_title_pos
    title_df2['dep'] = data2_title_dep
    title_df2['is_stop'] = data2_title_stop

    description_df1['text'] = data1_description_text
    description_df1['pos'] = data1_description_pos
    description_df1['dep'] = data1_description_dep
    description_df1['is_stop'] = data1_description_stop
    description_df2['text'] = data2_description_text
    description_df2['pos'] = data2_description_pos
    description_df2['dep'] = data2_description_dep
    description_df2['is_stop'] = data2_description_stop

    title_df1.to_csv("../../../sybil-identification/data/non-anonymous/silk-road/ds_silk_title_moderately_active_stats.csv")
    title_df2.to_csv("../../../sybil-identification/data/non-anonymous/dream/ds_dreams_title_moderately_active_stats.csv")
    description_df1.to_csv("../../../sybil-identification/data/non-anonymous/silk-road/ds_silk_description_moderately_active_stats.csv")
    description_df2.to_csv("../../../sybil-identification/data/non-anonymous/dream/ds_dreams_description_moderately_active_stats.csv")
    # ------------------------------------------------------------------------------#

    #------------------------------ least Active user -------------------------------#
    # least active silk user
    temp_silk_df = silk_df[silk_df['vendor']==least_active_user]
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : str(x).lower())
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : str(x).lower())
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : emoji.demojize(x))
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : emoji.demojize(x))
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : clean_data(x))
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : clean_data(x))

    # least active Dreams user
    temp_dreams_df = dreams_df[dreams_df['vendor']==least_active_user]
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : str(x).lower())
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : str(x).lower())
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : emoji.demojize(x))
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : emoji.demojize(x))
    temp_dreams_df['title'] = temp_dreams_df['title'].apply(lambda x : clean_data(x))
    temp_dreams_df['description'] = temp_dreams_df['description'].apply(lambda x : clean_data(x))

    silk_title = list(temp_silk_df['title'])
    dreams_title = list(temp_dreams_df['title'])
    silk_description = list(temp_silk_df['description'])
    dreams_description = list(temp_dreams_df['description'])

    # Generating the title data
    title_doc = [nlp(title) for title in silk_title]
    data1_title_text = [[token.text for token in title] for title in title_doc]
    data1_title_text = [item for sublist in data1_title_text for item in sublist]
    data1_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_pos = [item for sublist in data1_title_pos for item in sublist]
    data1_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_dep = [item for sublist in data1_title_dep for item in sublist]
    data1_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data1_title_stop = [item for sublist in data1_title_stop for item in sublist]
    title_doc = [nlp(title) for title in dreams_title]
    data2_title_text = [[token.text for token in title] for title in title_doc]
    data2_title_text = [item for sublist in data2_title_text for item in sublist]
    data2_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_pos = [item for sublist in data2_title_pos for item in sublist]
    data2_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_dep = [item for sublist in data2_title_dep for item in sublist]
    data2_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data2_title_stop = [item for sublist in data2_title_stop for item in sublist]

    # Generating the description data
    description_doc = [nlp(description) for description in silk_description]
    data1_description_text = [[token.text for token in description] for description in description_doc]
    data1_description_text = [item for sublist in data1_description_text for item in sublist]
    data1_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_pos = [item for sublist in data1_description_pos for item in sublist]
    data1_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_dep = [item for sublist in data1_description_dep for item in sublist]
    data1_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data1_description_stop = [item for sublist in data1_description_stop for item in sublist]
    description_doc = [nlp(description) for description in dreams_description]
    data2_description_text = [[token.text for token in description] for description in description_doc]
    data2_description_text = [item for sublist in data2_description_text for item in sublist]
    data2_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_pos = [item for sublist in data2_description_pos for item in sublist]
    data2_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_dep = [item for sublist in data2_description_dep for item in sublist]
    data2_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data2_description_stop = [item for sublist in data2_description_stop for item in sublist]

    # Creating an empty dataset
    title_df1 = pd.DataFrame(columns=['text','pos','dep','is_stop'])
    title_df2 = title_df1.copy()
    description_df1 = title_df1.copy()
    description_df2 = title_df1.copy()

    title_df1['text'] = data1_title_text
    title_df1['pos'] = data1_title_pos
    title_df1['dep'] = data1_title_dep
    title_df1['is_stop'] = data1_title_stop
    title_df2['text'] = data2_title_text
    title_df2['pos'] = data2_title_pos
    title_df2['dep'] = data2_title_dep
    title_df2['is_stop'] = data2_title_stop

    description_df1['text'] = data1_description_text
    description_df1['pos'] = data1_description_pos
    description_df1['dep'] = data1_description_dep
    description_df1['is_stop'] = data1_description_stop
    description_df2['text'] = data2_description_text
    description_df2['pos'] = data2_description_pos
    description_df2['dep'] = data2_description_dep
    description_df2['is_stop'] = data2_description_stop

    title_df1.to_csv("../../../sybil-identification/data/non-anonymous/silk-road/ds_silk_title_least_active_stats.csv")
    title_df2.to_csv("../../../sybil-identification/data/non-anonymous/dream/ds_dreams_title_least_active_stats.csv")
    description_df1.to_csv("../../../sybil-identification/data/non-anonymous/silk-road/ds_silk_description_least_active_stats.csv")
    description_df2.to_csv("../../../sybil-identification/data/non-anonymous/dream/ds_dreams_description_least_active_stats.csv")
    # ------------------------------------------------------------------------------#
################################################################################


################################################################################
#################### Finding Sybils in Alphabay and Silk-Road dataset ##############
elif args.mode == "alphabay-silk":
    print("Finding Sybils in Alphabay and Silk-Road dataset...")
    silk_df = silk_listing_df.copy()
    alphabay_df = alphabay_df_temp.copy()
    as_sybils = set(alphabay_df['vendor'].unique()).intersection(set(silk_df['vendor'].unique()))
    as_sybils = [str(sybil).lower() for sybil in as_sybils]

    vendor_list = list(alpha_silk['vendor'])
    vendor_list = [str(vendor).lower() for vendor in vendor_list]

    for index, vendor in enumerate(vendor_list):
        if vendor in as_sybils:
            pass
        else:
            vendor_list[index] = 'others'

    alphabay_silk_stats = dict(Counter(vendor_list))
    del alphabay_silk_stats['others']
    alphabay_silk_stats = {k: v for k, v in sorted(alphabay_silk_stats.items(), key=lambda item: item[1], reverse=True)}

    most_active_user = Counter(alphabay_silk_stats).most_common()[0][0]
    least_active_user = Counter(alphabay_silk_stats).most_common()[-1][0]
    moderately_active_user = Counter(alphabay_silk_stats).most_common()[int(len(Counter(alphabay_silk_stats).most_common())/2)][0]

    #------------------------------ Most Active user -------------------------------#
    # Most active silk user
    silk_df['vendor'] = silk_df['vendor'].apply(lambda x : str(x).lower())
    temp_silk_df = silk_df[silk_df['vendor']==most_active_user]
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : str(x).lower())
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : str(x).lower())
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : emoji.demojize(x))
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : emoji.demojize(x))
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : clean_data(x))
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : clean_data(x))

    # Most active alphabay user
    alphabay_df['vendor'] = alphabay_df['vendor'].apply(lambda x : str(x).lower())
    temp_alphabay_df = alphabay_df[alphabay_df['vendor']==most_active_user]
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : str(x).lower())
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : str(x).lower())
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : emoji.demojize(x))
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : emoji.demojize(x))
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : clean_data(x))
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : clean_data(x))

    silk_title = list(temp_silk_df['title'])
    alphabay_title = list(temp_alphabay_df['title'])
    silk_description = list(temp_silk_df['description'])
    alphabay_description = list(temp_alphabay_df['description'])

    # Generating the title data
    title_doc = [nlp(title) for title in alphabay_title]
    data1_title_text = [[token.text for token in title] for title in title_doc]
    data1_title_text = [item for sublist in data1_title_text for item in sublist]
    data1_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_pos = [item for sublist in data1_title_pos for item in sublist]
    data1_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_dep = [item for sublist in data1_title_dep for item in sublist]
    data1_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data1_title_stop = [item for sublist in data1_title_stop for item in sublist]
    title_doc = [nlp(title) for title in silk_title]
    data2_title_text = [[token.text for token in title] for title in title_doc]
    data2_title_text = [item for sublist in data2_title_text for item in sublist]
    data2_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_pos = [item for sublist in data2_title_pos for item in sublist]
    data2_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_dep = [item for sublist in data2_title_dep for item in sublist]
    data2_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data2_title_stop = [item for sublist in data2_title_stop for item in sublist]

    # Generating the description data
    description_doc = [nlp(description) for description in alphabay_description]
    data1_description_text = [[token.text for token in description] for description in description_doc]
    data1_description_text = [item for sublist in data1_description_text for item in sublist]
    data1_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_pos = [item for sublist in data1_description_pos for item in sublist]
    data1_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_dep = [item for sublist in data1_description_dep for item in sublist]
    data1_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data1_description_stop = [item for sublist in data1_description_stop for item in sublist]
    description_doc = [nlp(description) for description in silk_description]
    data2_description_text = [[token.text for token in description] for description in description_doc]
    data2_description_text = [item for sublist in data2_description_text for item in sublist]
    data2_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_pos = [item for sublist in data2_description_pos for item in sublist]
    data2_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_dep = [item for sublist in data2_description_dep for item in sublist]
    data2_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data2_description_stop = [item for sublist in data2_description_stop for item in sublist]

    # Creating an empty dataset
    title_df1 = pd.DataFrame(columns=['text','pos','dep','is_stop'])
    title_df2 = title_df1.copy()
    description_df1 = title_df1.copy()
    description_df2 = title_df1.copy()

    title_df1['text'] = data1_title_text
    title_df1['pos'] = data1_title_pos
    title_df1['dep'] = data1_title_dep
    title_df1['is_stop'] = data1_title_stop
    title_df2['text'] = data2_title_text
    title_df2['pos'] = data2_title_pos
    title_df2['dep'] = data2_title_dep
    title_df2['is_stop'] = data2_title_stop

    description_df1['text'] = data1_description_text
    description_df1['pos'] = data1_description_pos
    description_df1['dep'] = data1_description_dep
    description_df1['is_stop'] = data1_description_stop
    description_df2['text'] = data2_description_text
    description_df2['pos'] = data2_description_pos
    description_df2['dep'] = data2_description_dep
    description_df2['is_stop'] = data2_description_stop

    title_df2.to_csv("../../../sybil-identification/data/non-anonymous/silk-road/as_silk_title_most_active_stats.csv")
    title_df1.to_csv("../../../sybil-identification/data/non-anonymous/alphabay/as_alphabay_title_most_active_stats.csv")
    description_df2.to_csv("../../../sybil-identification/data/non-anonymous/silk-road/as_silk_description_most_active_stats.csv")
    description_df1.to_csv("../../../sybil-identification/data/non-anonymous/alphabay/as_alphabay_description_most_active_stats.csv")
    # ------------------------------------------------------------------------------#

    #------------------------------ moderately Active user -------------------------------#
    # moderately active silk user
    temp_silk_df = silk_df[silk_df['vendor']==moderately_active_user]
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : str(x).lower())
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : str(x).lower())
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : emoji.demojize(x))
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : emoji.demojize(x))
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : clean_data(x))
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : clean_data(x))

    # moderately active alphabay user
    temp_alphabay_df = alphabay_df[alphabay_df['vendor']==moderately_active_user]
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : str(x).lower())
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : str(x).lower())
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : emoji.demojize(x))
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : emoji.demojize(x))
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : clean_data(x))
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : clean_data(x))

    silk_title = list(temp_silk_df['title'])
    alphabay_title = list(temp_alphabay_df['title'])
    silk_description = list(temp_silk_df['description'])
    alphabay_description = list(temp_alphabay_df['description'])

    # Generating the title data
    title_doc = [nlp(title) for title in alphabay_title]
    data1_title_text = [[token.text for token in title] for title in title_doc]
    data1_title_text = [item for sublist in data1_title_text for item in sublist]
    data1_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_pos = [item for sublist in data1_title_pos for item in sublist]
    data1_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_dep = [item for sublist in data1_title_dep for item in sublist]
    data1_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data1_title_stop = [item for sublist in data1_title_stop for item in sublist]
    title_doc = [nlp(title) for title in silk_title]
    data2_title_text = [[token.text for token in title] for title in title_doc]
    data2_title_text = [item for sublist in data2_title_text for item in sublist]
    data2_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_pos = [item for sublist in data2_title_pos for item in sublist]
    data2_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_dep = [item for sublist in data2_title_dep for item in sublist]
    data2_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data2_title_stop = [item for sublist in data2_title_stop for item in sublist]

    # Generating the description data
    description_doc = [nlp(description) for description in alphabay_description]
    data1_description_text = [[token.text for token in description] for description in description_doc]
    data1_description_text = [item for sublist in data1_description_text for item in sublist]
    data1_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_pos = [item for sublist in data1_description_pos for item in sublist]
    data1_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_dep = [item for sublist in data1_description_dep for item in sublist]
    data1_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data1_description_stop = [item for sublist in data1_description_stop for item in sublist]
    description_doc = [nlp(description) for description in silk_description]
    data2_description_text = [[token.text for token in description] for description in description_doc]
    data2_description_text = [item for sublist in data2_description_text for item in sublist]
    data2_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_pos = [item for sublist in data2_description_pos for item in sublist]
    data2_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_dep = [item for sublist in data2_description_dep for item in sublist]
    data2_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data2_description_stop = [item for sublist in data2_description_stop for item in sublist]

    # Creating an empty dataset
    title_df1 = pd.DataFrame(columns=['text','pos','dep','is_stop'])
    title_df2 = title_df1.copy()
    description_df1 = title_df1.copy()
    description_df2 = title_df1.copy()

    title_df1['text'] = data1_title_text
    title_df1['pos'] = data1_title_pos
    title_df1['dep'] = data1_title_dep
    title_df1['is_stop'] = data1_title_stop
    title_df2['text'] = data2_title_text
    title_df2['pos'] = data2_title_pos
    title_df2['dep'] = data2_title_dep
    title_df2['is_stop'] = data2_title_stop

    description_df1['text'] = data1_description_text
    description_df1['pos'] = data1_description_pos
    description_df1['dep'] = data1_description_dep
    description_df1['is_stop'] = data1_description_stop
    description_df2['text'] = data2_description_text
    description_df2['pos'] = data2_description_pos
    description_df2['dep'] = data2_description_dep
    description_df2['is_stop'] = data2_description_stop

    title_df2.to_csv("../../../sybil-identification/data/non-anonymous/silk-road/as_silk_title_moderately_active_stats.csv")
    title_df1.to_csv("../../../sybil-identification/data/non-anonymous/alphabay/as_alphabay_title_moderately_active_stats.csv")
    description_df2.to_csv("../../../sybil-identification/data/non-anonymous/silk-road/as_silk_description_moderately_active_stats.csv")
    description_df1.to_csv("../../../sybil-identification/data/non-anonymous/alphabay/as_alphabay_description_moderately_active_stats.csv")
    # ------------------------------------------------------------------------------#

    #------------------------------ least Active user -------------------------------#
    # least active silk user
    temp_silk_df = silk_df[silk_df['vendor']==least_active_user]
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : str(x).lower())
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : str(x).lower())
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : emoji.demojize(x))
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : emoji.demojize(x))
    temp_silk_df['title'] = temp_silk_df['title'].apply(lambda x : clean_data(x))
    temp_silk_df['description'] = temp_silk_df['description'].apply(lambda x : clean_data(x))

    # least active alphabay user
    temp_alphabay_df = alphabay_df[alphabay_df['vendor']==least_active_user]
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : str(x).lower())
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : str(x).lower())
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : emoji.demojize(x))
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : emoji.demojize(x))
    temp_alphabay_df['title'] = temp_alphabay_df['title'].apply(lambda x : clean_data(x))
    temp_alphabay_df['description'] = temp_alphabay_df['description'].apply(lambda x : clean_data(x))

    silk_title = list(temp_silk_df['title'])
    alphabay_title = list(temp_alphabay_df['title'])
    silk_description = list(temp_silk_df['description'])
    alphabay_description = list(temp_alphabay_df['description'])

    # Generating the title data
    title_doc = [nlp(title) for title in alphabay_title]
    data1_title_text = [[token.text for token in title] for title in title_doc]
    data1_title_text = [item for sublist in data1_title_text for item in sublist]
    data1_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_pos = [item for sublist in data1_title_pos for item in sublist]
    data1_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data1_title_dep = [item for sublist in data1_title_dep for item in sublist]
    data1_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data1_title_stop = [item for sublist in data1_title_stop for item in sublist]
    title_doc = [nlp(title) for title in silk_title]
    data2_title_text = [[token.text for token in title] for title in title_doc]
    data2_title_text = [item for sublist in data2_title_text for item in sublist]
    data2_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_pos = [item for sublist in data2_title_pos for item in sublist]
    data2_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
    data2_title_dep = [item for sublist in data2_title_dep for item in sublist]
    data2_title_stop = [[token.is_stop for token in title] for title in title_doc]
    data2_title_stop = [item for sublist in data2_title_stop for item in sublist]

    # Generating the description data
    description_doc = [nlp(description) for description in alphabay_description]
    data1_description_text = [[token.text for token in description] for description in description_doc]
    data1_description_text = [item for sublist in data1_description_text for item in sublist]
    data1_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_pos = [item for sublist in data1_description_pos for item in sublist]
    data1_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data1_description_dep = [item for sublist in data1_description_dep for item in sublist]
    data1_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data1_description_stop = [item for sublist in data1_description_stop for item in sublist]
    description_doc = [nlp(description) for description in silk_description]
    data2_description_text = [[token.text for token in description] for description in description_doc]
    data2_description_text = [item for sublist in data2_description_text for item in sublist]
    data2_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_pos = [item for sublist in data2_description_pos for item in sublist]
    data2_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
    data2_description_dep = [item for sublist in data2_description_dep for item in sublist]
    data2_description_stop = [[token.is_stop for token in description] for description in description_doc]
    data2_description_stop = [item for sublist in data2_description_stop for item in sublist]

    # Creating an empty dataset
    title_df1 = pd.DataFrame(columns=['text','pos','dep','is_stop'])
    title_df2 = title_df1.copy()
    description_df1 = title_df1.copy()
    description_df2 = title_df1.copy()

    title_df1['text'] = data1_title_text
    title_df1['pos'] = data1_title_pos
    title_df1['dep'] = data1_title_dep
    title_df1['is_stop'] = data1_title_stop
    title_df2['text'] = data2_title_text
    title_df2['pos'] = data2_title_pos
    title_df2['dep'] = data2_title_dep
    title_df2['is_stop'] = data2_title_stop

    description_df1['text'] = data1_description_text
    description_df1['pos'] = data1_description_pos
    description_df1['dep'] = data1_description_dep
    description_df1['is_stop'] = data1_description_stop
    description_df2['text'] = data2_description_text
    description_df2['pos'] = data2_description_pos
    description_df2['dep'] = data2_description_dep
    description_df2['is_stop'] = data2_description_stop

    title_df2.to_csv("../../../sybil-identification/data/non-anonymous/silk-road/as_silk_title_least_active_stats.csv")
    title_df1.to_csv("../../../sybil-identification/data/non-anonymous/alphabay/as_alphabay_title_least_active_stats.csv")
    description_df2.to_csv("../../../sybil-identification/data/non-anonymous/silk-road/as_silk_description_least_active_stats.csv")
    description_df1.to_csv("../../../sybil-identification/data/non-anonymous/alphabay/as_alphabay_description_least_active_stats.csv")
    # ------------------------------------------------------------------------------#
################################################################################

else:
    raise Exception("Choose the correct value of mode from (alphabay-dreams, dreams-silk, or alphabay-silk)")
