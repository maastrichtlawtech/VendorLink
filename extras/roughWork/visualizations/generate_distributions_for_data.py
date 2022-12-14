# %% Importing Libraries
import os
from collections import Counter, defaultdict
import pickle

from tqdm import tqdm

import numpy as np
import pandas as pd

import spacy
nlp = spacy.load('en_core_web_sm')

import emoji

import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings('ignore')

# %% Loading dataset
"""ad_alpha_title_most = pd.read_csv("../data/non-anonymous/alphabay/ad_alphabay_title_most_active_stats.csv")
ad_dreams_title_most = pd.read_csv("../data/non-anonymous/dream/ad_dreams_title_most_active_stats.csv")
ad_alpha_des_most = pd.read_csv("../data/non-anonymous/alphabay/ad_alphabay_description_most_active_stats.csv")
ad_dreams_des_most = pd.read_csv("../data/non-anonymous/dream/ad_dreams_description_most_active_stats.csv")
ad_alpha_title_moderately = pd.read_csv("../data/non-anonymous/alphabay/ad_alphabay_title_moderately_active_stats.csv")
ad_dreams_title_moderately = pd.read_csv("../data/non-anonymous/dream/ad_dreams_title_moderately_active_stats.csv")
ad_alpha_des_moderately = pd.read_csv("../data/non-anonymous/alphabay/ad_alphabay_description_moderately_active_stats.csv")
ad_dreams_des_moderately = pd.read_csv("../data/non-anonymous/dream/ad_dreams_description_moderately_active_stats.csv")
ad_alpha_title_least = pd.read_csv("../data/non-anonymous/alphabay/ad_alphabay_title_least_active_stats.csv")
ad_dreams_title_least = pd.read_csv("../data/non-anonymous/dream/ad_dreams_title_least_active_stats.csv")
ad_alpha_des_least = pd.read_csv("../data/non-anonymous/alphabay/ad_alphabay_description_least_active_stats.csv")
ad_dreams_des_least = pd.read_csv("../data/non-anonymous/dream/ad_dreams_description_least_active_stats.csv")

ds_silk_title_most = pd.read_csv("../data/non-anonymous/silk-road/ds_silk_title_most_active_stats.csv")
ds_dreams_title_most = pd.read_csv("../data/non-anonymous/dream/ds_dreams_title_most_active_stats.csv")
ds_silk_des_most = pd.read_csv("../data/non-anonymous/silk-road/ds_silk_description_most_active_stats.csv")
ds_dreams_des_most = pd.read_csv("../data/non-anonymous/dream/ds_dreams_description_most_active_stats.csv")
ds_silk_title_moderately = pd.read_csv("../data/non-anonymous/silk-road/ds_silk_title_moderately_active_stats.csv")
ds_dreams_title_moderately = pd.read_csv("../data/non-anonymous/dream/ds_dreams_title_moderately_active_stats.csv")
ds_silk_des_moderately = pd.read_csv("../data/non-anonymous/silk-road/ds_silk_description_moderately_active_stats.csv")
ds_dreams_des_moderately = pd.read_csv("../data/non-anonymous/dream/ds_dreams_description_moderately_active_stats.csv")
ds_silk_title_least = pd.read_csv("../data/non-anonymous/silk-road/ds_silk_title_least_active_stats.csv")
ds_dreams_title_least = pd.read_csv("../data/non-anonymous/dream/ds_dreams_title_least_active_stats.csv")
ds_silk_des_least = pd.read_csv("../data/non-anonymous/silk-road/ds_silk_description_least_active_stats.csv")
ds_dreams_des_least = pd.read_csv("../data/non-anonymous/dream/ds_dreams_description_least_active_stats.csv")

as_silk_title_most = pd.read_csv("../data/non-anonymous/silk-road/as_silk_title_most_active_stats.csv")
as_alpha_title_most = pd.read_csv("../data/non-anonymous/alphabay/as_alphabay_title_most_active_stats.csv")
as_silk_des_most = pd.read_csv("../data/non-anonymous/silk-road/as_silk_description_most_active_stats.csv")
as_alpha_des_most = pd.read_csv("../data/non-anonymous/alphabay/as_alphabay_description_most_active_stats.csv")
as_silk_title_moderately = pd.read_csv("../data/non-anonymous/silk-road/as_silk_title_moderately_active_stats.csv")
as_alpha_title_moderately = pd.read_csv("../data/non-anonymous/alphabay/as_alphabay_title_moderately_active_stats.csv")
as_silk_des_moderately = pd.read_csv("../data/non-anonymous/silk-road/as_silk_description_moderately_active_stats.csv")
as_alpha_des_moderately = pd.read_csv("../data/non-anonymous/alphabay/as_alphabay_description_moderately_active_stats.csv")
as_silk_title_least = pd.read_csv("../data/non-anonymous/silk-road/as_silk_title_least_active_stats.csv")
as_alpha_title_least = pd.read_csv("../data/non-anonymous/alphabay/as_alphabay_title_least_active_stats.csv")
as_silk_des_least = pd.read_csv("../data/non-anonymous/silk-road/as_silk_description_least_active_stats.csv")
as_alpha_des_least = pd.read_csv("../data/non-anonymous/alphabay/as_alphabay_description_least_active_stats.csv")"""


# %% Distribution function
def generate_distribution(data1, data2, bar_color1, bar_color2, token_color1, token_color2, label1, label2, title, filename, tag_type='pos', ntokens=None):
    data1 = data1[['text', tag_type]]
    data2 = data2[['text', tag_type]]

    if tag_type == 'is_stop':
        data1 = data1[data1[tag_type]==True]
        data2 = data2[data2[tag_type]==True]

    # Dropping rows with None values
    data1 = data1.dropna()
    data2 = data2.dropna()
    
    all_tags = set(data1[tag_type].unique()).intersection(set(data2[tag_type].unique()))
    
    data1_ = data1.set_index('text')
    data2_ = data2.copy().set_index('text')
    data1_density = dict(Counter(data1_[tag_type]))
    data2_density = dict(Counter(data2_[tag_type]))
    # Normalizing data density
    data1_density = {key:value/max(data1_density.values()) for key,value in data1_density.items()}
    data2_density = {key:value/max(data2_density.values()) for key,value in data2_density.items()}

    # Getting all the tokens and tags from the data
    data1_tokens = list(data1['text'])
    data1_tag = list(data1[tag_type])
    data2_tokens = list(data2['text'])
    data2_tag = list(data2[tag_type])

    if ntokens == None:
        tuple_data1 = dict(Counter([(data1_tokens[index], data1_tag[index]) for index, value in enumerate(data1_tokens)]))
        tuple_data2 = dict(Counter([(data2_tokens[index], data2_tag[index]) for index, value in enumerate(data2_tokens)]))
        
    else:
        tuple_data1 = dict(Counter([(data1_tokens[index], data1_tag[index]) for index, value in enumerate(data1_tokens)]).most_common(ntokens))
        tuple_data2 = dict(Counter([(data2_tokens[index], data2_tag[index]) for index, value in enumerate(data2_tokens)]).most_common(ntokens))
    
    # Normalizing the token frequency frequency
    tuple_data1 = {key:value/max(tuple_data1.values()) for key,value in tuple_data1.items()}
    tuple_data2 = {key:value/max(tuple_data2.values()) for key,value in tuple_data2.items()}
    
    ####################################### Collecting visualization data for data1 ################################################
    # Flipping the dictionary to get it in the order of {pos-tags:list(tuple(token,mean_frequency))}
    data1_token_dict = defaultdict(list)
    for (token, tag), norm_freq in tuple_data1.items():
        data1_token_dict[tag].append((token, norm_freq))
        
    # Adding the null features for the tags not present
    for tags in all_tags:
        if tags not in data1_token_dict.keys():
            data1_token_dict[tags].append((' ',0.0))

    # Sorting dict on the basis of the names
    sorted_data1_dict = {}
    for key in sorted(data1_token_dict.keys()):
        sorted_data1_dict[key] = data1_token_dict[key]

    ####################################### Collecting visualization data for data2 ################################################
    # Flipping the dictionary to get it in the order of {pos-tags:list(tuple(token,mean_frequency))}
    data2_token_dict = defaultdict(list)
    for (token, tag), norm_freq in tuple_data2.items():
        data2_token_dict[tag].append((token, norm_freq))
        
    # Adding the null features for the tags not present
    for tags in all_tags:
        if tags not in data2_token_dict.keys():
            data2_token_dict[tags].append((' ',0.0))

    # Sorting dict on the basis of the names
    sorted_data2_dict = {}
    for key in sorted(data2_token_dict.keys()):
        sorted_data2_dict[key] = data2_token_dict[key]

    ###################################################### Generating Visualization ################################################
    fig = go.Figure()
    # Plotting the bar plot
    fig.add_trace(go.Bar(x=list(data1_density.keys()), y=list(data1_density.values()), name=label1, marker_color=bar_color1, opacity=0.6))
    fig.add_trace(go.Bar(x=list(data2_density.keys()), y=list(data2_density.values()), name=label2, marker_color=bar_color2, opacity=0.6))
    
    # Plotting the tokens on the top of the bar plot
    tag_data1 = list(sorted_data1_dict.keys())
    values_data1 = list(sorted_data1_dict.values())
    tag_data2 = list(sorted_data2_dict.keys())
    values_data2 = list(sorted_data2_dict.values())
    data1_value = [[(value[0],np.nan) if value[1]==0.0 else (value[0],value[1]) for value in pairs] for pairs in values_data1]
    data2_value = [[(value[0],np.nan) if value[1]==0.0 else (value[0],value[1]) for value in pairs] for pairs in values_data2]
    data1_token = [[value[0] for value in pairs] for pairs in data1_value]
    data1_frequency = [[value[1] for value in pairs] for pairs in data1_value]
    data2_token = [[value[0] for value in pairs] for pairs in data2_value]
    data2_frequency = [[value[1] for value in pairs] for pairs in data2_value]

    tag_data1_list, frequency_data1_list, token_data1_list = ([] for i in range(3))
    for index in range(len(tag_data1)):
        for frequency_list_index, frequency in enumerate(data1_frequency[index]):
            if frequency >= 0.0:
                tag_data1_list.append(tag_data1[index])
                frequency_data1_list.append(frequency)
                token_data1_list.append(data1_token[index][frequency_list_index])

    fig.add_trace(go.Scatter(x=tag_data1_list, y=frequency_data1_list, text=token_data1_list, mode='markers+text', 
                            marker_color=token_color1, name=label1, textposition='bottom left', textfont={'color': token_color1}))

    tag_data2_list, frequency_data2_list, token_data2_list = ([] for i in range(3))
    for index in range(len(tag_data2)):
        for frequency_list_index, frequency in enumerate(data2_frequency[index]):
            if frequency >= 0.0:
                tag_data2_list.append(tag_data2[index])
                frequency_data2_list.append(frequency)
                token_data2_list.append(data2_token[index][frequency_list_index])
    
    fig.add_trace(go.Scatter(x=tag_data2_list, y=frequency_data2_list, text=token_data2_list, mode='markers+text', 
                            marker_color=token_color2, name=label2, textposition='bottom right', textfont={'color':token_color2}))

    fig.update_layout(title_text=title,
                    xaxis_title= tag_type + "-tags",
                    yaxis_title="Normalized Frequency",
                    xaxis = go.XAxis(showticklabels=True),
                    yaxis = go.YAxis(showticklabels=True),
                    yaxis_type="log"
                    )
        
    # plotly.offline.plot(fig, filename = filename), auto_open=False)
    fig.show()

def clean_data(text):
    text = text.replace("♕","kingPieceEmoji ").replace("★","starEmoji ")
    text = text.replace("\r", " \r ").replace("\n", " \n ")
    return text

def hellinger_distance(p, q):
    """
    Calculates the hellinger distance for two different probability distribution.
    :param p,q: a dictionary with its keys being unique tokens and value being their frequency(dtype:dict) 
    :return: average hellinger distance
    """
    all_unique_tokens = set(p.keys()) | set(q.keys())
    
    # Adding tokens from all_unique_tokens to the dictionaries p and q
    for token in all_unique_tokens:
        if token not in p:
            p[token] = 0
        if token not in q:
            q[token] = 0
    
    # preparing the sorted dictionaries
    a, b = ({} for i in range(2))
    sorted_keys = sorted(p.keys())
    for token in sorted_keys:
        a[token] = p[token]
        b[token] = q[token]
    
    hellinger_distance = np.sqrt(np.sum((np.sqrt(list(a.values())) - np.sqrt(list(b.values()))) ** 2)) / _SQRT2
    return hellinger_distance/len(a)


_SQRT2 = np.sqrt(2)     # sqrt(2) with default precision np.float64
def calculate_hellinger_distance_between_data_distribution(data1, data2):
    """
    Calculates the hellinger distance for two different data distribution.
    :param data1,data2: two dataframes with columns being text, pos, dep, is_stop(dtype:pandas dataframe) 
    :return: average hellinger distance for all the features
    """
    vendor_title_dict, vendor_description_dict = ({} for i in range(2))

    data1 = data1[['vendor', 'title', 'description']]
    data2 = data2[['vendor', 'title', 'description']]
    data1['vendor'] = data1['vendor'].apply(lambda x : str(x).lower())
    data2['vendor'] = data2['vendor'].apply(lambda x : str(x).lower())
    sybils = set(data1['vendor']).intersection(set(data2['vendor']))
    # Dropping rows with None values
    data1 = data1.dropna()
    data2 = data2.dropna()

    # filtering out for drugs category
    
    # Iterating through all unique vendors
    pbar = tqdm(total=len(sybils))
    for sybil in sybils:
        data1_temp = data1[data1['vendor']==sybil]
        data2_temp = data2[data2['vendor']==sybil]
        
        data1_temp['title'] = data1_temp['title'].apply(lambda x : str(x).lower())
        data1_temp['description'] = data1_temp['description'].apply(lambda x : str(x).lower())
        data1_temp['title'] = data1_temp['title'].apply(lambda x : emoji.demojize(x))
        data1_temp['description'] = data1_temp['description'].apply(lambda x : emoji.demojize(x))
        data1_temp['title'] = data1_temp['title'].apply(lambda x : clean_data(x))
        data1_temp['description'] = data1_temp['description'].apply(lambda x : clean_data(x))

        data2_temp['title'] = data2_temp['title'].apply(lambda x : str(x).lower())
        data2_temp['description'] = data2_temp['description'].apply(lambda x : str(x).lower())
        data2_temp['title'] = data2_temp['title'].apply(lambda x : emoji.demojize(x))
        data2_temp['description'] = data2_temp['description'].apply(lambda x : emoji.demojize(x))
        data2_temp['title'] = data2_temp['title'].apply(lambda x : clean_data(x))
        data2_temp['description'] = data2_temp['description'].apply(lambda x : clean_data(x))

        title1 = list(data1_temp['title'])
        description1 = list(data1_temp['description'])
        title2 = list(data2_temp['title'])
        description2 = list(data2_temp['description'])

        # Generating the title data
        title_doc = [nlp(title) for title in title1]
        data1_title_text = [[token.text for token in title] for title in title_doc]
        data1_title_text = [item for sublist in data1_title_text for item in sublist]
        data1_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
        data1_title_pos = [item for sublist in data1_title_pos for item in sublist]
        data1_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
        data1_title_dep = [item for sublist in data1_title_dep for item in sublist]
        data1_title_stop = [[token.is_stop for token in title] for title in title_doc]
        data1_title_stop = [item for sublist in data1_title_stop for item in sublist]
        title_doc = [nlp(title) for title in title2]
        data2_title_text = [[token.text for token in title] for title in title_doc]
        data2_title_text = [item for sublist in data2_title_text for item in sublist]
        data2_title_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
        data2_title_pos = [item for sublist in data2_title_pos for item in sublist]
        data2_title_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in title] for title in title_doc]
        data2_title_dep = [item for sublist in data2_title_dep for item in sublist]
        data2_title_stop = [[token.is_stop for token in title] for title in title_doc]
        data2_title_stop = [item for sublist in data2_title_stop for item in sublist]

        # Generating the description data
        description_doc = [nlp(description) for description in description1]
        data1_description_text = [[token.text for token in description] for description in description_doc]
        data1_description_text = [item for sublist in data1_description_text for item in sublist]
        data1_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
        data1_description_pos = [item for sublist in data1_description_pos for item in sublist]
        data1_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
        data1_description_dep = [item for sublist in data1_description_dep for item in sublist]
        data1_description_stop = [[token.is_stop for token in description] for description in description_doc]
        data1_description_stop = [item for sublist in data1_description_stop for item in sublist]
        description_doc = [nlp(description) for description in description2]
        data2_description_text = [[token.text for token in description] for description in description_doc]
        data2_description_text = [item for sublist in data2_description_text for item in sublist]
        data2_description_pos = [[token.pos_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
        data2_description_pos = [item for sublist in data2_description_pos for item in sublist]
        data2_description_dep = [[token.dep_ if "Emoji" not in token.text else 'EMOJI' for token in description] for description in description_doc]
        data2_description_dep = [item for sublist in data2_description_dep for item in sublist]
        data2_description_stop = [[token.is_stop for token in description] for description in description_doc]
        data2_description_stop = [item for sublist in data2_description_stop for item in sublist]

        ################################################## Generating stats for title #############################################################
        all_unique_tokens1 = dict(Counter(data1_title_text))
        all_unique_tokens2 = dict(Counter(data2_title_text))
        # Normalizing the frequency
        all_unique_tokens1 = {key:value/sum(all_unique_tokens1.values()) for key,value in all_unique_tokens1.items()}
        all_unique_tokens2 = {key:value/sum(all_unique_tokens2.values()) for key,value in all_unique_tokens2.items()}

        all_unique_pos1 = dict(Counter(data1_title_pos))
        all_unique_pos2 = dict(Counter(data2_title_pos))
        # Normalizing the frequency
        all_unique_pos1 = {key:value/sum(all_unique_pos1.values()) for key,value in all_unique_pos1.items()}
        all_unique_pos2 = {key:value/sum(all_unique_pos2.values()) for key,value in all_unique_pos2.items()}

        all_unique_dep1 = dict(Counter(data1_title_dep))
        all_unique_dep2 = dict(Counter(data2_title_dep))
        # Normalizing the frequency
        all_unique_dep1 = {key:value/sum(all_unique_dep1.values()) for key,value in all_unique_dep1.items()}
        all_unique_dep2 = {key:value/sum(all_unique_dep2.values()) for key,value in all_unique_dep2.items()}

        all_unique_stop_words1 = dict(Counter(data1_title_stop))
        all_unique_stop_words2 = dict(Counter(data2_title_stop))
        # Normalizing the frequency
        all_unique_stop_words1 = {key:value/sum(all_unique_stop_words1.values()) for key,value in all_unique_stop_words1.items()}
        all_unique_stop_words2 = {key:value/sum(all_unique_stop_words2.values()) for key,value in all_unique_stop_words2.items()}

        token_distance = hellinger_distance(all_unique_tokens1, all_unique_tokens2)
        pos_distance = hellinger_distance(all_unique_pos1, all_unique_pos2)
        dep_distance = hellinger_distance(all_unique_dep1, all_unique_dep2)
        stop_distance = hellinger_distance(all_unique_stop_words1, all_unique_stop_words2)

        vendor_title_dict[sybil] = {'text':token_distance, 'pos':pos_distance, 'dep':dep_distance, 'is_stop':stop_distance} 

        ################################################## Generating stats for description #############################################################
        all_unique_tokens1 = dict(Counter(data1_description_text))
        all_unique_tokens2 = dict(Counter(data2_description_text))
        # Normalizing the frequency
        all_unique_tokens1 = {key:value/sum(all_unique_tokens1.values()) for key,value in all_unique_tokens1.items()}
        all_unique_tokens2 = {key:value/sum(all_unique_tokens2.values()) for key,value in all_unique_tokens2.items()}

        all_unique_pos1 = dict(Counter(data1_description_pos))
        all_unique_pos2 = dict(Counter(data2_description_pos))
        # Normalizing the frequency
        all_unique_pos1 = {key:value/sum(all_unique_pos1.values()) for key,value in all_unique_pos1.items()}
        all_unique_pos2 = {key:value/sum(all_unique_pos2.values()) for key,value in all_unique_pos2.items()}

        all_unique_dep1 = dict(Counter(data1_description_dep))
        all_unique_dep2 = dict(Counter(data2_description_dep))
        # Normalizing the frequency
        all_unique_dep1 = {key:value/sum(all_unique_dep1.values()) for key,value in all_unique_dep1.items()}
        all_unique_dep2 = {key:value/sum(all_unique_dep2.values()) for key,value in all_unique_dep2.items()}

        all_unique_stop_words1 = dict(Counter(data1_description_stop))
        all_unique_stop_words2 = dict(Counter(data2_description_stop))
        # Normalizing the frequency
        all_unique_stop_words1 = {key:value/sum(all_unique_stop_words1.values()) for key,value in all_unique_stop_words1.items()}
        all_unique_stop_words2 = {key:value/sum(all_unique_stop_words2.values()) for key,value in all_unique_stop_words2.items()}

        token_distance = hellinger_distance(all_unique_tokens1, all_unique_tokens2)
        pos_distance = hellinger_distance(all_unique_pos1, all_unique_pos2)
        dep_distance = hellinger_distance(all_unique_dep1, all_unique_dep2)
        stop_distance = hellinger_distance(all_unique_stop_words1, all_unique_stop_words2)

        vendor_description_dict[sybil] = {'text':token_distance, 'pos':pos_distance, 'dep':dep_distance, 'is_stop':stop_distance} 
        
        pbar.update(1)

    pbar.close()
    return vendor_title_dict, vendor_description_dict


# %% POS distribution for Alphabay-Dream most activated titles
"""generate_distribution(ad_alpha_title_most, ad_dreams_title_most, '#ef9a9a', '#c5e1a5', '#ff867c', '#7cb342', 'Alphabay', 'Dreams',
                        'POS distribution for Alphabay-Dream most activated titles', 'abs', ntokens=50)
"""