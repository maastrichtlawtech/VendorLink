"""
Python version : 3.8
Description: Takes the raw darknet files and brings them to proper format to run further analysis
"""

# %% Importing libraries
import os
import argparse

import pandas as pd

# %% Loading and merging data from different files
alpha_items_df = pd.read_csv("../data/non-anonymous/alphabay/items.csv", error_bad_lines=False, 
                    lineterminator='\n', usecols=['marketplace', 'title', 'vendor', 'prediction', 'ships_to', 'ships_from', 'description']).drop_duplicates()
alpha_feedback_df = pd.read_csv("../data/non-anonymous/alphabay/feedbacks.csv", error_bad_lines=False, 
                    lineterminator='\n', usecols=['reciever', 'order_title', 'order_amount_usd'])
alpha_feedback_df.columns = ['vendor', 'title', 'order_amount_usd']
alpha_df = alpha_items_df.merge(alpha_feedback_df, how = 'inner', on = ['title', 'vendor'])

dreams_items_df = pd.read_csv("../data/non-anonymous/dream/items.csv", error_bad_lines=False, 
                    lineterminator='\n', usecols=['marketplace', 'title', 'vendor', 'prediction', 'ships_to', 'ships_from', 'description']).drop_duplicates()
dreams_feedback_df = pd.read_csv("../data/non-anonymous/dream/feedbacks.csv", error_bad_lines=False, 
                    lineterminator='\n', usecols=['reciever', 'order_title', 'order_amount_usd'])
dreams_feedback_df.columns = ['vendor', 'title', 'order_amount_usd']
dreams_df = dreams_items_df.merge(dreams_feedback_df, how = 'inner', on = ['title', 'vendor'])

silk_df = pd.read_csv("../data/non-anonymous/silk-road/items.csv", error_bad_lines=False, 
                    lineterminator='\n', usecols=['marketplace', 'title', 'seller_id', 'category', 'ship_to', 'ship_from', 'listing_description', 'price_btc']).drop_duplicates()
silk_df.columns = ['marketplace' ,'title', 'prediction', 'order_amount_usd', 'ships_to', 'ships_from', 'vendor', 'description']
silk_df['order_amount_usd'] = silk_df['order_amount_usd'].apply(lambda x: x*54.46)

agora_df = pd.read_csv(os.path.join("../data/Agora.csv"), error_bad_lines=False, encoding = "ISO-8859-1")
agora_df = agora_df[['Vendor', ' Item', ' Item Description']]
agora_df.columns = ['vendor', 'title', 'description']
agora_df['vendor'] = agora_df['vendor'].apply(lambda x : str(x).lower())

data_df = {"alpha":alpha_df, "dreams":dreams_df, "silk":silk_df, "agora":agora_df}

# %% Saving the dataframes as preprocessed files in the data directory
alpha_df.to_csv(os.path.join(os.getcwd(), "../data/preprocessed_alpha.csv"))
dreams_df.to_csv(os.path.join(os.getcwd(), "../data/preprocessed_dreams.csv"))
silk_df.to_csv(os.path.join(os.getcwd(), "../data/preprocessed_silk.csv"))
agora_df.to_csv(os.path.join(os.getcwd(), "../data/preprocessed_agora.csv"))