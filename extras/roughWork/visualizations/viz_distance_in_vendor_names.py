# %% Loading the libraries
import os
import pickle
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


# %% Init argparser
parser = argparse.ArgumentParser(description="Plotting Vendors with Similarity above threshold")
parser.add_argument('--threshold', type=float, default=0.80, help="Threshold")
args = parser.parse_args()


# %% Importing the data
df = pd.read_csv("../data/processed_data/distance/vendor_distance.csv")
vendor_name = list(df['Unnamed: 0'])
vendor_name[1] = "XyZ"
df['Unnamed: 0'] = vendor_name
df.rename({"Unnamed: 0": "Vendors", "Unnamed: 2": "XyZ"}, axis='columns', inplace=True)

def get_similar_vendors_above_threshold(df, threshold):
    vendor_dict = {}
    vendor_name = list(df.columns)[1:]
    pbar = tqdm(total=df.shape[0])
    for index, row in df.iterrows():
        vendor = list(row)[0]
        row = list(row)[1:]
        row = [(vendor_name[index], similarity) for index, similarity in enumerate(row) if similarity >= threshold]
        vendor_dict[vendor] = row
        pbar.update(1)
    pbar.close()
    return vendor_dict

vendor_dict = get_similar_vendors_above_threshold(df, args.threshold)

with open(os.path.join(os.getcwd(), '../data/processed_data/vendor_distance.pickle'), 'wb') as handle:
    pickle.dump(vendor_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


