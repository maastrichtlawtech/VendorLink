"""
Python version : 3.8
Description : Takes the vendor similarity and plots a scatter plot with similarity score on y axis, vendor aliases as scatter point, and parent vendor on x-axis .
Note : Please make sure to run it after compute_similarity.py
"""

# %% Importing libraries
import os, logging, sys
import random
import pickle
import argparse
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd

import plotly
import plotly.graph_objects as go
import plotly.express as px

# Loading the custom library
sys.path.append('../utilities/')
from visualizations import fetch_n_vendor, get_vendor_color, get_markers
from load_data import FetchData

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Plots a scatter plot indicating vendor similarity")
parser.add_argument('--data',  type=str, default='alpha-dreams', help="""Datasets to be evaluated (can be "shared", "alpha" for alphabay, "dreams", "silk" for silk-road, "alpha-dreams", "dreams-silk", "alpha-silk", or "alpha-dreams-silk")""")
parser.add_argument('--data_dir', type=str, default='../data', help="""Data directory of the pre-processed data (if your data is not pre-processed, we recommend you to go to the statistical model and process it first.)""")
parser.add_argument('--plot_dir', type=str, default="../plots/", help="Directories ")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
parser.add_argument('--n_vendors', type=int, default=3, help="Number of aliases/vendor to be plottled")
parser.add_argument('--version', type=str, default='full', help='Run of small data of full data. can be ("small" or "full").')
parser.add_argument('--setting', type=str, default='high', help="Low for low-resource setting and High for high-resource setting")
parser.add_argument('--split_ratio', type=float, default=0.25, help="Splitting ratio for the dataset")
parser.add_argument('--preprocess_flag', action='store_true', help='Preprocess data')
parser.add_argument('--ads_count', type=int, default=20, help="Minimum number of advertisements per vendor")
args = parser.parse_args()

logging.basicConfig(level=logging.ERROR)

# setting random seed
# pl.seed_everything(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

Path(args.plot_dir).mkdir(parents=True, exist_ok=True)

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

df_alpha = pd.concat([train_alpha, test_alpha])
df_dreams = pd.concat([train_dreams, test_dreams])
df_silk = pd.concat([train_silk, test_silk])

# Finding invidual vendors from alphabay-dreams-silk road markets
vendormarket_dict = {
                "alpha" : list(df_alpha.vendors.unique()),
                "dreams" : list(df_alpha.vendors.unique()),
                "silk" : list(df_silk.vendors.unique())
            }

# %% Loading the extracted vendor similarity file from compute_similarity.py
with open(os.path.join(args.pickle_dir, "vendor_similarity.pickle"), 'rb') as handle:
    vendor_similarity = pickle.load(handle)

X_vendors = list(vendor_similarity.keys())
Y_vendors = [list(vendor_similarity[vendor].keys()) for vendor in X_vendors]
Y_similarity = [list(vendor_similarity[vendor].values()) for vendor in X_vendors]

df = pd.DataFrame(data={"vendors":X_vendors, "aliases":Y_vendors, "similarity":Y_similarity})

# %% Fetching top n likely aliases per vendor
df = fetch_n_vendor(df, args.n_vendors, vendors_to_plot = X_vendors)

# %% Plotting vendors
vendor_color = [get_vendor_color(vendormarket_dict, vendor) for vendor in df.vendors.to_list()]
alias_color = [get_vendor_color(vendormarket_dict, vendor) for vendor in df.aliases.to_list()]
alias_marker = get_markers(alias_color)

x_ = df.vendors.to_list()
y_ = df.similarity.to_list()
scatter_text = df.aliases.to_list()

keys = dict(zip(scatter_text, alias_color))

fig = go.Figure()
for index, vendors in enumerate(x_):
    fig.add_trace(go.Scatter(x=[vendors], y=[y_[index]], mode='markers', marker=dict(color=alias_color[index], size=10, opacity=0.6, line=dict(color=alias_color[index], width=2)),marker_symbol=alias_marker[index]))
    fig.add_annotation(dict(font=dict(color="black",size=14), x=vendors, y=y_[index], showarrow=False, text=scatter_text[index],
                            font_size = 20, align='right', arrowhead=1, arrowside='start', yanchor = 'bottom', textangle=0,
                            font_color = alias_color[index], yshift = 5, ax=20))

# fig.update_traces(textposition='top center')
fig.add_hline(y=0.9, line_width=3, line_dash="dash", line_color="green")
fig.add_hrect(y0=0.9, y1=1.0, line_width=0, fillcolor="red", opacity=0.1)

fig.update_layout(height=700)
fig.update_layout(yaxis={'visible': True, 'showticklabels': True}, xaxis_title='Vendors', yaxis_title='Similarity', font=dict(size=18))
fig.update_layout(xaxis={'visible': True, 'showticklabels': False}, font=dict(size=18))

fig.update_layout(showlegend=False)
# fig.update_layout(xaxis=dict(tickmode='array', ticktext=ticktext, tickvals=color_))
plotly.offline.plot(fig, filename = os.path.join(os.getcwd(), "vendor_similarity.pdf"), auto_open=False)
fig.show()