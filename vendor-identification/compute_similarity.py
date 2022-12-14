"""
Python version : 3.8
Description : Takes the input pcikled file with sentence representations and computes text similarity between advertisements 
              of two vendors.
Note : Please make sure to run it after generate_vendorRepresentations.py
"""

# %% Importing libraries
import os
import sys
import logging
import random
import pickle
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Computes text similarity in the vendor advertisements using cosine distance")
parser.add_argument('--pickle_dir', type=str, default="../pickled/", help="Can be roberta or gpt2")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
args = parser.parse_args()

logging.basicConfig(level=logging.ERROR)

# setting random seed
# pl.seed_everything(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# %% Loading the extracted vendor advertisement representations
with open(os.path.join(args.pickle_dir, "sentence_representations_for_vendors.pickle"), 'rb') as handle:
    vendor_representations = pickle.load(handle)

# While it's possible to compute text similarity for all the vendors in our datasets, vendor identification is a very computationally
# expensive task. Below are some of the vendors from our dataset that are displayed for a quick understanding. 
vendor_list = [('qualitymedicine', 'qualitymedical'), ('emeraldtriangle', 'theemeraldtriangle'), 
                ('dutchaussies', 'dutch-aussie'), ('dutchdreams', 'dutchdream'), ('adderall', 'adderallz'), 
                ('amsterdam', 'amsterdaminc'), ('dutch-aussie', 'dutchaussies'), ('zeus', 'zeuss'), ('drwhite', 'dr.whit3'),
                ('greensupreme', 'greensupremacy'), ('high-society', 'highsocietyuk'), ('dutchquality3.0', 'dutchquality'),
                ('emeraldgeminiagora', 'emeraldgemini'), ('greensupremacy', 'greensupreme'), ('peterpan3', 'peter-pan'),
                ('asianvixen', 'asianvixene'), ('emeraldgemini', 'emeraldgeminiagora'), 
                ('qualitymedical', 'qualitymedicine'), ('asianvixene', 'asianvixen'), ('kriminale1', 'kriminale2'), 
                ('kriminale', 'kriminale2'), ('daydreamer', 'daydreamer33'), ('zeuss', 'zeus'), ('bcbud', 'bcbuds'), 
                ('alldrugs', 'walldrug'), ('walldrug', 'alldrugs'), ('theemeraldtriangle', 'emeraldtriangle'), 
                ('worldwideweed', 'worldwide'), ('doritos_doritos', 'doritos'), ('dutchdream', 'dutchdreams'), 
                ('meds4uk', 'meds4u'), ('dutchquality', 'dutchquality3.0'), ('kriminale2', 'kriminale1'), 
                ('kriminale', 'kriminale1'), ('adderallz', 'adderall'), ('highsocietyuk', 'high-society'),
                ('bcbuds', 'bcbud'), ('doritos', 'doritos_doritos'), ('amsterdaminc', 'amsterdam'),
                ('daydreamer33', 'daydreamer'), ('kriminale2', 'kriminale'), ('kriminale1', 'kriminale'),
                ('dr.whit3', 'drwhite'), ('meds4u', 'meds4uk'), ('worldwide', 'worldwideweed'), ('peter-pan', 'peterpan3')]

# %% Calculating similarity between vendor advertisements
def compute_similarity_between_vendors(vendor_representations, vendor_list=None):
    unique_vendors = list(vendor_representations.keys())
    similarity_dict = {}
    
    if vendor_list == None:
        pbar = tqdm(total=len(unique_vendors))
        for vendor1 in unique_vendors:
            hidden_repr1 = vendor_representations[vendor1]
            self_similarity1 = cosine_similarity(hidden_repr1, hidden_repr1, dense_output=False).mean()
            pbar.update(1)
            for vendor2 in unique_vendors:
                if vendor2 != vendor1:
                    hidden_repr2 = vendor_representations[vendor2]
                    similarity = cosine_similarity(hidden_repr1, hidden_repr2, dense_output=False).mean()
                    self_similarity2 = cosine_similarity(hidden_repr2, hidden_repr2, dense_output=False).mean()
                    normalized_similarity = (2*similarity) / (self_similarity1 + self_similarity2)
                    if vendor1 not in similarity_dict.keys():
                        similarity_dict[vendor1] = {vendor2:normalized_similarity}
                    else:
                        similarity_dict[vendor1].update({vendor2:normalized_similarity})
        pbar.close()
    else:
        pbar = tqdm(total=len(vendor_list))
        for (vendor1, vendor2) in vendor_list:
            hidden_repr1 = vendor_representations[vendor1]
            hidden_repr2 = vendor_representations[vendor2]
            similarity = cosine_similarity(hidden_repr1, hidden_repr2, dense_output=False).mean()
            self_similarity1 = cosine_similarity(hidden_repr1, hidden_repr1, dense_output=False).mean()
            self_similarity2 = cosine_similarity(hidden_repr2, hidden_repr2, dense_output=False).mean()
            if vendor1 not in similarity_dict.keys():
                similarity_dict[vendor1] = {vendor2:normalized_similarity}
            else:
                similarity_dict[vendor1].update({vendor2:normalized_similarity})
            pbar.update(1)
        pbar.close()

    return similarity_dict

# To compute similarity between all vendors, set vendor_list to None
vendor_similarity = compute_similarity_between_vendors(vendor_representations, vendor_list)

with open(os.path.join(args.pickle_dir, "vendor_similarity.pickle"), 'wb') as handle:
    pickle.dump(vendor_similarity, handle, protocol=pickle.HIGHEST_PROTOCOL)