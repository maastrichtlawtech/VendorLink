"""
Python version : 3.8
Description : contains helper functions for all visualizations.
"""

# %% Importing libraries
from tqdm import tqdm

import pandas as pd

def fetch_n_vendor(df, n, vendors_to_plot=None):
    """
    param df : dataframe with vendors and similarity as column names
    param n : number of vendors to be displayed
    param vendors_to_plot : List of vendors to be processed (set it to None if you want to process all vendors)
    returns : A dataframe with vendors and their most likely n aliases
    """
    temp_df_list = []
    unique_vendors_list = list(df['vendors'].unique())

    if vendors_to_plot != None: 
        vendors_to_iterate = vendors_to_plot
    else:
        vendors_to_iterate = unique_vendors_list
    
    pbar = tqdm(total=len(vendors_to_iterate))    
    for vendor in vendors_to_iterate:
        temp_df = df[df['vendors']==vendor]
        temp_df = temp_df.sort_values(by=['similarity'], ascending=False).iloc[:n]
        temp_similarity = list(temp_df['similarity'])
        temp_similarity = [similarity/max(temp_similarity) for similarity in temp_similarity]
        temp_df['similarity'] = temp_similarity
        temp_df_list.append(temp_df)
        pbar.update(1)
    
    pbar.close()
    df_temp = pd.concat(temp_df_list)

    return df_temp

def get_markers(color_list):
    """
    param color_list : List of vendor denoted by colors based on what market they belong to.
    returns : Specific marker types associated with each markets
    """
    marker_list = []
    for color in color_list:
        if color == 'red':
            marker_list.append('triangle-up-open-dot')
        elif color == 'green':
            marker_list.append('circle-open-dot')
        elif color == 'blue':
            marker_list.append('star-open-dot')
        elif color == 'olive':
            marker_list.append('cross-open-dot')
        elif color == 'tan':
            marker_list.append('diamond-open-dot')
        elif color == 'orange':
            marker_list.append('square-open-dot')
        elif color == 'black':
            marker_list.append('pentagon-open-dot')
        else:
            print("Color :", color)
            raise Exception("No marker designated to the passed color")
            
    return marker_list

def get_vendor_color(vendormarket_dict, vendor):
    """
    param vendor : vendor name
    param vendormarket_dict : dictionary of list of vendors belonging to each market
    returns : color coding for a single market
    """
    color_dict = {"alpha" : "red", "dreams" : "green", "silk" : "blue", "alpha-dreams" : "olive", "alpha-silk" : "tan",
                    "dreams-silk" : "orange", "alpha-dreams-silk" : "black"}

    if vendor in vendormarket_dict["alpha"]:
        color = color_dict["alpha"]
    if vendor in vendormarket_dict["dreams"]:
        color = color_dict["dreams"]
    if vendor in vendormarket_dict["silk"]:
        color = color_dict["silk"]
    if vendor in vendormarket_dict["alpha"] and vendor in vendormarket_dict["dreams"]:
        color = color_dict["alpha-dreams"]
    if vendor in vendormarket_dict["alpha"] and vendor in vendormarket_dict["silk"]:
        color = color_dict["alpha-silk"]
    if vendor in vendormarket_dict["silk"] and vendor in vendormarket_dict["dreams"]:
        color = color_dict["dreams-silk"]
    if vendor in vendormarket_dict["alpha"] and vendor in vendormarket_dict["dreams"] and vendor in vendormarket_dict["silk"]:
        color = color_dict["alpha-dreams-silk"]
    else:
        raise Exception("Vendor not in any chosen markets")     
    
    return color

