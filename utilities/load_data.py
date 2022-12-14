"""
Python version : 3.8.12
Description : 
            1) Fetch, loads, and returns the Alphabay, Dreams, Silk-Road, Valhalla, Berlusconi, Traderoute, and Agora
              pre-processed data
            2) Loads and Splits the data for BiGRU classifier with fasttext embeddings

""" 

# %% Importing libraries
from pathlib import Path
from collections import Counter

import emoji

import pandas as pd

from sklearn.model_selection import train_test_split

from torchtext.legacy import data

# %% Helper class
class FetchData(object):
    def __init__(self, data_df, version, data, split_ratio, preprocess_flag, setting, count_threshold, seed):
        self.version = version
        self.data_df = data_df
        self.data = data
        self.preprocess_flag = preprocess_flag
        self.split_ratio = split_ratio
        self.count_threshold = count_threshold
        self.seed = seed
        self.setting = setting
        self.alpha_df, self.dreams_df, self.silk_df, self.agora_df = self.if_trim_data()

    def if_trim_data(self):
        # Checking whether to run the model on small dataset or big dataset
        # The small datasets are used for the demo runs
        if self.version == 'small':
            alpha_df = self.data_df["alpha"].iloc[:100]
            dreams_df = self.data_df["dreams"].iloc[:100]
            silk_df = self.data_df["silk"].iloc[:100]
            agora_df = self.data_df["agora"].iloc[:100]

        else:
            alpha_df = self.data_df["alpha"]
            dreams_df = self.data_df["dreams"]
            silk_df = self.data_df["silk"]
            agora_df = self.data_df["agora"]

        return alpha_df, dreams_df, silk_df, agora_df

    def clean_data(self, text):
        # Our experiments demonstrated that performing cleaning on the Darknet data was counter-productive
        text = str(text).lower()
        text = emoji.demojize(text)
        text = str(text).replace("♕","kingPieceEmoji ").replace("★","starEmoji ")
        text = str(text).replace("\r", " \r ").replace("\n", " \n ")
        return text

    def process_data(self, df, preprocess_flag=False):
        # processing the train data
        if preprocess_flag == True:
            print("Turning the data into lower case ...")
            df['marketplace'] = df['marketplace'].apply(lambda x : str(x).lower())
            # df['title'] = df['title'].apply(lambda x : self.clean_data(x))
            # df['description'] = df['description'].apply(lambda x : self.clean_data(x))
            df['title'] = df['title'].apply(lambda x : str(x))
            df['description'] = df['description'].apply(lambda x : str(x))
            df['vendor'] = df['vendor'].apply(lambda x : str(x).lower())
            df['prediction'] = df['prediction'].apply(lambda x : str(x).lower())
            df['ships_to'] = df['ships_to'].apply(lambda x : str(x).lower())
            df['ships_from'] = df['ships_from'].apply(lambda x : str(x).lower())
        else:
            pass        
        return df

    def split_data(self):
        # cleaning the data
        print("Cleaning the data ...")
        self.alpha_df = self.convert_unnecessary_vendors(self.alpha_df, preprocess_flag=True)
        self.dreams_df = self.convert_unnecessary_vendors(self.dreams_df, preprocess_flag=True)
        self.silk_df = self.convert_unnecessary_vendors(self.silk_df, preprocess_flag=True)
        self.agora_df = self.convert_unnecessary_vendors(self.agora_df, preprocess_flag=True)

        self.alpha_df = self.process_data(self.alpha_df, preprocess_flag=self.preprocess_flag)
        self.dreams_df = self.process_data(self.dreams_df, preprocess_flag=self.preprocess_flag)
        self.silk_df = self.process_data(self.silk_df, preprocess_flag=self.preprocess_flag)

        # self.alpha_df.to_csv("../data/preprocessed_alpha.csv", sep=',', encoding='utf-8')
        # self.dreams_df.to_csv("../data/preprocessed_dreams.csv", sep=',', encoding='utf-8')
        # self.silk_df.to_csv("../data/preprocessed_silk.csv", sep=',', encoding='utf-8')
        
        self.dream_df = self.dreams_df[self.dreams_df['marketplace']=='dream']
        self.valhalla_df = self.dreams_df[self.dreams_df['marketplace']=='valhalla']
        self.traderoute_df = self.dreams_df[self.dreams_df['marketplace']=='traderoute']
        self.berlusconi_df = self.dreams_df[self.dreams_df['marketplace']=='berlusconi']

        # Getting the test and train dataset
        if self.setting == "high":
            train_alpha, test_alpha = train_test_split(self.alpha_df, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
            train_dreams, test_dreams = train_test_split(self.dream_df, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
            train_silk, test_silk = train_test_split(self.silk_df, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
            print("Training and Test data size for Alphabay market :", train_alpha.shape, test_alpha.shape)
            print("Training and Test data size for Dreams market :", train_dreams.shape, test_dreams.shape)
            print("Training and Test data size for Silk-Road market :", train_silk.shape, test_silk.shape)
        else:
            train_valhalla, test_valhalla = train_test_split(self.valhalla_df, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
            train_traderoute, test_traderoute = train_test_split(self.traderoute_df, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
            train_berlusconi, test_berlusconi = train_test_split(self.berlusconi_df, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
            train_agora, test_agora = train_test_split(self.agora_df, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
            print("Training and Test data size for Valhalla market :", train_valhalla.shape, test_valhalla.shape)
            print("Training and Test data size for Traderoute market :", train_traderoute.shape, test_traderoute.shape)
            print("Training and Test data size for Berlusconi market :", train_berlusconi.shape, test_berlusconi.shape)
            print("Training and Test data size for Agora market :", train_agora.shape, test_agora.shape)

        if self.data == "alpha" or self.data == "dreams" or "self.data" == "silk":
            print("Collecting and splitting the individual data ...")
            train_data = pd.concat([train_alpha, train_dreams, train_silk])
            test_data = pd.concat([test_alpha, test_dreams, test_silk])
            data_list = [(train_data, train_alpha, train_dreams, train_silk), (test_data, test_alpha, test_dreams, test_silk)]

        elif self.data == "valhalla" or self.data == "traderoute" or self.data == "berlusconi":
            print("Collecting and splitting the individual data ...")
            train_data = pd.concat([train_valhalla, train_traderoute, train_berlusconi])
            test_data = pd.concat([test_valhalla, test_traderoute, test_berlusconi])
            data_list = [(train_data, train_valhalla, train_traderoute, train_berlusconi), (test_data, test_valhalla, test_traderoute, test_berlusconi)] 

        elif self.data == "shared":
            print("Splitting shared data ...")
            if self.setting == "high":
                alpha_vendors = list(self.alpha_df['vendor'].unique())
                dreams_vendors = list(self.dreams_df['vendor'].unique())
                silk_vendors = list(self.silk_df['vendor'].unique())
                shared_vendors = set(alpha_vendors) & set(dreams_vendors) & set(silk_vendors)

                alpha_df = self.alpha_df[self.alpha_df['vendor'].isin(shared_vendors)]
                dreams_df = self.dreams_df[self.dreams_df['vendor'].isin(shared_vendors)]
                silk_df = self.silk_df[self.silk_df['vendor'].isin(shared_vendors)] 

                # alpha_df.to_csv("../data/non-anonymous/alphabay/alpha_shared.csv", sep=',', encoding='utf-8')
                # dreams_df.to_csv("../data/non-anonymous/dream/dreams_shared.csv", sep=',', encoding='utf-8')
                # silk_df.to_csv("../data/non-anonymous/silk-road/silk_shared.csv", sep=',', encoding='utf-8')

                # Getting the test and train dataset
                train_alpha, test_alpha = train_test_split(alpha_df, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
                train_dreams, test_dreams = train_test_split(dreams_df, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
                train_silk, test_silk = train_test_split(silk_df, test_size=self.split_ratio, random_state=self.seed, shuffle=True)

                train_alpha = train_alpha[['marketplace','vendor', 'title', 'description']].drop_duplicates()
                test_alpha = test_alpha[['marketplace','vendor', 'title', 'description']].drop_duplicates()
                train_dreams = train_dreams[['marketplace','vendor', 'title', 'description']].drop_duplicates()
                test_dreams = test_dreams[['marketplace','vendor', 'title', 'description']].drop_duplicates()
                train_silk = train_silk[['marketplace','vendor', 'title', 'description']].drop_duplicates()
                test_silk = test_silk[['marketplace','vendor', 'title', 'description']].drop_duplicates()

                data_list = [(train_alpha, train_dreams, train_silk), (test_alpha, test_dreams, test_silk)]
            
            else:
                valhalla_vendors = list(self.valhalla_df['vendor'].unique())
                traderoute_vendors = list(self.traderoute_df['vendor'].unique())
                berlusconi_vendors = list(self.berlusconi_df['vendor'].unique())
                shared_vendors = set(valhalla_vendors) & set(traderoute_vendors) & set(berlusconi_vendors)

                valhalla_df = self.valhalla_df[self.valhalla_df['vendor'].isin(shared_vendors)]
                traderoute_df = self.traderoute_df[self.traderoute_df['vendor'].isin(shared_vendors)]
                berlusconi_df = self.berlusconi_df[self.berlusconi_df['vendor'].isin(shared_vendors)] 

                # valhalla_df.to_csv("../data/non-anonymous/valhalla/valhalla_shared.csv", sep=',', encoding='utf-8')
                # traderoute_df.to_csv("../data/non-anonymous/traderoute/traderoute_shared.csv", sep=',', encoding='utf-8')
                # berlusconi_df.to_csv("../data/non-anonymous/berlusconi/berlusconi_shared.csv", sep=',', encoding='utf-8')

                # Getting the test and train dataset
                train_valhalla, test_valhalla = train_test_split(valhalla_df, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
                train_traderoute, test_traderoute = train_test_split(traderoute_df, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
                train_berlusconi, test_berlusconi = train_test_split(berlusconi_df, test_size=self.split_ratio, random_state=self.seed, shuffle=True)

                train_valhalla = train_valhalla[['marketplace','vendor', 'title', 'description']].drop_duplicates()
                test_valhalla = test_valhalla[['marketplace','vendor', 'title', 'description']].drop_duplicates()
                train_traderoute = train_traderoute[['marketplace','vendor', 'title', 'description']].drop_duplicates()
                test_traderoute = test_traderoute[['marketplace','vendor', 'title', 'description']].drop_duplicates()
                train_berlusconi = train_berlusconi[['marketplace','vendor', 'title', 'description']].drop_duplicates()
                test_berlusconi = test_berlusconi[['marketplace','vendor', 'title', 'description']].drop_duplicates()

                data_list = [(train_valhalla, train_traderoute, train_berlusconi), (test_valhalla, test_traderoute, test_berlusconi)]

        elif self.data == "alpha-dreams" or self.data == "dreams-silk" or self.data == "alpha-silk":
            print("Splitting combined data ...")
            alpha_dreams = pd.concat([self.alpha_df, self.dreams_df], axis=0)
            dreams_silk = pd.concat([self.dreams_df, self.silk_df], axis=0)
            alpha_silk = pd.concat([self.alpha_df, self.silk_df], axis=0)
            alpha_dreams_silk = pd.concat([self.alpha_df, self.dreams_df, self.silk_df], axis=0)

            train_alpha_dreams, test_alpha_dreams = train_test_split(alpha_dreams, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
            train_dreams_silk, test_dreams_silk = train_test_split(dreams_silk, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
            train_alpha_silk, test_alpha_silk = train_test_split(alpha_silk, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
            train_alpha_dreams_silk, test_alpha_dreams_silk = train_test_split(alpha_dreams_silk, test_size=self.split_ratio, random_state=self.seed, shuffle=True)

            train_alpha = train_alpha[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            test_alpha = test_alpha[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            train_dreams = train_dreams[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            test_dreams = test_dreams[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            train_silk = train_silk[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            test_silk = test_silk[['marketplace','vendor', 'title', 'description']].drop_duplicates()

            train_alpha_dreams = train_alpha_dreams[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            test_alpha_dreams = test_alpha_dreams[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            train_dreams_silk = train_dreams_silk[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            test_dreams_silk = test_dreams_silk[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            train_alpha_silk = train_alpha_silk[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            test_alpha_silk = test_alpha_silk[['marketplace','vendor', 'title', 'description']].drop_duplicates()

            train_alpha_dreams_silk = train_alpha_dreams_silk[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            test_alpha_dreams_silk = test_alpha_dreams_silk[['marketplace','vendor', 'title', 'description']].drop_duplicates()

            data_list = [(train_alpha_dreams, train_dreams_silk, train_alpha_silk, train_alpha_dreams_silk, train_alpha, train_dreams, train_silk),
                        (test_alpha_dreams, test_dreams_silk, test_alpha_silk, test_alpha_dreams_silk, test_alpha, test_dreams, test_silk)]

        elif self.data == "valhalla-traderoute" or self.data == "traderoute-berlusconi" or self.data == "valhalla-berlusconi" or self.data == "traderoute-agora":
            print("Splitting combined data ...")
            valhalla_traderoute = pd.concat([self.valhalla_df, self.traderoute_df], axis=0)
            traderoute_berlusconi = pd.concat([self.traderoute_df, self.berlusconi_df], axis=0)
            valhalla_berlusconi = pd.concat([self.valhalla_df, self.berlusconi_df], axis=0)
            traderoute_agora = pd.concat([self.traderoute_df, self.agora_df], axis=0)
            valhalla_traderoute_berlusconi = pd.concat([self.valhalla_df, self.traderoute_df, self.berlusconi_df], axis=0)

            train_valhalla_traderoute, test_valhalla_traderoute = train_test_split(valhalla_traderoute, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
            train_traderoute_berlusconi, test_traderoute_berlusconi = train_test_split(traderoute_berlusconi, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
            train_valhalla_berlusconi, test_valhalla_berlusconi = train_test_split(valhalla_berlusconi, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
            train_traderoute_agora, test_traderoute_agora = train_test_split(traderoute_agora, test_size=self.split_ratio, random_state=self.seed, shuffle=True)
            train_valhalla_traderoute_berlusconi, test_valhalla_traderoute_berlusconi = train_test_split(valhalla_traderoute_berlusconi, test_size=self.split_ratio, random_state=self.seed, shuffle=True)

            train_valhalla = train_valhalla[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            test_valhalla = test_valhalla[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            train_traderoute = train_traderoute[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            test_traderoute = test_traderoute[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            train_berlusconi = train_berlusconi[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            test_berlusconi = test_berlusconi[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            train_agora = train_agora[['vendor', 'title', 'description']].drop_duplicates()
            test_agora = test_agora[['vendor', 'title', 'description']].drop_duplicates()

            train_valhalla_traderoute = train_valhalla_traderoute[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            test_valhalla_traderoute = test_valhalla_traderoute[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            train_traderoute_berlusconi = train_traderoute_berlusconi[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            test_traderoute_berlusconi = test_traderoute_berlusconi[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            train_valhalla_berlusconi = train_valhalla_berlusconi[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            test_valhalla_berlusconi = test_valhalla_berlusconi[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            train_traderoute_agora = train_traderoute_agora[['vendor', 'title', 'description']].drop_duplicates()
            test_traderoute_agora = test_traderoute_agora[['vendor', 'title', 'description']].drop_duplicates()

            train_valhalla_traderoute_berlusconi = train_valhalla_traderoute_berlusconi[['marketplace','vendor', 'title', 'description']].drop_duplicates()
            test_valhalla_traderoute_berlusconi = test_valhalla_traderoute_berlusconi[['marketplace','vendor', 'title', 'description']].drop_duplicates()

            data_list = [(train_valhalla_traderoute, train_traderoute_berlusconi, train_valhalla_berlusconi, train_traderoute_agora, train_valhalla_traderoute_berlusconi, train_valhalla, train_traderoute, train_berlusconi, train_agora),
                        (test_valhalla_traderoute, test_traderoute_berlusconi, test_valhalla_berlusconi, test_traderoute_agora, test_valhalla_traderoute_berlusconi, test_valhalla, test_traderoute, test_berlusconi, test_agora)]

        else:
            raise Exception("""Datasets to be evaluated (can be "shared" for shared vendors across different markets, "alpha" for alphabay, "dreams", "silk" for silk-road, "alpha-dreams", "traderoute-berlusconi", "valhalla-traderoute", "valhalla-berlusconi", "dreams-silk", "alpha-silk", "traderoute-agora", or "alpha-dreams-silk")""")

        return data_list

    def convert_unnecessary_vendors(self, data, preprocess_flag=False):
        if preprocess_flag == True:
            ads_count = dict(Counter(data['vendor']))
            vendors_to_be_converted = [vendor for vendor, count in ads_count.items() if count < self.count_threshold]
            data['vendor'] = data['vendor'].replace(vendors_to_be_converted, 'others')
        else:
            pass
        return data

# Helper class to load and split the datasets for BiGRU classifier with fasttext embeddings
class LoadDataForBiGRUFasttext(data.Dataset):

    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            label = row.labels if not is_test else None
            text = row.text
            examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, False, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)

