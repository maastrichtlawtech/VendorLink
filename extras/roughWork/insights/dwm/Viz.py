#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[18]:


import os
import numpy as np
import pandas as pd

import emoji

import spacy
from spacy_cld import LanguageDetector
import contextualSpellCheck

import matplotlib.pyplot as plt

import plotly
import plotly.express as px
import plotly.graph_objects as go


# In[ ]:


nlp = spacy.load('en_core_web_sm')
language_detector = LanguageDetector()
nlp.add_pipe(language_detector)


# # Importing Datasets

# In[94]:


markets = pd.read_csv("../data/processed_data/market_data.tsv", sep="\t", error_bad_lines=False)
forums = pd.read_csv("../data/processed_data/forum_data.tsv", sep="\t", error_bad_lines=False)


"""
# # Decided Features

# In[18]:


markets.columns


# In[19]:


print("No. of logs :")
markets.shape[0], forums.shape[0]


# # Displaying the data

# In[37]:


markets.head(100)


# In[27]:


forums.head()


# # Title

# In[54]:


market_title_ = markets[markets["title"] != "None"]
print("Markets title :")
list(market_title_.title)[0:10]


# In[55]:


forum_title_ = forums[forums["title"] != "None"]
print("Forums title :")
list(forum_title_.title)[0:10]


# # Description

# In[61]:


market_layout_ = markets[markets["description"] != "None"]
print("Markets layout :")
list(market_layout_.description)[20:25]


# In[63]:


forums_layout_ = forums[forums["description"] != "None"]
print("forumss layout :")
list(forums_layout_.description)[20:25]


# # Paragraphs

# In[64]:


market_paragraphs_ = markets[markets["paragraphs"] != "None"]
print("Markets paragraphs :")
list(market_paragraphs_.paragraphs)[20:25]


# In[65]:


forum_paragraphs_ = forums[forums["paragraphs"] != "None"]
print("forums paragraphs :")
list(forum_paragraphs_.paragraphs)[20:25]


# # Headings

# In[66]:


market_headings = markets[markets["headings"] != "None"]
print("Markets headings :")
list(market_headings.headings)[20:25]


# In[67]:


forum_headings = forums[forums["headings"] != "None"]
print("forums headings :")
list(forum_headings.headings)[20:25]


# # Listings

# In[69]:


market_listings = markets[markets["listings"] != "None"]
print("markets listings :")
list(market_listings.listings)[20:25]


# In[68]:


forum_listings = forums[forums["listings"] != "None"]
print("forums listings :")
list(forum_listings.listings)[20:25]


# # Links text

# In[72]:


market_link_text = markets[markets["link-text"] != "None"]
print("markets link_text :")
list(market_link_text["link-text"])[20:25]


# # Inconsistency in tagging

# In[36]:


markets.loc[2500:2510]


# # Stats

# In[10]:


x = ["title", "description", "paragraphs", "headings", "listings", "link-text"]
"""

"""x_ = [index for index, value in enumerate(x)]
x1 = [value-0.15 for value in x]
x2 = [value+0.15 for value in x]
"""

"""
market_list = [markets[markets["title"] != "None"].shape[0], markets[markets["description"] != "None"].shape[0],
               markets[markets["paragraphs"] != "None"].shape[0],
              markets[markets["headings"] != "None"].shape[0], markets[markets["listings"] != "None"].shape[0], 
              markets[markets["link-text"] != "None"].shape[0]]
forum_list = [forums[forums["title"] != "None"].shape[0], forums[forums["description"] != "None"].shape[0],
              forums[forums["paragraphs"] != "None"].shape[0],
              forums[forums["headings"] != "None"].shape[0], forums[forums["listings"] != "None"].shape[0], 
              forums[forums["link-text"] != "None"].shape[0]]
market_list = [value/markets.shape[0] for value in market_list]
forum_list = [value/forums.shape[0] for value in forum_list]


# In[15]:


fig = go.Figure()
fig.add_trace(go.Bar(x=['title', 'description', 'paragraphs', 'headings', 'listings', 'link-text'], y=market_list, name="Markets", marker_color="blue"))
fig.add_trace(go.Bar(x=['title', 'description', 'paragraphs', 'headings', 'listings', 'link-text'], y=forum_list, name="Forums", marker_color="orange"))
fig.update_layout(barmode='relative', 
                    title_text='Dark Web Stats',
                    xaxis_title="Features",
                    yaxis_title="% count",
                    xaxis = go.XAxis(showticklabels=False),
                    yaxis = go.YAxis(showticklabels=False)
                    )
fig.show()


# # Processing data

# Performing Spelling Check : Since all the SOTA dependecy parsers are trained on public datasets like WikiText-103, Pen TreeBank corpus, and Brown corpus; there is a discrepency in what these tagger predicts and the language on the Dark Web. Hence, performing spelling check on the Dark Web Data is not a good option. 
# 
# Removing special symbols also removes tokens like $, Euro, \% which can be important for our end task. Hence, we decided against removing special symbols.

# Cleaning data

# In[126]:
"""

def check_language_if_english(text):
    doc_ = nlp(text)
    if bool(doc_._.languages) :
        if doc_._.languages[0] == 'en':
            return True
        else:
            return False
    else:
        return True

def clean_entries(entries):
    data_entries = []
    # converting the entries to their actual datatype
    if entries != None:
        for entry in entries:
            # checking if the entry is in English Language
            if check_language_if_english(entry) == True:
                entry = entry.lower() # converting the entry to lowercase letters 
                entry = emoji.demojize(entry) # demojizing the emojis
                data_entries.append(entry)
            else:
                pass
        return data_entries
    else:
        return "None"


# In[127]:

"""
for columns in ['description', 'paragraphs', 'headings', 'listings', 'link-text']:
    markets[columns] = markets[columns].apply(lambda x : eval(x))
    markets[columns] = markets[columns].apply(lambda x:clean_entries(x))


# In[128]:


markets.head()


# In[129]:


compression_opts = dict(method='zip',archive_name='markets.csv')
markets.to_csv('markets.zip', index=False, compression=compression_opts) 
"""

# In[ ]:


for columns in ['description', 'paragraphs', 'headings', 'listings', 'link-text']:
    forums[columns] = forums[columns].apply(lambda x : eval(x))
    forums[columns] = forums[columns].apply(lambda x:clean_entries(x))


# In[ ]:


compression_opts = dict(method='zip',archive_name='forums.csv')
forums.to_csv('forums.zip', index=False, compression=compression_opts) 


# In[ ]:




