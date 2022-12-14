"""
Python version : 3.8.12
Description : Contains the utility, training, and evaluate functions for the transformer-based classifiers
"""

# %% Importing libraries
import os
import re

from collections import Counter

import pandas as pd


def clip_gradient(model, clip_value):
    """
    param model : model class to be trained
    param clip_value : scales under the range of -clip_value to clip_value
    return : clipped parameters
    """
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def merge_and_create_dataframe(df):
    """
    param df : input dataframe
    return : dataframe with "text" column consisting text sentences and "labels" consisting label values 
    """
    title = list(df['title'])
    description = list(df['description'])
    labels = list(df['vendor'])
    text = [str(title[i]) + ' [SEP] ' + str(description[i]) for i in range(df.shape[0])]
    data = [[text[index],labels[index]] for index in range(len(text))]
    data = pd.DataFrame(data)
    data.columns = ["text", "labels"]
    return data

def merge_and_create_dataframe_for_probing(df):
    """
    param df : input dataframe
    return : dataframe with "text" column consisting text sentences and "labels" consisting label values 
    """
    title = list(df['title'])
    description = list(df['description'])
    labels = list(df['prediction'])
    text = [str(title[i]) + ' [SEP] ' + str(description[i]) for i in range(df.shape[0])]
    data = [[text[index],labels[index]] for index in range(len(text))]
    data = pd.DataFrame(data)
    data.columns = ["text", "labels"]
    return data

def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def add_tokens_to_vocabulary(tokenizer, train_data_dir, test_data_dir, min_occurence):
    """
    : param tokenizer : BERT pre-loaded tokenizer
    : param train_data_dir : directory of training data text file
    : param test_data_dir : directory of test data text file
    : return : tokenizer with extended vocabulary
    """
    my_file = open(train_data_dir, "r")
    train_file = my_file.read()
    train_data = train_file.split("\n")
    my_file.close()

    my_file = open(test_data_dir, "r")
    test_file = my_file.read()
    test_data = test_file.split("\n")
    my_file.close()

    data = train_data + test_data

    print('Getting BERT vocab....')
    bert_vocab = list(tokenizer.get_vocab().keys())
    print('Getting keys with alphanumeric characters....')
    alpha_numeric_keys = set([w for s in data for w in s.split() if any(c.isalpha() for c in w) and any(c.isdigit() for c in w)])
    data = [sent.split(" ") for sent in data]
    print('Getting all the tokens....')
    data = [item for sublist in data for item in sublist]
    print('Removing vocab on the basis of occurences...')
    print('Removing BERT vocab....')
    data_count_dict = {k:v for k,v in dict(Counter(data)).items() if v >= min_occurence and k not in bert_vocab}
    print("Number of newly added tokens : ", len(data_count_dict.keys()))
    print('Removing the alphanumeric characters...')
    data = [token for token,count in data_count_dict.items() if token not in alpha_numeric_keys]
    print("Adding new tokens in the tokenizer")
    tokenizer.add_tokens(list(data_count_dict.keys()))
    return tokenizer

def clean_data(text):
    """
    We also experimented with in-depth cleaning and pre-processing of the data. The results showed that any cleaning 
    and/or pre-processing was only decreasing our model's performance. Finally, we decided against it and realized
    that any noise in the text advertisement is a representation of an vendors writing style.
    """
    possibilites = list(range(0,10001))
    pat = re.compile(r"""([.()#-+⫷▐*)\/◄➽━┌┐â–¬█─┘—▬&;¢âœ°•‘✽˜‰s"▁”-└➊➋➌➍➎➏➐➑ˆâ▂▃▅▇▓▒▀▄▀▄▀îžº◘╬♬˜��…❻¤@%<>£$€{}[]═✩^|¦~☆¤`'►:_,❷0❶îž░↓▆=_▼?▲◆▄!])""")
    text = str(text).lower()
    # Removing the whitespaces
    text = text.split(" ")
    text = list(filter(None, text))
    # Removing tokens with #
    text = [string for string in text if "#" not in string]
    text = " ".join(text)
    # force-tagging the quantities inside the headings
    text = text.replace("gr.", "g ").replace("gr ", "g ").replace("gram", "g ").replace("grams", "g ")
    text = text.replace("0iu", "0-iu ").replace("1iu", "1-iu ").replace("2iu", "2-iu ").replace("3iu", "3-iu ").replace("4iu", "4-iu ").replace("5iu", "5-iu ").replace("6iu", "6-iu ").replace("7iu", "7-iu ").replace("8iu", "8-iu ").replace("9iu", "9-iu ").replace(" iu ","-iu ").replace("10 325","3250").replace("two","2")
    text = text.replace("0mg", "0-mg ").replace("1mg", "1-mg ").replace("2mg", "2-mg ").replace("3mg", "3-mg ").replace("4mg", "4-mg ").replace("5mg", "5-mg ").replace("6mg", "6-mg ").replace("7mg", "7-mg ").replace("8mg", "8-mg ").replace("9mg", "9-mg ").replace("+/- mg","-mg ").replace(".mg","-mg")
    text = text.replace("0g", "0-g ").replace("1g", "1-g ").replace("2g", "2-g ").replace("3g", "3-g ").replace("4g", "4-g ").replace("5g", "5-g ").replace("6g", "6-g ").replace("7g", "7-g ").replace("8g", "8-g ").replace("9g", "9-g ").replace("}g", "-g ").replace(". g","-g ").replace(".g", "-g ")
    text = text.replace("0-g", "0-g ").replace("1-g", "1-g ").replace("2-g", "2-g ").replace("3-g", "3-g ").replace("4-g", "4-g ").replace("5-g", "5-g ").replace("6-g", "6-g ").replace("7-g", "7-g ").replace("8-g", "8-g ").replace("9-g", "9-g ")
    text = text.replace("0 -g", "0-g ").replace("1 -g", "1-g ").replace("2 -g", "2-g ").replace("3 -g", "3-g ").replace("4 -g", "4-g ")
    text = text.replace("0 mg", "0-mg ").replace("1 mg", "1-mg ").replace("2 mg", "2-mg ").replace("3 mg", "3-mg ").replace("4 mg", "4-mg ").replace("5 mg", "5-mg ").replace("6 mg", "6-mg ").replace("7 mg", "7-mg ").replace("8 mg", "8-mg").replace("9 mg", "9-mg ")
    text = text.replace("milligs","millig").replace("0millig", "0-mg ").replace("1millig", "1-mg ").replace("2millig", "2-mg ").replace("3millig", "3-mg ").replace("4millig", "4-mg ").replace("5millig", "5-mg ").replace("6millig", "6-mg ").replace("7millig", "7-mg ").replace("8millig", "8-mg ").replace("9millig", "9-mg ")
    text = text.replace("0 millig", "0-mg ").replace("1 millig", "1-mg ").replace("2 millig", "2-mg ").replace("3 millig", "3-mg ").replace("4 millig", "4-mg ").replace("5 millig", "5-mg ").replace("6 millig", "6-mg ").replace("7 millig", "7-mg ").replace("8 millig", "8-mg ").replace("9 millig", "9-mg ")
    text = text.replace("0 g", "0-g ").replace("1 g", "1-g ").replace("2 g", "2-g ").replace("3 g", "3-g ").replace("4 g", "4-g ").replace("5 g", "5-g ").replace("6 g", "6-g ").replace("7 g", "7-g ").replace("8 g", "8-g ").replace("9 g", "9-g ")
    text = text.replace("0kg", "0-kg ").replace("1kg", "1-kg ").replace("2kg", "2-kg ").replace("3kg", "3-kg ").replace("4kg", "4-kg ").replace("5kg", "5-kg ").replace("6kg", "6-kg ").replace("7kg", "7-kg ").replace("8kg", "8-kg ").replace("9kg", "9-kg ")
    text = text.replace("0 kg", "0-kg ").replace("1 kg", "1-kg ").replace("2 kg", "2-kg ").replace("3 kg", "3-kg ").replace("4 kg", "4-kg ").replace("5 kg", "5-kg ").replace("6 kg", "6-kg ").replace("7 kg", "7-kg ").replace("8 kg", "8-kg ").replace("9 kg", "9-kg ")
    text = text.replace("0oz", "0-oz ").replace("1oz", "1-oz ").replace("2oz", "2-oz ").replace("3oz", "3-oz ").replace("4oz", "4-oz ").replace("5oz", "5-oz ").replace("6oz", "6-oz ").replace("7oz", "7-oz ").replace("8oz", "8-oz ").replace("9oz", "9-oz ").replace(" 0z","-oz ").replace(" fl. oz","-oz ").replace("-oz","-oz(qunatity) ")
    text = text.replace("0 oz", "0-oz ").replace("1 oz", "1-oz ").replace("2 oz", "2-oz ").replace("3 oz", "3-oz ").replace("4 oz", "4-oz ").replace("5 oz", "5-oz ").replace("6 oz", "6-oz ").replace("7 oz", "7-oz ").replace("8 oz", "8-oz ").replace("9 oz", "9-oz ")
    text = text.replace("0ug", "0-ug ").replace("1ug", "1-ug ").replace("2ug", "2-ug ").replace("3ug", "3-ug ").replace("4ug", "4-ug ").replace("5ug", "5-ug ").replace("6ug", "6-ug ").replace("7ug", "7-ug ").replace("8ug", "8-ug ").replace("9ug", "9-ug ").replace("~ug","-ug ")
    text = text.replace("0 ug", "0-ug ").replace("1 ug", "1-ug ").replace("2 ug", "2-ug ").replace("3 ug", "3-ug ").replace("4 ug", "4-ug ").replace("5 ug", "5-ug ").replace("6 ug", "6-ug ").replace("7 ug", "7-ug ").replace("8 ug", "8-ug ").replace("9 ug", "9-ug ")
    text = text.replace("0ml", "0-ml ").replace("1ml", "1-ml ").replace("2ml", "2-ml ").replace("3ml", "3-ml ").replace("4ml", "4-ml ").replace("5ml", "5-ml ").replace("6ml", "6-ml ").replace("7ml", "7-ml ").replace("8ml", "8-ml ").replace("9ml", "9-ml ")
    text = text.replace("0 ml", "0-ml ").replace("1 ml", "1-ml ").replace("2 ml", "2-ml ").replace("3 ml", "3-ml gorilla").replace("4 ml", "4-ml ").replace("5 ml", "5-ml ").replace("6 ml", "6-ml ").replace("7 ml", "7-ml ").replace("8 ml", "8-ml ").replace("9 ml", "9-ml ")
    text = text.replace("mcg's","mcg").replace("0mcg", "0-mcg ").replace("1mcg", "1-mcg ").replace("2mcg", "2-mcg ").replace("3mcg", "3-mcg ").replace("4mcg", "4-mcg ").replace("5mcg", "5-mcg ").replace("6mcg", "6-mcg ").replace("7mcg", "7-mcg ").replace("8mcg", "8-mcg ").replace("9mcg", "9-mcg ")
    text = text.replace("0 mcg", "0-mcg ").replace("1 mcg", "1-mcg ").replace("2 mcg", "2-mcg ").replace("3 mcg", "3-mcg gorilla").replace("4 mcg", "4-mcg ").replace("5 mcg", "5-mcg ").replace("6 mcg", "6-mcg ").replace("7 mcg", "7-mcg ").replace("8 mcg", "8-mcg ").replace("9 mcg", "9-mcg ")
    text = text.replace("ounces", "ounce").replace("0ounce", "0-ounce ").replace("1ounce", "1-ounce ").replace("2ounce", "2-ounce ").replace("3ounce", "3-ounce ").replace("4ounce", "4-ounce ").replace("5ounce", "5-ounce ").replace("6ounce", "6-ounce ").replace("7ounce", "7-ounce ").replace("8ounce", "8-ounce ").replace("9ounce", "9-ounce ")
    text = text.replace("th ounce", "-ounce").replace("0 ounce", "0-ounce ").replace("1 ounce", "1-ounce ").replace("2 ounce", "2-ounce ").replace("3 ounce", "3-ounce gorilla").replace("4 ounce", "4-ounce ").replace("5 ounce", "5-ounce ").replace("6 ounce", "6-ounce ").replace("7 ounce", "7-ounce ").replace("8 ounce", "8-ounce ").replace("9 ounce", "9-ounce ")
    text = text.replace("0lb", "0-pound ").replace("1lb", "1-pound ").replace("2lb", "2-pound ").replace("3lb", "3-pound ").replace("4lb", "4-pound ").replace("5lb", "5-pound ").replace("6lb", "6-pound ").replace("7lb", "7-pound ").replace("8lb", "8-pound ").replace("9lb", "9-pound ").replace("-lb","-pound ")
    text = text.replace("0 lb", "0-pound ").replace("1 lb", "1-pound ").replace("2 lb", "2-pound ").replace("3 lb", "3-pound ").replace("4 lb", "4-pound ").replace("5 lb", "5-pound ").replace("6 lb", "6-pound ").replace("7 lb", "7-pound ").replace("8 lb", "8-pound ").replace("9 lb", "9-pound ").replace(" lb", "-pound ")
    text = text.replace("pounds","pound").replace(" pound", "-pound ").replace("lb's", "pound").replace("lbs", "pound").replace("lb gorilla", "1-pound gorilla ").replace("1 lb", "1-pound(quanitity) ").replace("2 lb", "2-pound(quanitity) ").replace("3 lb", "3-pound(quanitity) ").replace("4 lb", "4-pound(quanitity) ").replace("5 lb", "5-pound(quanitity) ").replace("6 lb", "6-pound(quanitity) ").replace("7 lb", "7-pound(quanitity) ").replace("8 lb", "8-pound(quanitity) ").replace("9 lb", "9-pound(quanitity) ")
    text = text.replace("+ mg ", "-mg ").replace("+ug", "-ug ").replace("]g", "-g ").replace("th oz","-ounce ")
    text = text.replace("pints","pint").replace("0pint", "0-pint ").replace("1pint", "1-pint ").replace("2pint", "2-pint ").replace("3pint", "3-pint ").replace("4pint", "4-pint ").replace("5pint", "5-pint ").replace("6pint", "6-pint ").replace("7pint", "7-pint ").replace("8pint", "8-pint ").replace("9pint", "9-pint ")
    text = text.replace("0 pint", "0-pint ").replace("1 pint", "1-pint ").replace("2 pint", "2-pint ").replace("3 pint", "3-pint ").replace("4 pint", "4-pint ").replace("5 pint", "5-pint ").replace("6 pint", "6-pint ").replace("7 pint", "7-pint ").replace("8 pint", "8-pint ").replace("9 pint", "9-pint ")
    text = text.replace("kilog","kilo").replace("0kilo", "0-kilo ").replace("1kilo", "1-kilo ").replace("2kilo", "2-kilo ").replace("3kilo", "3-kilo ").replace("4kilo", "4-kilo ").replace("5kilo", "5-kilo ").replace("6kilo", "6-kilo ").replace("7kilo", "7-kilo ").replace("8kilo", "8-kilo ").replace("9kilo", "9-kilo ")
    text = text.replace("kilos","kilo").replace("0 kilo", "0-kilo ").replace("1 kilo", "1-kilo ").replace("2 kilo", "2-kilo ").replace("3 kilo", "3-kilo ").replace("4 kilo", "4-kilo ").replace("5 kilo", "5-kilo ").replace("6 kilo", "6-kilo ").replace("7 kilo", "7-kilo ").replace("8 kilo", "8-kilo ").replace("9 kilo", "9-kilo ")
    text = text.replace("liters", "liter").replace("liter's", "liter").replace("0liter", "0-liter ").replace("1liter", "1-liter ").replace("2liter", "2-liter ").replace("3liter", "3-liter ").replace("4liter", "4-liter ").replace("5liter", "5-liter ").replace("6liter", "6-liter ").replace("7liter", "7-liter ").replace("8liter", "8-liter ").replace("9liter", "9-liter ")
    text = text.replace("0 liter", "0-liter ").replace("1 liter", "1-liter ").replace("2 liter", "2-liter ").replace("3 liter", "3-liter ").replace("4 liter", "4-liter ").replace("5 liter", "5-liter ").replace("6 liter", "6-liter ").replace("7 liter", "7-liter ").replace("8 liter", "8-liter ").replace("9 liter", "9-liter ")
    
    # Changing " x 1 " to "1-x "
    text_possibilities = [str(" x " + str(possibility) + " ") for possibility in possibilites]
    text_replacements = [str(str(possibility) + "-x ") for possibility in possibilites]
    for i in range(len(text_replacements)):
        text = text.replace(text_possibilities[i], text_replacements[i])

    # Changing " x1 " to "1-x "
    text_possibilities = [" x" + str(str(possibility) + " ") for possibility in possibilites]
    text_replacements = [str(str(possibility) + "-x ") for possibility in possibilites]
    for i in range(len(text_replacements)):
        text = text.replace(text_possibilities[i], text_replacements[i])
    
    text = text.replace(" x 1 ", "1-x ").replace(" x 10 ", "10-x ").replace(" x 12 ", "12-x ").replace(" x 100 ", "100-x ").replace(" x 1000 ", "1000-x ").replace(" x 10000 ", "10000-x ")
    text = text.replace("0 x", "0-x ").replace("1 x", "1-x ").replace("2 x", "2-x ").replace("3 x", "3-x ").replace("4 x", "4-x ").replace("5 x", "5-x ").replace("6 x", "6-x ").replace("7 x", "7-x ").replace("8 x", "8-x ").replace("9 x", "9-x ")
    text = text.replace("0x", "0-x ").replace("1x", "1-x ").replace("2x", "2-x ").replace("3x", "3-x ").replace("4x", "4-x ").replace("5x", "5-x ").replace("6x", "6-x ").replace("7x", "7-x ").replace("8x", "8-x ").replace("9x", "9-x ")
    text = text.replace("0 seeds pack", "0-pack ").replace("1 seeds pack", "1-pack ").replace("2 seeds pack", "2-pack ").replace("3 seeds pack", "3-pack ").replace("4 seeds pack", "4-pack ").replace("5 seeds pack", "5-pack ").replace("6 seeds pack", "6-pack ").replace("7 seeds pack", "7-pack ").replace("8 seeds pack", "8-pack ").replace("9 seeds pack", "9-pack ")
    text = text.replace("0 seeds", "0-seeds ").replace("1 seeds", "1-seeds ").replace("2 seeds", "2-seeds ").replace("3 seeds", "3-seeds ").replace("4 seeds", "4-seeds ").replace("5 seeds", "5-seeds ").replace("6 seeds", "6-seeds ").replace("7 seeds", "7-seeds ").replace("8 seeds", "8-seeds ").replace("9 seeds", "9-seeds ")
    text = text.replace("0 regular seeds", "0-seeds ").replace("1 regular seeds", "1-seeds ").replace("2 regular seeds", "2-seeds ").replace("3 regular seeds", "3-seeds ").replace("4 regular seeds", "4-seeds ").replace("5 regular seeds", "5-seeds ").replace("6 regular seeds", "6-seeds ").replace("7 regular seeds", "7-seeds ").replace("8 regular seeds", "8-seeds ").replace("9 regular seeds", "9-seeds ")
    text = text.replace("0 pack", "0-pack ").replace("1 pack", "1-pack ").replace("2 pack", "2-pack ").replace("3 pack", "3-pack ").replace("4 pack", "4-pack ").replace("5 pack", "5-pack ").replace("6 pack", "6-pack ").replace("7 pack", "7-pack ").replace("8 pack", "8-pack ").replace("9 pack", "9-pack ")
    text = text.replace("0pack", "0-pack ").replace("1pack", "1-pack ").replace("2pack", "2-pack ").replace("3pack", "3-pack ").replace("4pack", "4-pack ").replace("5pack", "5-pack ").replace("6pack", "6-pack ").replace("7pack", "7-pack ").replace("8pack", "8-pack ").replace("9pack", "9-pack ")
    text = text.replace("0 tabs", "0-tabs ").replace("1 tabs", "1-tablets ").replace("2 tabs", "2-tablets ").replace("3 tabs", "3-tablets ").replace("4 tabs", "4-tablets ").replace("5 tabs", "5-tablets ").replace("6 tabs", "6-tablets ").replace("7 tabs", "7-tablets ").replace("8 tabs", "8-tablets ").replace("9 tabs", "9-tablets ")
    text = text.replace("0tabs", "0-tablets ").replace("1tabs", "1-tablets ").replace("2tabs", "2-tablets ").replace("3tabs", "3-tablets ").replace("4tabs", "4-tablets ").replace("5tabs", "5-tablets ").replace("6tabs", "6-tablets ").replace("7tabs", "7-tablets ").replace("8tabs", "8-tablets ").replace("9tabs", "9-tablets ").replace("+ tabs", "-tablets ")
    text = text.replace("0 tablets", "0-tablets ").replace("1 tablets", "1-tablets ").replace("2 tablets", "2-tablets ").replace("3 tablets", "3-tablets ").replace("4 tablets", "4-tablets ").replace("5 tablets", "5-tablets ").replace("6 tablets", "6-tablets ").replace("7 tablets", "7-tablets ").replace("8 tablets", "8-tablets ").replace("9 tablets", "9-tablets ")
    text = text.replace("0tablets", "0-tablets ").replace("1tablets", "1-tablets ").replace("2tablets", "2-tablets ").replace("3tablets", "3-tablets ").replace("4tablets", "4-tablets ").replace("5tablets", "5-tablets ").replace("6tablets", "6-tablets ").replace("7tablets", "7-tablets ").replace("8tablets", "8-tablets ").replace("9tablets", "9-tablets ")
    text = text.replace("0pills", "0-pills ").replace("1pills", "1-pills ").replace("2pills", "2-pills ").replace("3pills", "3-pills ").replace("4pills", "4-pills ").replace("5pills", "5-pills ").replace("6pills", "6-pills ").replace("7pills", "7-pills ").replace("8pills", "8-pills ").replace("9pills", "9-pills ")
    text = text.replace("0 pills", "0-pills ").replace("1 pills", "1-pills ").replace("2 pills", "2-pills ").replace("3 pills", "3-pills ").replace("3 pills", "3-pills ").replace("4 pills", "4-pills ").replace("5 pills", "5-pills ").replace("6 pills", "6-pills ").replace("7 pills", "7-pills ").replace("8 pills", "8-pills ").replace("9 pills", "9-pills ")
    
    # Getting the times tag from the 1st token in text data
    text_temp = text.split(" ")
    if "-" not in text_temp[0]:
        if text_temp[0].isdigit():
            text_temp[0] = text_temp[0] + "-x "
    else:
        pass
    
    text = " ".join(text_temp)
    
    # Tagging un-relevant knowledge
    text = text.replace("0 day", "zero day ").replace("1 day", "one day").replace("2 day", "two day").replace("3 day", "three day").replace("4 day", "four day").replace("5 day", "five day").replace("6 day", "six day").replace("7 day", "seven day").replace("8 day", "eight day").replace("9 day", "nine day")
    text = text.replace("0day", "zero-day ").replace("1day", "one-day").replace("2 day", "two-day").replace("3 day", "three-day").replace("4 day", "four-day").replace("5 day", "five-day").replace("6 day", "six-day").replace("7 day", "seven-day").replace("8 day", "eight-day").replace("9 day", "nine-day")
    
    # Additional cleaning
    text = text.replace("-"," - ").replace("|", " | ").replace("\\", " / ").replace("+", " + ").replace("[", " [ ").replace("]", " ] ")
    text = text.replace("("," ( ").replace(")", " ) ")
    text = text.replace(":", " : ").replace("!", " ! ").replace("*"," * ").replace("/", " / ").replace("~", " ~ ").replace("=", " = ").replace("—", " — ").replace(".", " . ").replace(",", " , ")
    text = text.replace("€", " € ").replace("£", " £ ").replace("$", " $ ")

    # Creating space between the special symbols
    text = pat.sub(" \\1 ", text)

    # Removing the whitespaces
    text = text.split(" ")
    text = list(filter(None, text))
    text = " ".join(text)
    return text

# %% Helper function to clean, merge, and tokenize data as per the darkBert model
def clean_and_merge_data_for_tokenization(data):
    data = merge_and_create_dataframe(data)
    data_text = list(data['text'])
    
    data_text = [re.findall(r"\S+", data) for data in data_text]
    # Removing the links
    data_text = [[token if "http" not in token else "LINK" for token in data] for data in data_text]
    data_text = [" ".join(data) for data in data_text]
    # Cleaning the data
    # data_text = [clean_data(data) for data in data_text]

    data["text"] = data_text

    return data

def convert_unnecessary_predictions(data, count_threshold):
    ads_count = dict(Counter(data['prediction']))
    predictions_to_be_converted = [prediction for prediction, count in ads_count.items() if count < count_threshold]
    data['prediction'] = data['prediction'].replace(predictions_to_be_converted, 'others')
    return data

def save_data_to_text_file(train_data, test_data, data_dir):
    train_data = merge_and_create_dataframe(train_data)
    test_data = merge_and_create_dataframe(test_data)
    train_data = list(train_data['text'].unique())
    test_data = list(test_data['text'].unique())

    train_data = [re.findall(r"\S+", data) for data in train_data]
    # Removing the links
    train_data = [[token if "http" not in token else "LINK" for token in data] for data in train_data]
    train_data = [" ".join(data) for data in train_data]
    
    test_data = [re.findall(r"\S+", data) for data in test_data]
    # Removing the links
    test_data = [[token if "http" not in token else "LINK" for token in data] for data in test_data]
    test_data = [" ".join(data) for data in test_data]

    with open(os.path.join(data_dir, 'dw_train_lm.txt'), 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write("%s\n" % item)
        f.close()

    with open(os.path.join(data_dir, 'dw_test_lm.txt'), 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write("%s\n" % item)
        f.close()

