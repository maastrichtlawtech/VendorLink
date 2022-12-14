"""
Python version : 3.8
References : https://www.w3schools.com/tags/ref_byfunc.asp
Description : Performs scraping on online Darknet markets
"""

################################################## Importing Libraries #############################################
import os
import sys
import re
import glob
import argparse

import requests
from bs4 import BeautifulSoup

import spacy
from spacy_cld import LanguageDetector

#####################################################################################################################


################################################## Initializing the Parser ##########################################
parser = argparse.ArgumentParser(description="Dark Web Scraper")
parser.add_argument('--raw_data_dir', type=str,
                    default="../data/raw_data/Forums",help="path of raw data directory")
parser.add_argument('--processed_data_dir', type=str,
                    default="../data/processed_data",help="path of processed data directory")
args = parser.parse_args()
#####################################################################################################################


# Initializing working directories
raw_data_directory = glob.glob(os.path.join(args.raw_data_dir, '**/*.html'), recursive=True)
if not os.path.exists(args.processed_data_dir):
    os.makedirs(args.processed_data_dir)

# Loading Spacy Language detector model
nlp = spacy.load('en_core_web_sm')
language_detector = LanguageDetector()
nlp.add_pipe(language_detector)

def check_language_if_english(text):
    doc_ = nlp(text)
    if bool(doc_._.languages) :
        if doc_._.languages[0] == 'en':
            return True
        else:
            return False
    else:
        return True

sys.stdout.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format("domain-type", "domain-name", "log-name", "title", "description", "paragraphs", "headings", "listings", "link-text"))

for file_ in raw_data_directory:
    logname = file_.split("/")[-1]
    domain_name = file_.split("/")[-3]
    domain = file_.split("/")[-4]
    """print("Domain type:", domain)
    print("Domain name:", domain_name)
    print("Log name:", logname)"""
    try:
        with open(file_) as html_file:
            soup = BeautifulSoup(html_file, 'html.parser')

        # Getting the title of the webpage
        if soup.title:
            if check_language_if_english(soup.title.text) == True:
                title_ = soup.title.text.replace("\n", "").replace("\t", "").strip()
            else:
                title_ = None
        else:
            title_ = None
        """print("Title :", title_)"""

        # Getting all the text content from the webpage
        if check_language_if_english(soup.get_text()) == True:
            text_data = soup.get_text().split("\n")
            text_data = set([string.replace("\t","").strip() for string in text_data if string.strip() != ''])
        else:
            text_data = None
        """print("Text Data:", text_data)"""
        
        if soup.body:
            # Getting the paragraphs
            if soup.body.find('p'):
                paragraphs = soup.body.find_all('p')
                # Cleaning the data
                paragraphs = [paragraph.text.replace('\n','').replace('\t','').strip() for paragraph in paragraphs if check_language_if_english(paragraph.text) == True]
                # removing the empty strings
                paragraphs = set([string for string in paragraphs if string != ""])
                if len(paragraphs) == 0:
                    paragraphs = None
            else:
                paragraphs = None

            # Getting the headings
            if bool(soup.body.find_all(re.compile('^h[1-6]$'))) == True:
                headings = [heading.text.split('\n') for heading in soup.body.find_all(re.compile('^h[1-6]$')) if check_language_if_english(heading.text) == True]
                headings = [[string.replace("\t", "").strip() for string in heading if string.strip() != ''] for heading in headings] # stripping the white spaces
                headings = [item for sublist in headings for item in sublist] # flattening the list
                # removing the empty strings
                headings = set([string for string in headings if string != ""])
                if len(headings) == 0:
                    headings = None
            else:
                headings = None

            # Checking for form contents
            form_dict = {}
            form = soup.body.find('form')
            if form:
                if form.find('input'):
                    # Getting the input fields
                    input_fields = form.find_all('input', recursive=False)
                    input_fields = [field.text.replace('\n','').replace('\t','').strip() for field in input_fields]
                    form_dict['inputs'] = input_fields
                if form.find('label'):
                    # Getting the labels and its neighbouring spans
                    label = {lab.text.replace('\n','').replace('\t','').strip():lab.find_next_sibling().text.replace('\n','').replace('\t','').strip() for lab in form.select('label') if lab.find_next_sibling()}
                    form_dict['labels and spans'] = label
                if form.find('textarea'):
                    # Getting the textarea
                    textareas = form.find_all('textarea')
                    textareas = [textarea.text.replace('\n','').replace('\t','').strip() for textarea in textareas if check_language_if_english(textarea.text) == True]
                    form_dict['textarea'] = textareas
                else:
                    form_dict = None
            else:
                form_dict = None

            # Getting the listing Items
            if soup.body.find('li'):
                for list_ in soup.body.select('li'):
                    if list_.find_next_sibling() != None:
                        listings = {}
                        listings[list_.text.replace('\n','').replace('\t','').strip()] = list_.find_next_sibling().text.replace('\n','').replace('\t','').strip()
                    else:
                        listings = soup.body.find_all('li')
                        listings = [list_.text.replace('\n','').replace('\t','').strip() for list_ in listings if check_language_if_english(list_.text) == True]
                        # removing the empty strings
                        listings = set([string for string in listings if string != ""])
                        if len(listings) == 0:
                            listings = None
            else:
                listings = None
        
            # Getting the link texts
            if soup.body.find('a'):
                links = soup.body.find_all('a')
                link_text = set([link.text.replace('\n','').replace('\t','').strip() for link in links if check_language_if_english(link.text) == True])
                # removing the empty strings
                link_text = set([string for string in link_text if string != ""])
                if len(link_text) == 0:
                    link_text = None
            else:
                link_text = None

        else:
            paragraphs = None
            headings = None
            form_dict = None
            listings = None
            link_text = None

        sys.stdout.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(
            domain, domain_name, logname, title_, text_data, paragraphs, headings, listings, link_text))
        
        """print("paragraphs :", paragraphs)
        print("Headings :", headings)
        print("Form Details :", form_dict)
        print("Listings :", listings)
        print("Links text :", link_text)
        """

    except IOError:
        raise Exception("Cannot open file :", file_)