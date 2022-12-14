# VendorLink: A (Semi-)Supervised NLP approach for Identifying & Linking Vendor Migrants & Aliases on Darknet Markets

# Dataset
For reproducibility purposes, we conduct our analyses on publically available Darknet advertisements datasets from Agora, Alphabay, Dreams, Traderoute, Valhalla, and Berlusconi non-anonymous markets. All the datasets are hosted by IMPACT cyber trust portal and Kaggle. They can be downloaded through the links below:

1) [Alphabay marketplace, 2017-18](https://github.com/user/repo/blob/branch/other_file.md) : AlphaBay was an online darknet market launched in September 2014, pre-launched in November 2014, and officially launched on December 22, 2014, an onion service of the Tor network. In 2017, it was shut down after a law enforcement action as a part of Operation Bayonet (along with Hansa market) in the United States, Canada, and Thailand. At its demise in July 2017, AlphaBay had over 400,000 users. The non-anonymous dataset was collected from a publicly available website over two and a half years, 2014-2017. The dataset consists of 1,771,258 advertisements from 6,250 unique vendors.
2) [Dream, Traderoute, Berlusconi and Valhalla marketplaces, 2017-2018](http://dx.doi.org/10.23721/116/1503879) : Dream Market was an online darknet market founded in late 2013. Following the seizures and shutdowns of the AlphaBay and Hansa markets, Traderoute and Dream Market became predominant marketplaces on the Dark Web. During the time, Dream Market was estimated as the second-largest darknet marketplace, with AlphaBay being the largest and Hansa the third-largest. After Operation Bayonet, many vendors and buyers from AlphaBay and Hansa communities migrated to Dream Market. At the time, Dream Market was reported to have 57,000 listings for drugs and 4,000 listings for opioids. The marketplace sold a variety of content, including drugs, stolen data, and counterfeit consumer goods, all using cryptocurrency. In addition, dream provided an escrow service, with disputes handled by staff. The market also had accompanying forums hosted on a different URL, where buyers, vendors, and other community members could interact. Eventually, Administrator and prolific vendor Gal Vallerius was arrested in August 2017 and the site shut down on April 2019. The non-anonymous Dream dataset collected by Carnegie Mellon University has 1,816,854 listings by 5,780 vendors between 2016 -18.
3) [Silk Road marketplace, 2012-13](http://dx.doi.org/10.23721/116/1406256) : Silk Road-1 was the first modern online black market notoriously known for selling illegal drugs. The website was first launched in February 2011, with a limited number of new seller accounts available. Later, new sellers acquired other accounts in an auction for a fixed fee. Finally, on October 2013, the FBI seized the website and arrested Ross Ulbricht for being the site's pseudonymous founder, "Dread Pirate Roberts." The non-anonymous Silk Road-1 dataset from Carnegie Mellon University has 1,065,810 listings by 2,872 vendors between 2012-13.
4) [Agora marketplace, 2014-15](https://www.kaggle.com/datasets/philipjames11/dark-net-marketplace-drug-data-agora-20142015) : Agora was operated on the Tor network between 2013-15. After the demise of Evolution and Silk Road 2.0, Agora became the largest marketplace in March 2015. The Kaggle data parse contains drugs, weapons, books, and services. Duplicate listings have been removed, and prices have been averaged for duplicates. The data is in a CSV file and has over 100,000 unique listings.

<p align="center">
  <img src="/docs/Images/data.png" width="275" height="350">
</p>

# Setup
This repository is tested on Python 3.8+. First, you should install a virtual environment:
```
python3 -m venv .venv/dw
source .venv/dw/bin/activate
```

Then, you can install all dependencies:
```
pip install -r requirements.txt
```

Additionally, you should also consider installing the pre-trained English [fastText](https://fasttext.cc/docs/en/crawl-vectors.html#models) embeddings if you would like to initialize the BiGRU classifier baseline with Fasttext embeddings. Although, the experiments show that our BiGRU classifier after being initialized with the pre-trained embeddings of benchmark classifer outperforms the one with Fasttext embeddings by drastic margin.

# Experiments

Prior to running experiments, let us first merge the required features from different input files and bring them together. By running the command below, we filter out all the unnecessary features from the item and feedback files and merge them to create preprocessed_alpha.csv, preprocessed_dreams.csv, preprocessed_silk.csv, and preprocessed_agora.csv files in the data directory. For all our upcoming experiments, we load these files as input to our scripts.

```
python3 utilities/formatData.py
```

#### Closed-Set Vendor Verification Task : Verifying / Classifying migrating vendors across markets
In this research, we first establish a benchmark by performing the vendor verification task using a BERT-based-cased classifier on the Alphabay-Dreams-Silk Road dataset. Compared to other baselines (please refer to them), the BERT-based-cased classifier outperforms (refer to the table below) by a reasonable margin. 


<p align="center">
  <img src="/docs/Images/baselines.png" width="305" height="355">
</p>

In order to train the BERT-cased classifier, run:

```
python3 vendor-verification/contextualized_models.py --model bert --save_dir ../models/
```

#### Open-Set Vendor Identification Task : Computing text similarity to verify existing migrants and identify potential aliases
In order to compute similarity between vendor advertisements, we first extract sentence representations from the above-trained classifier and save it in a pickled file. To extract the sentence representations from the trained model, run: 

```
python3 vendor-identification/generate_vendorRepresentations.py --model_dir ../models/bert  --pickle_dir ../pickled/ --load_model pretrained_bert_classifier.model
```

Then to compute similarity between the advertisements of vendors, run (Make sure to set vendor_list parameter in compute_similarity_between_vendors function to None to compute similarity in the advertisements of all the vendors):

```
python3 vendor-identification/compute_similarity.py --model_dir ../models/bert  --pickle_dir ../pickled/ --load_model pretrained_bert_classifier.model
```
