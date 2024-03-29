# VendorLink: A (Semi-)Supervised NLP approach for Identifying & Linking Vendor Migrants & Aliases on Darknet Markets

The anonymity on the Darknet allows vendors to stay undetected by using multiple vendor aliases or frequently migrating between markets. Consequently, illegal markets and their connections are challenging to uncover on the Darknet. To identify relationships between illegal markets and their vendors, we propose VendorLink, an NLP-based approach that examines writing patterns to verify, identify, and link unique vendor accounts across text advertisements (ads) on seven public Darknet markets. In contrast to existing literature, VendorLink utilizes the strength of supervised pre-training to perform closed-set vendor verification, open-set vendor identification, and low-resource domain adaption tasks. Altogether, our approach can help Law Enforcement Agencies (LEA) make more informed decisions by verifying and identifying migrating vendors and their potential aliases with state-of-the-art (SOTA) performance on both existing and emerging low-resource (LR) Darknet markets.

![(i) Closed-Set Vendor Verification Task: A supervised-pretraining task that performs classification in a closed-set environment setting to verify unique vendor migrants across known markets (ii) Open-set Vendor Identification Task: A text-similarity task in open-set environment setting that utilizes embeddings from the pre-trained classifier to verify known vendors and identify potential-aliases (iii) Low-resource domain adaptation task: A knowledge-transfer task to adapt new domain knowledge and verify migrants in a closed-set environment setting across low-resource emerging markets.](docs/Images/vendorLink.png)
(i) __Closed-Set Vendor Verification Task:__ A supervised-pretraining task that performs classification in a closed-set environment setting to verify unique vendor migrants across known markets (ii) __Open-set Vendor Identification Task:__ A text-similarity task in open-set environment setting that utilizes embeddings from the pre-trained classifier to verify known vendors and identify potential-aliases (iii) __Low-resource domain adaptation task:__ A knowledge-transfer task to adapt new domain knowledge and verify migrants in a closed-set environment setting across low-resource emerging markets.

# Dataset
For reproducibility purposes, we conduct our analyses on publically available Darknet advertisement datasets from Agora, Alphabay, Dreams, Traderoute, Valhalla, and Berlusconi non-anonymous markets. All the datasets are hosted by IMPACT cyber trust portal and Kaggle. They can be downloaded through the links below:

1) [Alphabay marketplace, 2017-18](http://dx.doi.org/10.23721/116/1462165) : AlphaBay was an online darknet market launched in September 2014, pre-launched in November 2014, and officially launched on December 22, 2014, an onion service of the Tor network. In 2017, it was shut down after a law enforcement action as a part of Operation Bayonet (along with Hansa market) in the United States, Canada, and Thailand. At its demise in July 2017, AlphaBay had over 400,000 users. The non-anonymous dataset was collected from a publicly available website over two and a half years, 2014-2017. The dataset consists of 1,771,258 advertisements from 6,250 unique vendors.
2) [Dream, Traderoute, Berlusconi and Valhalla marketplaces, 2017-2018](http://dx.doi.org/10.23721/116/1503879) : Dream Market was an online darknet market founded in late 2013. Following the seizures and shutdowns of the AlphaBay and Hansa markets, Traderoute and Dream Market became predominant marketplaces on the Dark Web. During the time, Dream Market was estimated as the second-largest darknet marketplace, with AlphaBay being the largest and Hansa the third-largest. After Operation Bayonet, many vendors and buyers from AlphaBay and Hansa communities migrated to Dream Market. At the time, Dream Market was reported to have 57,000 listings for drugs and 4,000 listings for opioids. The marketplace sold a variety of content, including drugs, stolen data, and counterfeit consumer goods, all using cryptocurrency. In addition, dream provided an escrow service, with disputes handled by staff. The market also had accompanying forums hosted on a different URL, where buyers, vendors, and other community members could interact. Eventually, Administrator and prolific vendor Gal Vallerius was arrested in August 2017 and the site shut down on April 2019. The non-anonymous Dream dataset collected by Carnegie Mellon University has 1,816,854 listings by 5,780 vendors between 2016 -18.
3) [Silk Road marketplace, 2012-13](http://dx.doi.org/10.23721/116/1406256) : Silk Road-1 was the first modern online black market notoriously known for selling illegal drugs. The website was first launched in February 2011, with a limited number of new seller accounts available. Later, new sellers acquired other accounts in an auction for a fixed fee. Finally, on October 2013, the FBI seized the website and arrested Ross Ulbricht for being the site's pseudonymous founder, "Dread Pirate Roberts." The non-anonymous Silk Road-1 dataset from Carnegie Mellon University has 1,065,810 listings by 2,872 vendors between 2012-13.
4) [Agora marketplace, 2014-15](https://www.kaggle.com/datasets/philipjames11/dark-net-marketplace-drug-data-agora-20142015) : Agora was operated on the Tor network between 2013-15. After the demise of Evolution and Silk Road 2.0, Agora became the largest marketplace in March 2015. The Kaggle data parse contains drugs, weapons, books, and services. Duplicate listings have been removed, and prices have been averaged for duplicates. The data is in a CSV file and has over 100,000 unique listings.

<p align="center">
  <img src="docs/Images/data.png" width="225" height="260">
</p>

# Setup
This repository is tested on Python 3.8+. First, you should install a virtual environment:
```
python3 -m venv .venv/DW
source .venv/DW/bin/activate
```

Then, you can install all dependencies:
```
pip install -r requirements.txt
```

Please consider installing the pre-trained English [fastText](https://fasttext.cc/docs/en/crawl-vectors.html#models) embeddings if you would like to initialize the BiGRU classifier baseline with Fasttext embeddings. Although, our experiments show that the BiGRU classifier, initialized with the embeddings of the trained BERT-cased classifier, outperforms the one with Fasttext embeddings by a drastic margin.


# Experiments

Before running experiments, let us merge the required features from different input files and bring them together. By running the command below, we filter out all the unnecessary features from the item and feedback files and merge them to create preprocessed_alpha.csv, preprocessed_dreams.csv, preprocessed_silk.csv, and preprocessed_agora.csv files in the data directory. Then, we load these files for all our future experiments as input to our scripts.

```
python utilities/formatData.py
```

### Closed-Set Vendor Verification Task : Verifying / Classifying migrating vendors across markets
In this research, we first establish a benchmark by performing the vendor verification task using a BERT-based-cased classifier on the Alphabay-Dreams-Silk Road dataset. Compared to other baselines (please refer to them), the BERT-based-cased classifier outperforms (refer to the table below) by a reasonable margin. 


<p align="center">
  <img src="docs/Images/baselines.png" width="305" height="355">
</p>

In order to train the BERT-cased classifier, run:

```
python vendor-verification/contextualized_models.py --model bert --save_dir ../models/ --intent benchmark
```

### Open-Set Vendor Identification Task : Computing text similarity to verify existing migrants and identify potential aliases
In order to compute the similarity between vendor advertisements, we first extract style representations from the above-trained classifier and save them in a pickled file. To extract the style representations from the trained model, run: 

```
python3 vendor-identification/generate_vendorRepresentations.py --model_dir ../models/bert  --pickle_dir ../pickled/ --load_model pretrained_bert_classifier.model --layer weighted-sum-last-four
```

Then, to compute the similarity between the vendor advertisements, run (Make sure to set vendor_list parameter in compute_similarity_between_vendors function to None to compute similarity in the advertisements of all the vendors):

```
python vendor-identification/compute_similarity.py --model_dir ../models/bert  --pickle_dir ../pickled/ --load_model pretrained_bert_classifier.model
```

Finally, to visualize the vendor and their potential aliases, run:

```
python vendor-identification/plot_vendor_similarity.py --n_vendors 3 --plot_dir ../plots/
```

The script above should generate a plot with parent vendors on x-axis with their potential aliases (scatter markers) and advertisement similarity on y-axis. The color coding indicates the existence of plotted vendors on Alphabay (red and triangle-up-open-dot), Dreams (green and circle-open-dot), and Silk Road (blue and star-open-dot) markets.


<p align="center">
  <img src="docs/Images/similarity.png" width="500" height="350">
</p>

### Low-resource domain adaptation task: Utilizing knowledge-transfer to adapt to new domain knowledge and performing vendor verification on the emerging LR dataset.
In this research, we demonstrate that by applying knowledge transfer from the trained BERT-cased classifier to a 2-layered BiGRU, our trained model adapts new domain knowledge and outperforms (refer to the table below) all the established baseline in this research for an emerging LR, Valhalla-Berlusconi dataset. 

<p align="center">
  <img src="docs/Images/lr_exp.png" width="290" height="325">
</p>
                                                                  
To train our BiGRU classifier initialized with pre-trained BERT-cased embeddings, run 
                                                                  
```
python vendor-verification/transfer_BiGRU.py --data_to_train valhalla-berlusconi --load_model ../models/bert/pretrained_bert_classifier.model
```

# Citation

```bibtex
@inproceedings{saxena-etal-2023-vendorlink,
    title = "{V}endor{L}ink: An {NLP} approach for Identifying {\&} Linking Vendor Migrants {\&} Potential Aliases on {D}arknet Markets",
    author = "Saxena, Vageesh  and
      Rethmeier, Nils  and
      van Dijck, Gijs  and
      Spanakis, Gerasimos",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.481",
    doi = "10.18653/v1/2023.acl-long.481",
    pages = "8619--8639",
}
```
