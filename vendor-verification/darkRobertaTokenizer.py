# %% Importing Libraries
import os, glob
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

# Creating directory to save tokenizer
Path(os.path.join(os.getcwd(), "../models/Roberta")).mkdir(parents=True, exist_ok=True)

# %% Initializing tokenizer
# Getting all the advertisements from training and test data
ads = [str(x) for x in Path(os.path.join(os.getcwd(), "../data/")).glob('**/*.txt')]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(ads, min_frequency=5, vocab_size=135_000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer.save_model(os.path.join(os.getcwd(), "../models/Roberta"))