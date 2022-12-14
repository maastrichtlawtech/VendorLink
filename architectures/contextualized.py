"""
Python version : 3.8
Description : Loads hugging face prertained checkpoints of contextualized models
"""

# %% Importing libraries
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LoadHuggingFaceCheckpoints(object):

    def __init__(self, model_name, vendor_to_idx_dict):
        self.model_name = model_name
        self.vendor_to_idx_dict = vendor_to_idx_dict

    def load_model(self):
        if self.model_name == "bert" :    
            from transformers import BertTokenizer, BertForSequenceClassification
            # Load the BERT tokenizer and model
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased', truncation=True, do_lower_case=True)
            # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True, do_lower_case=False)
            model = BertForSequenceClassification.from_pretrained("bert-base-cased",
                                                        num_labels=len(self.vendor_to_idx_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False).to(device)

        elif self.model_name == "roberta":
            from transformers import RobertaTokenizer, RobertaForSequenceClassification
            # Load the RoBERTa tokenizer and model
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=False)
            model = RobertaForSequenceClassification.from_pretrained("roberta-base",
                                                        num_labels=len(self.vendor_to_idx_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False).to(device)

        elif self.model_name == "electra":
            from transformers import ElectraTokenizer, ElectraForSequenceClassification
            # Load the Electra tokenizer and model
            tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', truncation=True, do_lower_case=False)
            model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator',
                                                        num_labels=len(self.vendor_to_idx_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False).to(device)
            
        elif self.model_name == 'distill':
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            # Load the Distill bert tokenizer and model
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
            model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-cased",
                                                        num_labels=len(self.vendor_to_idx_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False).to(device)

        elif self.model_name == "gpt2":
            from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
            # Load the GPT2 tokenizer and model
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2', truncation=True, do_lower_case=True)
            model = GPT2ForSequenceClassification.from_pretrained('gpt2',
                                                        num_labels=len(self.vendor_to_idx_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False).to(device)

        else:
            raise Exception("model argument can only be one amongst the bert, roberta, electra, distill, or gpt2")

        return tokenizer, model
