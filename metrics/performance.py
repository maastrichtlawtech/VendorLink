"""
Python version: 3.8
Description: Contains helper functions to evaluate model performance
"""

# %% Loading libraries
import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import torch

# %% Metric functions
def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels, label_dict):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

def evaluate_and_generate_classification_report(model, iterator):
    prediction_list, label_list = [], []
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths.cpu()).squeeze(1)
            predictions = torch.argmax(predictions, dim=1)
            prediction_list.extend(predictions.tolist())
            label_list.extend(batch.label.to(torch.long).tolist())
        
    print(classification_report(prediction_list, label_list, digits=4))
