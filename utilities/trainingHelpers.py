"""
Python version : 3.8
Description : Contains the training and evaluation helper functions for different models
"""

# %% Importing libraries
import os
from tqdm import tqdm
import numpy as np

from sklearn.metrics import classification_report

import torch

# %% Loading custom libraries 
sys.path.append('../metrics/')
from performance import f1_score_func

# Initializing the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% Helper functions

def trainBiGRUFasttext(model, iterator, optimizer, criterion):
    
    epoch_loss = 0    
    model.train()
    
    for batch in iterator:
        text, text_lengths = batch.text
        
        optimizer.zero_grad()
        predictions = model(text, text_lengths.cpu()).squeeze(1)
        loss = criterion(predictions, batch.label.to(torch.long))

        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()        

    return epoch_loss / len(iterator)

def evaluateBiGRUFasttext(model, iterator, criterion):
    
    #initialize every epoch
    epoch_loss = 0    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths.cpu()).squeeze(1)
            
            #compute loss
            loss = criterion(predictions, batch.label.to(torch.long))
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def extract_layer_representations_from_bert_layer(model, layer, batch, device):    
    # Initializing the layers with zero activation with (layer, mean embedding)
    
    model.eval()
    batch = tuple(b.to(device) for b in batch)
    inputs = {'input_ids':      batch[0],
              'attention_mask': batch[1],
              'labels':         batch[2],
             }
    
    with torch.no_grad():        
        outputs = model(**inputs)
        if layer == "last":
            hidden_repr = outputs['hidden_states'][-1]
        elif layer == "second-to-last":
            hidden_repr = outputs['hidden_states'][-2]
        elif layer == "first":
            hidden_repr = outputs['hidden_states'][0]
        elif layer == "weighted-sum-last-four":
            hidden_repr = torch.stack(outputs['hidden_states'][-4:]).sum(0)
        elif layer == "all":
            hidden_repr = torch.stack(outputs['hidden_states'][0:]).sum(0)
        elif layer == "concat-last-4":
            hidden_repr = torch.cat(outputs['hidden_states'][-4:], dim=1)
        else:
            raise Exception("Layer can only be last, second-to-last, first, weighted-sum-last-four, all, or concat-last-4")
        
        # Stacking all the layers of the model
        # hidden_repr = torch.stack(hidden_repr, dim=0).detach().cpu()
        # Concatinating all batches
        # concat_hidden_repr.append(hidden_repr)

        mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_repr.size()).float()
        # Each vector above represents a single token attention mask - each token now has a vector of size 768 
        # representing it's attention_mask status. Then we multiply the two tensors to apply the attention mask
        masked_embeddings = hidden_repr.to(device) * mask.to(device)
        
    return masked_embeddings

# Save and Load Functions
def save_checkpoint(save_path, model, optimizer, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


# Training Function
def trainGRUBERT(model, pretrained_model, layer, optimizer, criterion, train_loader, 
                valid_loader, num_epochs, max_seq_len, batch_size, eval_every, file_path, device, 
                best_valid_loss = float("Inf")):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    for epoch in range(num_epochs):
        for idx, batch in enumerate(train_loader):
            text = extract_layer_representations_from_bert_layer(pretrained_model, layer, batch, device)
            # training loop
            model.train()
            labels = batch[2]
            labels = torch.autograd.Variable(labels).long()

            # Putting tensors and model on GPUs, if available
            if torch.cuda.is_available():
                model.to(device)
                text = text.to(device)
                labels = labels.to(device)

            # One of the batch returned by Dataloader can have a length different than batch_size. 
            if (text.size()[0] is not batch_size):
                continue

            output = model(text)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # validation loop
                    for idx, batch in enumerate(valid_loader):
                        text = extract_layer_representations_from_bert_layer(pretrained_model, layer, batch, device)
                        labels = batch[2]
                        labels = torch.autograd.Variable(labels).long()

                        # Putting tensors and model on GPUs, if available
                        if torch.cuda.is_available():
                            model.to(device)
                            text = text.to(device)
                            labels = labels.to(device)

                        # One of the batch returned by Dataloader can have a length different than batch_size. 
                        if (text.size()[0] is not batch_size):
                            continue
                            
                        output = model(text)
                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()
                        
                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)
                        
                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()
                
                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
                    
    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
    
def evaluateGRUBERT(model, pretrained_model, layer, valid_loader, batch_size, device):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(valid_loader):
            text = extract_layer_representations_from_bert_layer(pretrained_model, layer, batch, device)
            labels = batch[2]
            labels = torch.autograd.Variable(labels).long()

            # Putting tensors and model on GPUs, if available
            if torch.cuda.is_available():
                model.to(device)
                text = text.to(device)
                labels = labels.to(device)

            # One of the batch returned by Dataloader can have a length different than batch_size. 
            if (text.size()[0] is not batch_size):
                continue

            output = model(text)
            output_classes = torch.argmax(output, dim=1)
            
            # output = (output > threshold).int()
            y_pred.extend(output_classes.tolist())
            y_true.extend(labels.tolist())
            

    print('Classification Report:')
    print(classification_report(np.array(y_true), np.array(y_pred), digits=4))

def trainTransformers(model, vendor_to_idx_dict, dataloader_train, dataloader_validation, epochs, optimizer, scheduler, device, save_dir):
    print("Training model ....")
    for epoch in tqdm(range(1, epochs+1)):
    
        model.train()
        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[2]}       

            outputs = model(**inputs)
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
            
        torch.save(model.state_dict(), os.path.join(save_dir, "epoch_" + str(epoch) + ".model" ))
            
        tqdm.write(f'\nEpoch {epoch}')
        loss_train_avg = loss_train_total/len(dataloader_train)            
        tqdm.write(f'Training loss: {loss_train_avg}')
        
        val_loss, predictions, true_vals = evaluateTransformers(model, dataloader_validation, device)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')

def evaluateTransformers(model, dataloader_val, device):
    print("Evaluating model ....")
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals