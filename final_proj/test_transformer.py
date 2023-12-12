import os
import argparse
from pathlib import Path
from models.transformer_model import Net, PositionalEncoding
import math
import json
from torch import nn
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import tensor
from torch import save
from torch.utils.data import DataLoader
import torch
import time
from test import data_parsing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
DATA_DIR = os.path.join(PARENT_DIR, 'fake_data')

notes_tensor = torch.load(os.path.join(DATA_DIR, 'train_notes_tensor.pt')).to(device)
chords_tensor = torch.load(os.path.join(DATA_DIR, 'train_chords_tensor.pt')).to(device)

dev_notes_tensor = torch.load(os.path.join(DATA_DIR, 'dev_notes_tensor.pt')).to(device)
dev_chords_tensor = torch.load(os.path.join(DATA_DIR, 'dev_chords_tensor.pt')).to(device)

train_dataset = [(notes_tensor[i], chords_tensor[i]) for i in range(len(chords_tensor))]

epochs = 50
BATCH_SIZE = 30
best_dev_acc = -1
best_checkpoint = None
best_epoch = -1
last_integer_accuracy_in_percent = -1
count_epochs = 0
SAVEFILE_NAME = ""


model = Net(
    vocab_size=23,
    d_model=256,
    nhead=8,  # the number of heads in the multiheadattention models
    dim_feedforward=50,  # the dimension of the feedforward network model in nn.TransformerEncoder
    num_layers=6,
    dropout=0.0,
    classifier_dropout=0.0,
).to(device)

criterion = nn.CrossEntropyLoss()

lr = 1e-4
optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad), lr=lr
)

torch.manual_seed(0)

start_time = time.time()
print("starting")
for epoch in range(epochs):
    print(f"epoch={epoch}")
    epoch_loss = 0
    epoch_correct = 0
    epoch_count = 0
    model.train()
    for batch in DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True):
        notes_batch, chords_batch = batch 
        predictions = model(notes_batch.to(device))
        labels = chords_batch.to(device)

        loss = criterion(predictions, labels)

        correct = predictions.argmax(axis=1) == labels.argmax(axis=1)
        acc = correct.sum().item() / correct.size(0)

        epoch_correct += correct.sum().item()
        epoch_count += correct.size(0)

        epoch_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()

    train_acc = epoch_correct / len(train_dataset)
    print ("train_acc = " + train_acc)
    # model.eval() # Set model to "eval mode", e.g. turns dropout off if you have dropout layers.
    # with torch.no_grad(): # Don't allocate memory for storing gradients, more efficient when not training
    #     dev_logits = model(dev_notes_tensor)
    #     dev_preds = torch.argmax(dev_logits, dim=1)
    #     dev_chords_preds = torch.argmax(dev_chords_tensor, dim=1)
    #     dev_num_correct = torch.sum(dev_preds == dev_chords_preds).item()
    #     dev_acc = dev_num_correct / len(dev_chords_tensor)
    #     if dev_acc > best_dev_acc:
    #         best_dev_acc = dev_acc
    #         best_checkpoint = model.state_dict()
    #         best_epoch = epoch
    # print(f"Epoch {epoch: < 2}: train_acc={train_acc}, dev_acc={dev_acc}")
    # if last_integer_accuracy_in_percent == int(train_acc * 100):
    #     count_epochs += 1
    # else:
    #     count_epochs = 0
    # if count_epochs >= 15:
    #     break
    # last_integer_accuracy_in_percent = int(train_acc * 100)

    print(f"epoch_loss={epoch_loss}")
    print(f"epoch accuracy: {epoch_correct / epoch_count}")
print("-------------- Done Training --------------")
print("")
print("-------------- Saving Best Model --------------")
print("")
# model.load_state_dict(best_checkpoint)
# end_time = time.time()
# print(f"Total time: {end_time - start_time:.2f} seconds")
# print(f"Best dev accuracy: {best_dev_acc} at epoch {best_epoch}")
# save(model.state_dict(), os.path.join(DATA_DIR, SAVEFILE_NAME))