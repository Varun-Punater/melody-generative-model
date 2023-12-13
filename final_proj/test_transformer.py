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
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
DATA_DIR = os.path.join(PARENT_DIR, 'fake_data')
SAVEFILE_NAME = "best_transformer_model.pt"
BATCH_SIZE = 30


def train():
    notes_tensor = torch.load(os.path.join(DATA_DIR, 'train_notes_tensor.pt')).to(device)
    chords_tensor = torch.load(os.path.join(DATA_DIR, 'train_chords_tensor.pt')).to(device)

    dev_notes_tensor = torch.load(os.path.join(DATA_DIR, 'dev_notes_tensor.pt')).to(device)
    dev_chords_tensor = torch.load(os.path.join(DATA_DIR, 'dev_chords_tensor.pt')).to(device)

    train_dataset = [(notes_tensor[i], chords_tensor[i]) for i in range(len(chords_tensor))]
    dev_dataset = [(dev_notes_tensor[i], dev_chords_tensor[i]) for i in range(len(dev_chords_tensor))]

    epochs = 20
    best_dev_acc = -1
    best_checkpoint = None
    best_epoch = -1
    last_integer_accuracy_in_percent = -1
    count_epochs = 0


    model = Net(
        vocab_size=23,
        d_model=256,
        nhead=8,  # the number of heads in the multiheadattention models
        dim_feedforward=64,  # the dimension of the feedforward network model in nn.TransformerEncoder
        num_layers=4,
        dropout=0.0,
        classifier_dropout=0.0,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    lr = 1e-4
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )


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
        model.eval() # Set model to "eval mode", e.g. turns dropout off if you have dropout layers.
        total_dev_correct = 0
        total_dev_samples = 0
        with torch.no_grad(): 
            for dev_batch in DataLoader(dev_dataset, batch_size=BATCH_SIZE):
                dev_notes, dev_chords = dev_batch
                dev_logits = model(dev_notes.to(device))
                dev_preds = torch.argmax(dev_logits, dim=1)
                dev_correct = dev_preds == dev_chords.argmax(dim=1)
                total_dev_correct += dev_correct.sum().item()
                total_dev_samples += dev_chords.size(0)

            dev_acc = total_dev_correct / total_dev_samples
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_checkpoint = model.state_dict()
                best_epoch = epoch
        print(f"Epoch {epoch}: train_acc={train_acc}, dev_acc={dev_acc}")
        if last_integer_accuracy_in_percent == int(train_acc * 100):
            count_epochs += 1
        else:
            count_epochs = 0
        if count_epochs >= 15:
            break
        last_integer_accuracy_in_percent = int(train_acc * 100)

        # print(f"epoch_loss={epoch_loss}")
        # print(f"epoch accuracy: {epoch_correct / epoch_count}")
    print("-------------- Done Training --------------")
    print("")
    print("-------------- Saving Best Model --------------")
    print("")
    model.load_state_dict(best_checkpoint)
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Best dev accuracy: {best_dev_acc} at epoch {best_epoch}")
    save(model.state_dict(), os.path.join(DATA_DIR, SAVEFILE_NAME))


def evaluate():
    print("----------------- Loading Testing Tensors -----------------")
    print("")

    test_notes_tensor = torch.load(os.path.join(DATA_DIR, 'test_notes_tensor.pt'))
    test_chords_tensor = torch.load(os.path.join(DATA_DIR, 'test_chords_tensor.pt'))

    test_dataset = [(test_notes_tensor[i], test_chords_tensor[i]) for i in range(len(test_chords_tensor))]
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Net(vocab_size=23, d_model=256, nhead=8, dim_feedforward=64, num_layers=4, dropout=0.0, classifier_dropout=0.0).to(device)

    # load model
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, SAVEFILE_NAME)))
    model.eval()

    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for notes_batch, chords_batch in test_loader:
            notes_batch, chords_batch = notes_batch.to(device), chords_batch.to(device)
            logits = model(notes_batch)
            preds = torch.argmax(logits, dim=1)
            chords_batch_preds = torch.argmax(chords_batch, dim=1)
            total_correct += torch.sum(preds == chords_batch_preds).item()
            total_samples += chords_batch.size(0)

    accuracy = total_correct / total_samples
    print(f"Test accuracy: {accuracy}")


if __name__ == "__main__":
    train()
    evaluate()

    # Data
    # epochs = list(range(20))
    # train_acc = [
    #     0.17736292497930187, 0.19523898548015206, 0.19773166323923475, 0.19934300136207034, 
    #     0.20017092647490853, 0.20059824266217985, 0.20087421769979258, 0.2007762910735429, 
    #     0.20116799757854162, 0.2011501927374053, 0.20119470484024607, 0.20126592420479128, 
    #     0.2019336057474027, 0.2015953137658129, 0.2013015338870639, 0.202182873523311, 
    #     0.2010344612700193, 0.20206714205592502, 0.2013549484104728, 0.20233421467296958
    # ]
    # dev_acc = [
    #     0.16342388263815502, 0.17001305401877292, 0.16609684838689626, 0.16765089824081555, 
    #     0.16547522844532853, 0.16634549636352333, 0.16491577049791759, 0.16497793249207435, 
    #     0.16715360228756138, 0.1675887362466588, 0.16448063653882017, 0.16771306023497234, 
    #     0.16889413812395102, 0.16895630011810778, 0.16516441847454466, 0.16491577049791759, 
    #     0.1652887424628582, 0.16653198234599367, 0.16678063032262075, 0.16920494809473488
    # ]

    # # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
    # plt.plot(epochs, dev_acc, label='Dev Accuracy', marker='x')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Training vs. Development Accuracy over Epochs')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
