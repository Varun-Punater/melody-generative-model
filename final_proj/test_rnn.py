import os
import argparse
from pathlib import Path
import pickle
from music21 import *
import math
from typing import List, Dict, Set
from tqdm import tqdm
import regex as re
import json
from torch import nn
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import tensor
from torch import save
from torch.utils.data import DataLoader
import torch
# from models.RNN_model import MusicRNNParams, MusicRNN
from models.LSTM_model import MusicRNNParams, MusicRNN
import time
from threading import Thread, Lock

# signaling_mutex = Lock()
# signal = 0

PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
OG_DATA_DIR = os.path.join(PARENT_DIR, 'data')
DATA_DIR = os.path.join(PARENT_DIR, 'fake_data')


NUM_EPOCHS = 1000
BATCH_SIZE = 100
LEARNING_RATE = 0.1 #0.05 earlier for the model
M0MTM = 0.1 # set to 0 for the default
DMPNG = 0 # set to 0 for the default
GAMMA = 0.999


def get_measures_from_score(score: stream.Score):
    part = score.parts[0]
    measures: List[stream.Measure] = []
    for x in part:
        if isinstance(x, stream.Measure):
            measures.append(x)
    return measures


def extract_features_from_measure(measure: stream.Measure):
    notes: List[str] = []
    chords: List[str] = []
    unique_notes = set()
    unique_chords = set()
    for datapoint in measure:
        if isinstance(datapoint, note.Note):
            number_to_append = int(datapoint.duration.quarterLength * 4)

            # remove numbers from chord / note names
            note_name = re.sub(r'[0-9]', '', datapoint.pitch.name)
            unique_notes.add(note_name)
            for i in range(number_to_append):
                notes.append(note_name)
        elif isinstance(datapoint, chord.Chord):
            chord_name = re.sub(r'[0-9]', '', f"{datapoint.root()} {datapoint.commonName}")
            unique_chords.add(chord_name)
            chords.append(chord_name)
        elif isinstance(datapoint, note.Rest):
            number_to_append = int(datapoint.duration.quarterLength * 4)            
            for i in range(number_to_append):
                notes.append("REST")
    return notes, chords, unique_notes, unique_chords


def generate_json_data(scores: List[stream.Score], file_name):
    print("-------------- Generating JSON Data --------------")

    pitches_vocab = set()
    chords_vocab = set()

    pitches_vocab.add("REST")
    chords_vocab.add("REST")

    json_data = []

    for i in tqdm(range(len(scores))):
        s = scores[i]
        measures = get_measures_from_score(s)
        
        for measure in measures[1:]:
            notes, chords, unique_notes, unique_chords = extract_features_from_measure(measure)
            pitches_vocab.update(unique_notes)
            chords_vocab.update(unique_chords)
            json_data.append({
                "notes": notes,
                "chords": chords
            })
        

    with open(os.path.join(DATA_DIR, file_name+'.json'), 'w') as outfile:
        json.dump(json_data, outfile)

    if file_name == "train" or file_name == "sample":
        with open(os.path.join(DATA_DIR, 'pitches_vocab.json'), 'w') as outfile:
            json.dump(list(pitches_vocab), outfile)
        with open(os.path.join(DATA_DIR, 'chords_vocab.json'), 'w') as outfile:
            json.dump(list(chords_vocab), outfile)
    
    print("-------------- Done Generating JSON Data --------------")


def pre_process():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print("----------------- Processing Data -----------------")
    parsed_xml_files = []

    print("------------- Loading Training Data -------------")
    with open(os.path.join(OG_DATA_DIR, 'training_parsed_xml_files.pkl'), 'rb') as f:
        parsed_xml_files = pickle.load(f)
    print("------------- Done Loading Training Data -------------")

    generate_json_data(parsed_xml_files, "train")

    print("------------- Loading Dev Data -------------")
    with open(os.path.join(OG_DATA_DIR, 'dev_parsed_xml_files.pkl'), 'rb') as f:
        parsed_xml_files = pickle.load(f)
    print("------------- Done Loading Dev Data -------------")

    generate_json_data(parsed_xml_files, "dev")

    print("------------- Loading Test Data -------------")
    with open(os.path.join(OG_DATA_DIR, 'test_parsed_xml_files.pkl'), 'rb') as f:
        parsed_xml_files = pickle.load(f)
    print("------------- Done Loading Test Data -------------")

    generate_json_data(parsed_xml_files, "test")

    print("----------------- FINISHED -----------------")

    return
            

def sample_pre_process():
    xml_file_paths = Path("../chord-melody-dataset/").glob('**/*.xml')
    xml_file_paths = [str(path) for path in xml_file_paths]

    parsed_xml_files = []
    for xml_file_path in xml_file_paths[:50]:
        s = converter.parse(xml_file_path)
        parsed_xml_files.append(s)
    
    generate_json_data(parsed_xml_files, "sample")
        

def remove_numbers_from_all_measures(data):
    for measure in data:
        notes = measure["notes"]
        chords = measure["chords"]
        for i in range(len(notes)):
            notes[i] = re.sub(r'[0-9]', '', notes[i])
        for i in range(len(chords)):
            chords[i] = re.sub(r'[0-9]', '', chords[i])
    return data

def clean_json_vocab_data():
    print("----------------- Cleaning JSON Data -----------------")
    data = []
    with open(os.path.join(DATA_DIR, 'chords_vocab.json')) as json_file:
        data = json.load(json_file)
    
    for i in range(len(data)):
        data[i] = re.sub(r'[0-9]', '', data[i])
    
    data = list(set(data))

    # save data
    with open(os.path.join(DATA_DIR, 'chords_vocab.json'), 'w') as outfile:
        json.dump(data, outfile)

    data = []
    with open(os.path.join(DATA_DIR, 'pitches_vocab.json')) as json_file:
        data = json.load(json_file)
    
    for i in range(len(data)):
        data[i] = re.sub(r'[0-9]', '', data[i])
    
    data = list(set(data))

    # save data
    with open(os.path.join(DATA_DIR, 'pitches_vocab.json'), 'w') as outfile:
        json.dump(data, outfile)
    
    print("----------------- Done Cleaning JSON Data -----------------")


def clean_json_train_data():
    print("----------------- Cleaning JSON Data -----------------")
    data = []
    with open(os.path.join(DATA_DIR, 'train.json')) as json_file:
        data = json.load(json_file)
    
    data = remove_numbers_from_all_measures(data)
    # save data
    with open(os.path.join(DATA_DIR, 'train.json'), 'w') as outfile:
        json.dump(data, outfile)

    data = []
    with open(os.path.join(DATA_DIR, 'dev.json')) as json_file:
        data = json.load(json_file)
    
    data = remove_numbers_from_all_measures(data)
    # save data
    with open(os.path.join(DATA_DIR, 'dev.json'), 'w') as outfile:
        json.dump(data, outfile)
    
    data = []
    with open(os.path.join(DATA_DIR, 'test.json')) as json_file:
        data = json.load(json_file)
    
    data = remove_numbers_from_all_measures(data)
    # save data
    with open(os.path.join(DATA_DIR, 'test.json'), 'w') as outfile:
        json.dump(data, outfile)
    
    print("----------------- Done Cleaning JSON Data -----------------")
    

def create_tensors(partition_type):
    print("----------------- Creating Tensors -----------------")

    data = []
    with open(os.path.join(DATA_DIR, f'{partition_type}.json')) as json_file:
        data = json.load(json_file)

    # get the vocab from the json file
    chords_vocab = []
    with open(os.path.join(DATA_DIR, 'chords_vocab.json')) as json_file:
        chords_vocab = json.load(json_file)

    notes_vocab = []
    with open(os.path.join(DATA_DIR, 'pitches_vocab.json')) as json_file:
        notes_vocab = json.load(json_file)

    notes_tensor = tensor([])
    chords_tensor = tensor([])

    for measure in tqdm(data):

        # we only proceed if the notes and chords are not empty
        if len(measure["notes"]) == 0 or len(measure["chords"]) == 0:
            continue

        notes = measure["notes"]
        chords = measure["chords"]

        # we need to convert the chords to indices
        curr_chords_tensor = []
        curr_chord = chords[0]
        try:
            curr_chords_tensor.append(chords_vocab.index(curr_chord))
        except Exception as e:
            continue

        curr_chords_tensor = tensor(curr_chords_tensor)

        # one hot encoding for the chords
        curr_chords_tensor = nn.functional.one_hot(curr_chords_tensor, num_classes=len(chords_vocab))

        chords_tensor = torch.cat((chords_tensor, curr_chords_tensor), 0)
        
        # we need to convert the notes and chords to indices
        curr_notes_tensor = []
        for note in notes:
            curr_notes_tensor.append(notes_vocab.index(note))
        curr_notes_tensor = tensor(curr_notes_tensor)

        curr_notes_tensor = curr_notes_tensor.unsqueeze(0)
        
        # add the current measure notes to the overall notes
        notes_tensor = torch.cat((notes_tensor, curr_notes_tensor), 0)

    # chords_tensor.unsqueeze(0)

    notes_tensor = notes_tensor.type(torch.LongTensor)

    # we need to save the tensors
    save(notes_tensor, os.path.join(DATA_DIR, f'{partition_type}_notes_tensor.pt'))
    save(chords_tensor, os.path.join(DATA_DIR, f'{partition_type}_chords_tensor.pt'))

def train(num_measures: int):
    start_time = time.time()
    # load the training tensors

    print("----------------- Loading Training Tensors -----------------")
    print("")

    notes_tensor = torch.load(os.path.join(DATA_DIR, 'train_notes_tensor.pt'))
    chords_tensor = torch.load(os.path.join(DATA_DIR, 'train_chords_tensor.pt'))

    chords_vocab = []
    with open(os.path.join(DATA_DIR, 'chords_vocab.json')) as json_file:
        chords_vocab = json.load(json_file)

    notes_vocab = []
    with open(os.path.join(DATA_DIR, 'pitches_vocab.json')) as json_file:
        notes_vocab = json.load(json_file)

    print("----------------- Done Loading Training Tensors -----------------")
    print("")

    print("----------------- Loading Dev Tensors -----------------")
    print("")
    dev_notes_tensor = torch.load(os.path.join(DATA_DIR, 'dev_notes_tensor.pt'))
    dev_chords_tensor = torch.load(os.path.join(DATA_DIR, 'dev_chords_tensor.pt'))

    print("----------------- Done Loading Dev Tensors -----------------")

    # global signal


    # create the model 
    params = MusicRNNParams(
        vocab_dim = len(notes_vocab),
        chord_dim = len(chords_vocab)
    )
    model = MusicRNN(params)

    print("----------------- Hyperparameters -----------------")
    print("")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("")
    print(f"embedding_dim: {params.embedding_dim}")
    print(f"hidden_dim: {params.hidden_dim}")

    # loss function
    loss_function = nn.CrossEntropyLoss()

    # optimizer option 1: SGD with or without scheduling, with or without momentum
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=M0MTM, dampening=DMPNG)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

    # optimizer option 2: Adam with weight decay
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    
    train_dataset = [(notes_tensor[i], chords_tensor[i]) for i in range(len(chords_tensor))]

    if num_measures != -1:
        train_dataset = train_dataset[:num_measures]

    best_dev_acc = -1
    best_checkpoint = None
    best_epoch = -1

    # for early stopping
    last_integer_accuracy_in_percent = -1
    count_epochs = 0

    print("-------------- Training --------------")
    for i in range(NUM_EPOCHS):
        # signaling_mutex.acquire()
        # if signal == 1:
        #     signaling_mutex.release()
        #     break
        # signaling_mutex.release()

        train_num_correct = 0

        # Training loop
        model.train() # Set model to "training mode", e.g. turns dropout on if you have dropout layers
        for batch in DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True):
            notes_batch, chords_batch = batch # unpack batch, which is a tuple (x_batch, y_batch)
                                                # x_batch is tensor of size (B, D)
                                                # y_batch is tensor of size (B, X)
            optimizer.zero_grad()   # Reset the gradients to zero
                                    # Recall how backpropagation works---gradients are initialized to zero and then accumulated
                                    # So we need to reset to zero before running on a new batch!

            logits = model(notes_batch) # tensor of size (B, C), each row is the logits (pre-softmax scores) for the C classes
            loss = loss_function(logits, chords_batch) # Compute the loss of the model output compared to true labels
            loss.backward() # Run backpropagation to compute gradients
            optimizer.step() # Take a SGD step
                             # Note that when we created the optimizer, we passed in model.parameters()
                             # This is a list of all parameters of all layers of the model
                             # optimizer.step() iterates over this list and does an SGD update to each parameter
            # Compute running count of number of training examples correct
            preds = torch.argmax(logits, dim=1)
            chords_tensor_preds = torch.argmax(chords_batch, dim=1)
            train_num_correct += torch.sum(preds == chords_tensor_preds).item()

        scheduler.step() # Update the learning rate
        
        # Evaluate train and dev accuracy at the end of each epoch
        train_acc = train_num_correct / len(train_dataset)
        model.eval() # Set model to "eval mode", e.g. turns dropout off if you have dropout layers.
        with torch.no_grad(): # Don't allocate memory for storing gradients, more efficient when not training
            dev_logits = model(dev_notes_tensor)
            dev_preds = torch.argmax(dev_logits, dim=1)
            dev_chords_preds = torch.argmax(dev_chords_tensor, dim=1)
            dev_num_correct = torch.sum(dev_preds == dev_chords_preds).item()
            dev_acc = dev_num_correct / len(dev_chords_tensor)
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_checkpoint = model.state_dict()
                best_epoch = i
        print(f"Epoch {i: < 2}: train_acc={train_acc}, dev_acc={dev_acc}")
        if last_integer_accuracy_in_percent == int(train_acc * 100):
            count_epochs += 1
        else:
            count_epochs = 0
        if count_epochs >= 15:
            break
        last_integer_accuracy_in_percent = int(train_acc * 100)
        
    print("-------------- Done Training --------------")
    print("")
    print("-------------- Saving Best Model --------------")
    print("")
    model.load_state_dict(best_checkpoint)
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Best dev accuracy: {best_dev_acc} at epoch {best_epoch}")
    save(model.state_dict(), os.path.join(DATA_DIR, 'best_model_10.pt'))

    # signaling_mutex.acquire()
    # signal = 1
    # signaling_mutex.release()
    

def evaluate():
    print("----------------- Loading Testing Tensors -----------------")
    print("")

    notes_tensor = torch.load(os.path.join(DATA_DIR, 'train_notes_tensor.pt'))
    chords_tensor = torch.load(os.path.join(DATA_DIR, 'train_chords_tensor.pt'))

    chords_vocab = []
    with open(os.path.join(DATA_DIR, 'chords_vocab.json')) as json_file:
        chords_vocab = json.load(json_file)

    notes_vocab = []
    with open(os.path.join(DATA_DIR, 'pitches_vocab.json')) as json_file:
        notes_vocab = json.load(json_file)

    print("----------------- Done Loading Testing Tensors -----------------")
    print("")

    # create model
    params = MusicRNNParams(
        vocab_dim = len(notes_vocab),
        chord_dim = len(chords_vocab)
    )
    model = MusicRNN(params)

    # load model
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'best_model_10.pt')))
    model.eval()

    logits = model(notes_tensor)
    preds = torch.argmax(logits, dim=1)
    chords_tensor_preds = torch.argmax(chords_tensor, dim=1)
    train_num_correct = torch.sum(preds == chords_tensor_preds).item()

    accuracy = train_num_correct / len(chords_tensor)

    print(f"accuracy: {accuracy}")


# def signal_handler():
#     global signal
#     while (1):
#         signaling_mutex.acquire()
#         if signal == 1:
#             signaling_mutex.release()
#             break
#         signaling_mutex.release()
#         val = input()
#         if val == 'q' or val == 'quit':
#             signaling_mutex.acquire()
#             signal = 1
#             signaling_mutex.release()
#             break
        

if __name__ == "__main__":

    # we need to add a way to use comand line options
    parser = argparse.ArgumentParser(description='Hyperparameters and mode for the model.')
    parser.add_argument('--mode', type=str, required=True, choices=['pre', 'create', 'train', 'eval'], help='Mode to run')
    parser.add_argument('--E', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--B', type=int, default=100, help='Batch size')
    parser.add_argument('--L', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--mu', type=float, default=0.1, help='Momentum')
    parser.add_argument('--gamma', type=float, default=0.999, help='Gamma')

    args = parser.parse_args()

    NUM_EPOCHS = args.E
    BATCH_SIZE = args.B
    LEARNING_RATE = args.L
    M0MTM = args.mu
    GAMMA = args.gamma

    if args.mode == 'pre':
        pre_process()
    elif args.mode == 'create':
        create_tensors('test')
        create_tensors('dev')
    elif args.mode == 'train':
        # print the hyperparameters
        print("----------------- Hyperparameters -----------------")
        print("")
        print(f"Number of epochs: {NUM_EPOCHS}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Momentum: {M0MTM}")
        print(f"Gamma: {GAMMA}")
        print("")
        print("----------------- Starting Training -----------------")
        train(-1)
    elif args.mode == 'eval':
        evaluate()
    
    # sample_pre_process() # use this if you want to train model on a smaller sample
    # pre_process()
    # clean_json_train_data() # i forgot to remove numbers originally... oops
    # clean_json_vocab_data() # i forgot to remove numbers originally... oops
    # create_tensors('test')
    # total of 112329 measures
    # evaluate()
