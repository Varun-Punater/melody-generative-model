import os
import pickle
from music21 import *
import math
from typing import List, Dict, Set
from tqdm import tqdm
import regex as re
import json
from torch import nn
from torch import optim
from torch import tensor
from torch import save
import torch
from models.RNN_model import MusicRNNParams, MusicRNN

PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
DATA_DIR = os.path.join(PARENT_DIR, 'data')

NUM_ITERS = 10
LEARNING_RATE = 0.001

def generate_json_data(xml_file_array, file_name):

    # we need a set of all pitches and all chords
    pitches_vocab = set()
    chords_vocab = set()

    # we need to add a padding token to the pitches_vocab and chords_vocab
    pitches_vocab.add("PAD")
    chords_vocab.add("PAD")

    data_json = []

    print(len(xml_file_array))
    for index, s in tqdm(enumerate(xml_file_array)):
        part = s.parts[0]
        measures: List[stream.Measure] = []
        for x in part:
            if isinstance(x, stream.Measure):
                measures.append(x)

        #print(f"----------------- ANALYZING SONG {index} -----------------")
        for index, measure in enumerate(measures):
            #print(F"---------------- MEASURE {index} ----------------")

            measure_json = {}
            measure_json["notes"] = []
            measure_json["chords"] = []
            

            notes_so_far = []
            for datapoint in measure:
                if isinstance(datapoint, note.Note):
                    notes_so_far.append(datapoint)
                    #print(F"PITCH: {datapoint.pitch}")
                    # notes_so_far.append(datapoint.pitch)

                    # need to get the pitch as a string without the octave
                    note_pitch = str(datapoint.pitch)
                    note_pitch = re.sub(r'\d+', '', note_pitch)
                    
                    #print(F"PITCH: {note_pitch}")

                    if file_name == "training":
                        pitches_vocab.add(note_pitch)

                    measure_json["notes"].append(note_pitch)

                elif isinstance(datapoint, chord.Chord):
                    #print(F"CHORD: {datapoint.root()} {datapoint.commonName}")
                    chord_string = str(datapoint.root()) + " " + str(datapoint.commonName)
                    chord_string = re.sub(r'\d+', '', chord_string)

                    if file_name == "training":
                        chords_vocab.add(chord_string)

                    measure_json["chords"].append(chord_string)
            
            data_json.append(measure_json)

            
    # need to save the data_json as a json file
    with open(os.path.join(DATA_DIR, file_name+'_data.json'), 'w') as outfile:
        json.dump(data_json, outfile)

    if file_name == "training":
        vocab_json = {}
        vocab_json["pitches"] = list(pitches_vocab)
        vocab_json["chords"] = list(chords_vocab)

        with open(os.path.join(DATA_DIR, 'vocab.json'), 'w') as outfile:
            json.dump(vocab_json, outfile)

def pre_processing():
    
    # take in the pickle data for training
    print("----------------- PROCESSING DATA -----------------")
    parsed_xml_files = []
    with open(os.path.join(DATA_DIR, 'training_parsed_xml_files.pkl'), 'rb') as f:
        parsed_xml_files = pickle.load(f)
    
    generate_json_data(parsed_xml_files, "training")

    with open(os.path.join(DATA_DIR, 'dev_parsed_xml_files.pkl'), 'rb') as f:
        parsed_xml_files = pickle.load(f)

    generate_json_data(parsed_xml_files, "dev")

    with open(os.path.join(DATA_DIR, 'test_parsed_xml_files.pkl'), 'rb') as f:
        parsed_xml_files = pickle.load(f)

    generate_json_data(parsed_xml_files, "test")

    print("----------------- FINISHED -----------------")

    return

def train_rnn():

    # we need to generate tensors for the data

    # we need to generate tensors for the labels

    # get the data from the json file
    data = []
    with open(os.path.join(DATA_DIR, 'training_data.json')) as json_file:
        data = json.load(json_file)

    # get the vocab from the json file
    vocab = {}
    with open(os.path.join(DATA_DIR, 'vocab.json')) as json_file:
        vocab = json.load(json_file)
    
    notes_tensor = tensor([])
    chords_tensor = tensor([])
    
    for measure in tqdm(data):

        # we only proceed if the notes and chords are not empty
        if len(measure["notes"]) == 0 or len(measure["chords"]) == 0:
            continue

        curr_notes_tensor = []
        for note in measure["notes"]:
            curr_notes_tensor.append(vocab["pitches"].index(note))
        curr_notes_tensor = tensor(curr_notes_tensor)

        # we need to pad the tensor to make it the same length as the longest tensor which has 16 notes
        num__pitch_pads = 16 - len(curr_notes_tensor)
        pitch_pads = [ vocab["pitches"].index("PAD") for i in range(num__pitch_pads)]
        pitch_pads = tensor(pitch_pads)
        curr_notes_tensor = torch.cat((curr_notes_tensor, pitch_pads), 0)

        
        curr_notes_tensor = curr_notes_tensor.unsqueeze(0)
        notes_tensor = torch.cat((notes_tensor, curr_notes_tensor), 0)
        
        # adding another notes tensor for the second chord
        # notes_tensor = torch.cat((notes_tensor, curr_notes_tensor), 0)

        # we need to create a tensor of the chords
        # curr_chords_tensor = []
        # for chord in measure["chords"]:
        #     curr_chords_tensor = []
        #     curr_chords_tensor.append(vocab["chords"].index(chord))
        #     curr_chords_tensor = tensor(curr_chords_tensor)

        #     # use one hot encoding for the chords from the tensor
        #     curr_chords_tensor = nn.functional.one_hot(curr_chords_tensor, num_classes=len(vocab["chords"]))

        #     curr_chords_tensor = curr_chords_tensor.unsqueeze(0)

        #     chords_tensor = torch.cat((chords_tensor, curr_chords_tensor), 0)

        curr_chord = measure["chords"][0]

        # generate the one hot encoding tensor for the chord
        curr_chords_tensor = []
        curr_chords_tensor.append(vocab["chords"].index(curr_chord))
        curr_chords_tensor = tensor(curr_chords_tensor)

        # use one hot encoding for the chords from the tensor
        curr_chords_tensor = nn.functional.one_hot(curr_chords_tensor, num_classes=len(vocab["chords"]))

        chords_tensor = torch.cat((chords_tensor, curr_chords_tensor), 0)

        # curr_chords_tensor = tensor(curr_chords_tensor)

        # num_chord_pads = 2 - len(curr_chords_tensor)
        # chord_pads = [ vocab["chords"].index("PAD") for i in range(num_chord_pads)]
        # chord_pads = tensor(chord_pads)
        # curr_chords_tensor = torch.cat((curr_chords_tensor, chord_pads), 0)

        # curr_chords_tensor = curr_chords_tensor.unsqueeze(0)
        # chords_tensor = torch.cat((chords_tensor, curr_chords_tensor), 0)

    chords_tensor = chords_tensor.unsqueeze(0)

    print(notes_tensor.shape)
    print(chords_tensor.shape)

    # make sure that the note tensor is an int/Long tensor
    notes_tensor = notes_tensor.type(torch.LongTensor)

    # we need to create the model
    params = MusicRNNParams()
    model = MusicRNN(params)

    # we need to create the loss function
    loss_function = nn.CrossEntropyLoss()

    # we need to create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("----------starting training----------")
    # we need to train the model on the data
    for i in tqdm(range(NUM_ITERS)):
        optimizer.zero_grad()
        output = model(notes_tensor)
        # print("shape of model output:", output.shape)

        loss = loss_function(output, chords_tensor)
        loss.backward()
        optimizer.step()
        #print(loss.item())

    # we need to save the model
    save(model.state_dict(), os.path.join(DATA_DIR, 'model.pt'))

    return

def test_rnn():

    # we need to test the model on the data

    # we need to load the saved model from the data dir called model.pt
    params = MusicRNNParams()
    model = MusicRNN(params)
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'model.pt')))
    model.eval()

    # load the test data from the json file
    test_data = []
    with open(os.path.join(DATA_DIR, 'test_data.json')) as json_file:
        test_data = json.load(json_file)
    
    # load the vocab from the json file
    vocab = {}
    with open(os.path.join(DATA_DIR, 'vocab.json')) as json_file:
        vocab = json.load(json_file)
    
    # we need to generate tensors for the data
    test_notes_tensor = tensor([])
    test_chords_tensor = tensor([])

    for measure in tqdm(test_data):
        
        if len(measure["notes"]) == 0 or len(measure["chords"]) == 0:
            continue

        curr_notes_tensor = []
        for note in measure["notes"]:
            curr_notes_tensor.append(vocab["pitches"].index(note))
        curr_notes_tensor = tensor(curr_notes_tensor)

        # we need to pad the tensor to make it the same length as the longest tensor which has 16 notes
        num__pitch_pads = 16 - len(curr_notes_tensor)
        pitch_pads = [ vocab["pitches"].index("PAD") for i in range(num__pitch_pads)]
        pitch_pads = tensor(pitch_pads)
        curr_notes_tensor = torch.cat((curr_notes_tensor, pitch_pads), 0)

        
        curr_notes_tensor = curr_notes_tensor.unsqueeze(0)
        test_notes_tensor = torch.cat((test_notes_tensor, curr_notes_tensor), 0)

        model_output = model(curr_notes_tensor)

        chord_index = torch.argmax(model_output)
        chord = vocab["chords"][chord_index]

        print("current notes:", measure["notes"])
        print("actual chord:", measure["chords"][0])
        print("predicted chord:", chord)

        exit()

    # we need to load the data from the test file

    return

if __name__ == "__main__":

    print("Hyperparameters:")
    print("NUM_ITERS:", NUM_ITERS)
    print("")

    #pre_processing()
    #train_rnn()
    test_rnn()