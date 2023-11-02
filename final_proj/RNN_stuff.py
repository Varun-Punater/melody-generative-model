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
from torch.utils.data import DataLoader
import torch
from models.RNN_model import MusicRNNParams, MusicRNN

PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
DATA_DIR = os.path.join(PARENT_DIR, 'data')

NUM_EPOCHS = 500
LEARNING_RATE = 0.0001
BATCH_SIZE = 10
NUM_TENSORS = 100 * BATCH_SIZE

TRAIN_RATIO = 0.01

def generate_json_data(xml_file_array, file_name):

    is_training = True if file_name == f"training_r={TRAIN_RATIO}" else False

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

                    if is_training:
                        pitches_vocab.add(note_pitch)

                    measure_json["notes"].append(note_pitch)

                elif isinstance(datapoint, chord.Chord):
                    #print(F"CHORD: {datapoint.root()} {datapoint.commonName}")
                    chord_string = str(datapoint.root()) + " " + str(datapoint.commonName)
                    chord_string = re.sub(r'\d+', '', chord_string)

                    if is_training:
                        chords_vocab.add(chord_string)

                    measure_json["chords"].append(chord_string)
            
            data_json.append(measure_json)
      
    # need to save the data_json as a json file
    with open(os.path.join(DATA_DIR, file_name+'_data.json'), 'w') as outfile:
        json.dump(data_json, outfile)

    if is_training:
        vocab_json = {}
        vocab_json["pitches"] = list(pitches_vocab)
        vocab_json["chords"] = list(chords_vocab)

        print("num pitches:", len(vocab_json["pitches"]))
        print("num chords:", len(vocab_json["chords"]))

        with open(os.path.join(DATA_DIR, f'vocab_r={TRAIN_RATIO}.json'), 'w') as outfile:
            json.dump(vocab_json, outfile)

def pre_processing(limit_size=False):

    # take in the pickle data for training
    print("----------------- PROCESSING DATA -----------------")
    training_xml_data = []
    dev_xml_data = []
    test_xml_data = []

    with open(os.path.join(DATA_DIR, 'training_parsed_xml_files.pkl'), 'rb') as f:
        training_xml_data = pickle.load(f)

    with open(os.path.join(DATA_DIR, 'dev_parsed_xml_files.pkl'), 'rb') as f:
        dev_xml_data = pickle.load(f)

    with open(os.path.join(DATA_DIR, 'test_parsed_xml_files.pkl'), 'rb') as f:
        test_xml_data = pickle.load(f)

    train_size = len(training_xml_data)
    dev_size = len(dev_xml_data)
    test_size = len(test_xml_data)

    if limit_size:
        train_size = math.floor(train_size * TRAIN_RATIO)
        dev_size = math.floor(dev_size * TRAIN_RATIO)
        test_size = math.floor(test_size * TRAIN_RATIO)
    
        
    generate_json_data(training_xml_data[:train_size], f"training_r={TRAIN_RATIO}")
    generate_json_data(dev_xml_data[:dev_size], f"dev_r={TRAIN_RATIO}")
    generate_json_data(test_xml_data[:test_size], f"test_r={TRAIN_RATIO}")

    print("----------------- FINISHED -----------------")

    return

def generate_tensors(data, vocab, max_num):

    batched_notes_tensor = tensor([])
    batched_chords_tensor = tensor([])

    notes_tensor = tensor([])
    chords_tensor = tensor([])

    batched_new_notes_tensor = tensor([])

    count = 0

    for measure in tqdm(data):

        # we only proceed if the notes and chords are not empty
        if len(measure["notes"]) == 0 or len(measure["chords"]) == 0:
            continue

        # need to change how we create the tensor of notes so that we keep in mind the 

        new_notes_tensor = tensor([])

        curr_notes_array = []
        for note in measure["notes"]:

            curr_note = vocab["pitches"].index(note)

            curr_note_encoding = nn.functional.one_hot(tensor([curr_note]), num_classes=len(vocab["pitches"]))
            new_notes_tensor = torch.cat((new_notes_tensor, curr_note_encoding), 0)

            curr_notes_array.append(curr_note)

        # if the tensor is longer than 8 notes, we cut it off, otherwise we pad it
        if new_notes_tensor.shape[0] > 8:
            new_notes_tensor = new_notes_tensor[:8]
        else:
            num_note_pads = 8 - new_notes_tensor.shape[0] 
            
            pad_encoding = nn.functional.one_hot(tensor([vocab["pitches"].index("PAD")]), num_classes=len(vocab["pitches"]))

            # we need to pad new_notes_tensor num_note_pads times
            
            for i in range(num_note_pads):
                new_notes_tensor = torch.cat((new_notes_tensor, pad_encoding), 0)


        # we need to pad the tensor to make it the same length as the longest tensor which has 16 notes
        num_pitch_pads = 16 - len(curr_notes_array)
        pad_array = [ vocab["pitches"].index("PAD") for i in range(num_pitch_pads) ]

        curr_notes_array += pad_array
        
        # we now need to create a tensor of the pitch pads of the correct length
        curr_notes_tensor = tensor(curr_notes_array)
        
        curr_notes_tensor = curr_notes_tensor.unsqueeze(0)
        
        
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

        # if the chord is not in the vocab, we skip it
        if curr_chord not in vocab["chords"]:
            continue

        curr_chords_tensor.append(vocab["chords"].index(curr_chord))
        curr_chords_tensor = tensor(curr_chords_tensor)

        # use one hot encoding for the chords from the tensor
        curr_chords_tensor = nn.functional.one_hot(curr_chords_tensor, num_classes=len(vocab["chords"]))
        # curr_chords_tensor = curr_chords_tensor.unsqueeze(0) # up in the air

        notes_tensor = torch.cat((notes_tensor, curr_notes_tensor), 0)
        chords_tensor = torch.cat((chords_tensor, curr_chords_tensor), 0)

        count += 1
        if count > max_num:
            break

        if (count > 0 ) and ( count%BATCH_SIZE == 0 ):
            batched_notes_tensor = torch.cat((batched_notes_tensor, notes_tensor), 0)
            batched_chords_tensor = torch.cat((batched_chords_tensor, chords_tensor), 0)
            notes_tensor = tensor([])
            chords_tensor = tensor([])

            new_notes_tensor = new_notes_tensor.unsqueeze(0)

            batched_new_notes_tensor = torch.cat((batched_new_notes_tensor, new_notes_tensor), 0)
            new_notes_tensor = tensor([])

        # curr_chords_tensor = tensor(curr_chords_tensor)

        # num_chord_pads = 2 - len(curr_chords_tensor)
        # chord_pads = [ vocab["chords"].index("PAD") for i in range(num_chord_pads)]
        # chord_pads = tensor(chord_pads)
        # curr_chords_tensor = torch.cat((curr_chords_tensor, chord_pads), 0)

        # curr_chords_tensor = curr_chords_tensor.unsqueeze(0)
        # chords_tensor = torch.cat((chords_tensor, curr_chords_tensor), 0)

    #chords_tensor = chords_tensor.unsqueeze(0)
    batched_notes_tensor = batched_notes_tensor.type(torch.LongTensor)

    print("shape of new batched notes tensor:", batched_new_notes_tensor.shape)
    print("shape of batched notes tensor:", batched_notes_tensor.shape)
    print("shape of batched chords tensor:", batched_chords_tensor.shape)

    exit()

    # print("shape of batched notes tensor:", batched_notes_tensor.shape)
    # print("shape of batched chords tensor:", batched_chords_tensor.shape)

    return batched_notes_tensor, batched_chords_tensor

def train_rnn():

    # get the data from the json file
    data = []
    with open(os.path.join(DATA_DIR, f'training_r={TRAIN_RATIO}_data.json')) as json_file:
        data = json.load(json_file)

    # get the vocab from the json file
    vocab = {}
    with open(os.path.join(DATA_DIR, f'vocab_r={TRAIN_RATIO}.json')) as json_file:
        vocab = json.load(json_file)

    batched_notes_tensor, batched_chords_tensor = generate_tensors(data, vocab, NUM_TENSORS)

    train_dataset = torch.utils.data.TensorDataset(batched_notes_tensor, batched_chords_tensor)

    # get the sizes of vocab["pitches"] and vocab["chords"]
    vocab_dim = len(vocab["pitches"])
    chord_dim = len(vocab["chords"])

    # set of training predicted chords
    training_pred_indexes = set()
    
    # we need to create the model
    params = MusicRNNParams(vocab_dim=vocab_dim, chord_dim=chord_dim)
    model = MusicRNN(params)

    # we need to create the loss function
    loss_function = nn.CrossEntropyLoss()

    # we need to create the optimizer
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print("----------starting training----------")
    for i in tqdm(range(NUM_EPOCHS)):

        model.train()

        loss_in_epoch = 0

        for batch in DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True):
            x_batch, y_batch = batch
            optimizer.zero_grad()

            # print("x_batch shape:", x_batch.shape)
            # print("y_batch shape:", y_batch.shape)

            logits = model(x_batch)
            # need to remove the 0th dimension of the logits

            #print("shape of model output:", logits.shape)

            for pred_i in logits[0]:
                chord_index = torch.argmax(pred_i)
                chord_pred = vocab["chords"][chord_index]
                #print("chord:", chord_pred)
                training_pred_indexes.add(chord_pred)

            #y_batch = y_batch.unsqueeze(0)

            loss = loss_function(logits, y_batch)
            loss.backward()
            optimizer.step()

            loss_in_epoch += loss.item()
            #print("loss value:", loss.item())
        
        print("loss in epoch:", loss_in_epoch)

        # we need to train the model on the data
        # for i in tqdm(range(NUM_EPOCHS)):
        #     optimizer.zero_grad()
        #     output = model(notes_tensor)
        #     # print("shape of model output:", output.shape)

        #     loss = loss_function(output, chords_tensor)
        #     loss.backward()
        #     optimizer.step()
        #     #print(loss.item())

    print("num predicted chords encountered:", len(training_pred_indexes))

    # we need to save the model, if it already exists, overwrite 
    if os.path.exists(os.path.join(DATA_DIR, f'r={TRAIN_RATIO}_model.pt')):
        os.remove(os.path.join(DATA_DIR, f'r={TRAIN_RATIO}_model.pt'))
    save(model.state_dict(), os.path.join(DATA_DIR, f'r={TRAIN_RATIO}_model.pt'))

    print("----------finished training----------")

    return

def unbatched_train_model():
    data = []
    with open(os.path.join(DATA_DIR, f'training_r={TRAIN_RATIO}_data.json')) as json_file:
        data = json.load(json_file)

    # get the vocab from the json file
    vocab = {}
    with open(os.path.join(DATA_DIR, f'vocab_r={TRAIN_RATIO}.json')) as json_file:
        vocab = json.load(json_file)

    batched_notes_tensor, batched_chords_tensor = generate_tensors(data, vocab, len(data))

    # train_dataset = torch.utils.data.TensorDataset(batched_notes_tensor, batched_chords_tensor)

    # get the sizes of vocab["pitches"] and vocab["chords"]
    vocab_dim = len(vocab["pitches"]) # 
    chord_dim = len(vocab["chords"]) #

    # set of training predicted chords
    training_pred_indexes = set()
    
    # we need to create the model
    params = MusicRNNParams(vocab_dim=vocab_dim, chord_dim=chord_dim)
    model = MusicRNN(params)

    # we need to create the loss function
    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print("----------starting training----------")
    for i in tqdm(range(NUM_EPOCHS)):
        model.train()

        logits = model(batched_notes_tensor)

        print(batched_chords_tensor)

        for pred_i in logits[0]:
                chord_index = torch.argmax(pred_i)
                chord_pred = vocab["chords"][chord_index]
                #print("chord:", chord_pred)
                training_pred_indexes.add(chord_pred)

        # logits = logits.squeeze(0)

        loss = loss_function(logits, batched_chords_tensor)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if loss.item() <= 2.931:
        #     print("stopping at epoch ", i)
        #     break

        print("loss value:", loss.item())
        print("num predicted chords encountered:", len(training_pred_indexes))

    # we need to save the model, if it already exists, overwrite 
    if os.path.exists(os.path.join(DATA_DIR, f'r={TRAIN_RATIO}_model.pt')):
        os.remove(os.path.join(DATA_DIR, f'r={TRAIN_RATIO}_model.pt'))
    save(model.state_dict(), os.path.join(DATA_DIR, f'r={TRAIN_RATIO}_model.pt'))

    print("----------finished training----------")

def test_rnn():

    # load the test data from the json file
    test_data = []
    with open(os.path.join(DATA_DIR, 'test_data.json')) as json_file:
        test_data = json.load(json_file)
    
    # load the vocab from the json file
    vocab = {}
    with open(os.path.join(DATA_DIR, 'vocab.json')) as json_file:
        vocab = json.load(json_file)

    vocab_dim = len(vocab["pitches"])
    chord_dim = len(vocab["chords"])
    
    # we need to load the model
    params = MusicRNNParams(vocab_dim=vocab_dim, chord_dim=chord_dim)
    model = MusicRNN(params)
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'model.pt')))
    model.eval()

    # we need to generate tensors for the data
    batched_test_notes_tensor, batched_test_chords_tensor = generate_tensors(test_data, vocab, NUM_TENSORS)

    test_dataset = torch.utils.data.TensorDataset(batched_test_notes_tensor, batched_test_chords_tensor)
    print("num test tensors:", len(test_dataset))

    predicted_chords_encountered = set()

    accurate_predictions = 0

    max = 10 # len(test_dataset)
    count = 0
    for note_tensor, chord_tensor in tqdm(test_dataset):

        note_tensor = note_tensor.unsqueeze(0)

        model_output = model(note_tensor)
        model_output = model_output.squeeze(0)
        model_output = model_output.squeeze(0)

        #print("model output:", model_output)

        if torch.argmax(model_output) == torch.argmax(chord_tensor):
            accurate_predictions += 1

        chord_index = torch.argmax(model_output)
        chord = vocab["chords"][chord_index]

        predicted_chords_encountered.add(chord)

        real_notes = [ vocab["pitches"][i] for i in note_tensor[0] ]

        # print("current notes:", real_notes)
        # print("predicted chord:", chord)
        # print("")

        count += 1
        if count > max:
            break

    print("num predicted chords encountered:", len(predicted_chords_encountered))
    print("num accurate predictions:", accurate_predictions)

    return

if __name__ == "__main__":

    print("---------------------------")
    print("Hyperparameters:")
    print("NUM_EPOCHS:", NUM_EPOCHS)
    print("LEARNING_RATE:", LEARNING_RATE)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("NUM_TENSORS:", NUM_TENSORS)
    print("TRAIN_RATIO:", TRAIN_RATIO)
    print("---------------------------")
    print("")

    # pre_processing(limit_size=True)
    # train_rnn()
    unbatched_train_model()
    # test_rnn()