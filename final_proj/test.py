import json
import os
import pickle
from music21 import *
import math
from typing import List, Dict
import pprint
from pathlib import Path
from multiprocessing import Process
from threading import Thread, Lock
import numpy as np
from tqdm import tqdm
from models.naive_bayes import NaiveBayes
from matplotlib import pyplot as plt


mutex = Lock()

NUM_SAMPLES = 0

# to create the data dir
PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
DATA_DIR = os.path.join(PARENT_DIR, 'data')

def parse_xml(xml_file_paths: List[str], parsed_xml_files: List[stream.Score]):
    for index in tqdm((range(len(xml_file_paths)))):
        s = converter.parse(xml_file_paths[index])
        mutex.acquire()
        parsed_xml_files.append(s)
        mutex.release()

def threaded_parse_xml(sample, partition_type):

    PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    DATA_DIR = os.path.join(PARENT_DIR, 'data')

    threads: List[Thread] = []

    NUM_THREADS = 10

    num_files_per_process = math.ceil(len(sample) / NUM_THREADS)
    parsed_xml_files = []

    # for this object, we want to store

    for i in range(0, NUM_THREADS):
        p = Thread(target=parse_xml, args=(
            sample[num_files_per_process*i:num_files_per_process*(i+1)], parsed_xml_files))
        p.start()
        threads.append(p)

    print("")
    print("started parsing for " + partition_type)

    for p in threads:
        p.join()

    print("")
    print("done parsing with " + partition_type)

    # we want to store the parsed_xml_files list in the data dir
    TARGET_DIR = os.path.join(DATA_DIR, partition_type + '_parsed_xml_files.pkl')
    with open(TARGET_DIR, 'wb') as f:
        pickle.dump(parsed_xml_files, f)

    print(partition_type + " saved to data")

def data_parsing():
    xml_file_paths = Path("../chord-melody-dataset/").glob('**/*.xml')
    xml_file_paths = [str(path) for path in xml_file_paths]

    NUM_SAMPLES = len(xml_file_paths)
    NUM_TRAIN = math.floor(NUM_SAMPLES * 0.7)
    NUM_DEV = math.floor(NUM_SAMPLES * 0.1)

    ## only for training data
    train_samples = xml_file_paths[0:NUM_TRAIN]
    dev_samples = xml_file_paths[NUM_TRAIN:NUM_TRAIN+NUM_DEV]
    test_samples = xml_file_paths[NUM_TRAIN+NUM_DEV:NUM_SAMPLES]

    threaded_parse_xml(train_samples, 'training')
    threaded_parse_xml(dev_samples, 'dev')
    threaded_parse_xml(test_samples, 'test')

    print("done with all")

def data_processing():

    notes = ['C', 'C#', 'D-', 'D', 'D#', 'E-', 'E', 'F',
             'F#', 'G-', 'G', 'G#', 'A-', 'A', 'A#', 'B-', 'B']

    parsed_xml_files = []
    print("loading pickle files for training")
    with open(os.path.join(DATA_DIR, 'training_parsed_xml_files.pkl'), 'rb') as f:
        parsed_xml_files = pickle.load(f)

    print(len(parsed_xml_files))


def plot_sweep(lambda_val, train_accuracy, dev_accuracy):
    plt.clf()
    plt.xscale('log')
    plt.plot(lambda_val, train_accuracy, color='r', marker='*', linestyle='-', label='train')
    plt.plot(lambda_val, dev_accuracy, color='b', marker='*', linestyle='-', label='dev')
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.title('Train/Dev Error vs. Degree of predictor')
    plt.legend()
    plt.savefig('lambda_train_dev.png')


if __name__ == "__main__":

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        data_parsing()

    # data_processing()
    # xml_file_paths = Path("../chord-melody-dataset/").glob('**/*.xml')
    # xml_file_path_1 = "../chord-melody-dataset/a_smooth_one/a.xml"
    # xml_file_path_2 = "../chord-melody-dataset/a_smooth_one/b.xml"
    # s = converter.parse(xml_file_path_1)
    # s2 = converter.parse(xml_file_path_2)
   
    data_parsing()
   
    print("loading pickle files for training")
    with open(os.path.join(DATA_DIR, 'training_parsed_xml_files.pkl'), 'rb') as f:
        parsed_xml_files = pickle.load(f)
    print("DONE LOADING PICKLE FILES FOR TRAINING")
    nb = NaiveBayes(parsed_xml_files, lambda_=100)
    print("BEGIN TRAINING")
    nb.train()
    print("DONE TRAINING")
    # pprint.pprint(nb.chord_note_frequencies)
    # pprint.pprint(nb.chord_frequencies)

    with open(os.path.join(DATA_DIR, 'chord_note_frequencies_data.json'), 'w') as outfile:
        json.dump(nb.chord_note_frequencies, outfile)
    
    with open(os.path.join(DATA_DIR, 'chord_frequencies_data.json'), 'w') as outfile:
        json.dump(nb.chord_frequencies, outfile)

    with open(os.path.join(DATA_DIR, 'unique_notes_data.json'), 'w') as outfile:
        json.dump(list(nb.unique_notes), outfile)

    # print("done")

    # nb = NaiveBayes([], lambda_=1)

    
    # with open(os.path.join(DATA_DIR, 'chord_frequencies_data.json')) as json_file:
    #     data = json.load(json_file)
    #     nb.chord_frequencies = data

    # with open(os.path.join(DATA_DIR, 'chord_note_frequencies_data.json')) as json_file:
    #     data = json.load(json_file)
    #     nb.chord_note_frequencies = data
    
    # with open(os.path.join(DATA_DIR, 'unique_notes_data.json')) as json_file:
    #     data = json.load(json_file)
    #     nb.unique_notes = set(data)

    # print ("LOADED JSON DATA")
    # print("LOADING TRAINING PICKLE FILES")
    # with open(os.path.join(DATA_DIR, 'training_parsed_xml_files.pkl'), 'rb') as f:
    #     train_parsed_xml_files = pickle.load(f)

    # print("LOADING DEV PICKLE FILES")
    # with open(os.path.join(DATA_DIR, 'dev_parsed_xml_files.pkl'), 'rb') as f:
    #     dev_parsed_xml_files = pickle.load(f)
    
    # print("DONE LOADING PICKLE FILES FOR TESTING")
    # print("BEGIN EVALUATION")
    # train_accuracies = []
    # dev_accuracies = []
    # lambda_values = [100000, 10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001]
    # for lambda_ in lambda_values:
    #     print(f"LAMBDA = {lambda_}")
    #     nb.lambda_ = lambda_
    #     train_accuracy = nb.evaluate_on_dataset(train_parsed_xml_files)
    #     dev_accuracy = nb.evaluate_on_dataset(dev_parsed_xml_files)
    #     train_accuracies.append(train_accuracy)
    #     dev_accuracies.append(dev_accuracy)


    # train_accuracies = [0.17879621469077442, 0.18085267384201764, 0.18872241362426445, 0.20536103766614142, 0.1426434847635072, 0.09673370189354485, 0.051358064257671664, 0.036384192862039186, 0.03481736684204435]
    # dev_accuracies = [0.1645459623859475, 0.16715287691639252, 0.17832536776115698, 0.19415306312457328, 0.12668363230091242, 0.087952330705729, 0.05189001303457265, 0.036620942213394576, 0.03444851343802371]

    nb = NaiveBayes([], lambda_=100)
    with open(os.path.join(DATA_DIR, 'chord_frequencies_data.json')) as json_file:
        data = json.load(json_file)
        nb.chord_frequencies = data

    with open(os.path.join(DATA_DIR, 'chord_note_frequencies_data.json')) as json_file:
        data = json.load(json_file)
        nb.chord_note_frequencies = data
    
    with open(os.path.join(DATA_DIR, 'unique_notes_data.json')) as json_file:
        data = json.load(json_file)
        nb.unique_notes = set(data)

    print ("LOADED JSON DATA")

    print("LOADING TESTING PICKLE FILES")
    with open(os.path.join(DATA_DIR, 'test_parsed_xml_files.pkl'), 'rb') as f:
        test_parsed_xml_files = pickle.load(f)
    
    print("DONE LOADING PICKLE FILES FOR TESTING")
    print("BEGIN EVALUATION")
    test_accuracy = nb.evaluate_on_dataset(test_parsed_xml_files)
    print(test_accuracy)
    
    
    # plot_sweep([x for x in lambda_values], [1-x for x in train_accuracies], [1-x for x in dev_accuracies])
    



    # print("BEGIN EVALUATION")
    # nb.evaluate_on_dataset(parsed_xml_files)

    exit()

    # print(nb.predict_chord_for_measure(s.parts[0][10]))
    # s.parts[0][10].show('text')

  