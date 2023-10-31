import os
import pickle
from music21 import *
import math
from typing import List, Dict
import pprint
from pathlib import Path
from multiprocessing import Process
from threading import Thread, Lock
from tqdm import tqdm


mutex = Lock()

NUM_SAMPLES = 0

# to create the data dir
PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
DATA_DIR = os.path.join(PARENT_DIR, 'data')

def parse_xml(xml_file_paths: List[str], parsed_xml_files: List[stream.Score]):
    for index in (range(len(xml_file_paths))): # removed tqdm as it was kinda buggy
        s = converter.parse(xml_file_paths[index])
        mutex.acquire()
        parsed_xml_files.append(s)
        mutex.release()

def process_notes_so_far(notes_so_far: List[note.Note], chord_class: chord.Chord, note_chord_frequencies: Dict[str, Dict[str, int]]): 
    for note_class in notes_so_far:
        note = note_class.pitch.name
        chord = f"{chord_class.root()} {chord_class.commonName}"
        if note not in note_chord_frequencies:
            note_chord_frequencies[note] = {}
        if chord not in note_chord_frequencies[note]:
            note_chord_frequencies[note][chord] = 0
        note_chord_frequencies[note][chord] += 1
    notes_so_far = []

def process_chord(chord: chord.Chord, chord_frequencies: Dict[str, int]):
    chord_name = f"{chord.root()} {chord.commonName}"
    if chord_name not in chord_frequencies:
        chord_frequencies[chord_name] = 0
    chord_frequencies[chord_name] += 1

def threaded_parse_xml(sample, partition_type):

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
    with open(os.path.join(DATA_DIR, 'training_parsed_xml_files.pkl'), 'rb') as f:
        parsed_xml_files = pickle.load(f)

    print(len(parsed_xml_files))
    for index, s in enumerate(parsed_xml_files):
        part = s.parts[0]
        measures: List[stream.Measure] = []
        for x in part:
            if isinstance(x, stream.Measure):
                measures.append(x)

        note_chord_frequencies = {}
        chord_frequencies = {}

        print(f"----------------- ANALYZING SONG {index} -----------------")

        for index, measure in enumerate(measures):
            print(F"---------------- MEASURE {index} ----------------")
            notes_so_far = []
            for datapoint in measure:
                if isinstance(datapoint, note.Note):
                    notes_so_far.append(datapoint)
                    print(F"PITCH: {datapoint.pitch}")
                    # notes_so_far.append(datapoint.pitch)
                elif isinstance(datapoint, chord.Chord):
                    print(F"CHORD: {datapoint.root()} {datapoint.commonName}")
                    process_notes_so_far(
                        notes_so_far, datapoint, note_chord_frequencies)
                    process_chord(datapoint, chord_frequencies)

if __name__ == "__main__":

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        data_parsing()

    data_processing()