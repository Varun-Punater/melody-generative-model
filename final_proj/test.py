from music21 import *
import math
from typing import List, Dict
import pprint
from pathlib import Path
from multiprocessing import Process
from threading import Thread, Lock
from tqdm import tqdm

from naive_bayes import NaiveBayes


mutex = Lock()

NUM_SAMPLES = 0

def parse_xml(xml_file_paths: List[str], parsed_xml_files: List[stream.Score]):
    for index in tqdm(range(len(xml_file_paths))):
        s = converter.parse(xml_file_paths[index])
        mutex.acquire()
        parsed_xml_files.append(s)
        mutex.release()


if __name__ == "__main__":

    # s = converter.parse("/Users/aharhar/Documents/USC Senior Year/CSCI 467/final_proj/chord-melody-dataset/hey_jude/d.xml")

    xml_file_paths = Path("../chord-melody-dataset/").glob('**/*.xml')
    xml_file_paths = [str(path) for path in xml_file_paths]

    notes = ['C', 'C#', 'D-', 'D', 'D#', 'E-', 'E', 'F', 'F#', 'G-', 'G', 'G#', 'A-', 'A', 'A#', 'B-', 'B']


    NUM_SAMPLES = len(xml_file_paths)
    NUM_TRAIN = math.floor(NUM_SAMPLES * 0.7)
    NUM_DEV = math.floor(NUM_SAMPLES * 0.1)
    NUM_TEST = NUM_SAMPLES - NUM_TRAIN - NUM_DEV

    NUM_THREADS = 10

    train_samples = xml_file_paths[0:NUM_TRAIN]
    dev_samples = xml_file_paths[NUM_TRAIN:NUM_TRAIN+NUM_DEV]
    test_samples = xml_file_paths[NUM_TRAIN+NUM_DEV:]

    threads: List[Thread] = []

    num_files_per_process = math.ceil(len(train_samples) / NUM_THREADS)
    parsed_xml_files = []

    for i in range(0, 10):
        p = Thread(target=parse_xml, args=(train_samples[num_files_per_process*i:num_files_per_process*(i+1)], parsed_xml_files))
        p.start()
        threads.append(p)

    for p in threads:
        p.join()


    print("finally done")
    
    nb = NaiveBayes(parsed_xml_files)
    nb.train()
    s = converter.parse(dev_samples[18])
    # nb.predict_on_song(s)
    parsed = [s]
    # nb = NaiveBayes(parsed)
    # nb.train()
    # dev_samples[11].show('text')
    nb.predict_on_song(s)
    print(dev_samples[18])


    # unique_chord_types = set()
    # for index, s in enumerate(parsed_xml_files):
    #     part = s.parts[0]
    #     measures: List[stream.Measure] = []
    #     for x in part:
    #         if isinstance(x, stream.Measure):
    #             measures.append(x)
        

    #     for index, measure in enumerate(measures):
    #         for datapoint in measure:
    #             if isinstance(datapoint, chord.Chord):
    #                 unique_chord_types.add(f"{datapoint.commonName}")

    
        # note_chord_frequencies = {}
        # chord_frequencies = {}

        # print(f"----------------- ANALYZING SONG {index} -----------------")

        # for index, measure in enumerate(measures):
        #     print(F"---------------- MEASURE {index} ----------------")
        #     notes_so_far = []
        #     for datapoint in measure:
        #         if isinstance(datapoint, note.Note):
        #             notes_so_far.append(datapoint)
        #             print(F"PITCH: {datapoint.pitch}")
        #             # notes_so_far.append(datapoint.pitch)
        #         elif isinstance(datapoint, chord.Chord):
        #             print(F"CHORD: {datapoint.root()} {datapoint.commonName}")
        #             process_notes_so_far(notes_so_far, datapoint, note_chord_frequencies)
        #             process_chord(datapoint, chord_frequencies)
        
    
    
        
    
        
   
    # s.write('midi', fp='/Users/aharhar/Documents/USC Senior Year/CSCI 467/final_proj/chord-melody-dataset/hey_jude/d.mid')

   
    # first_two = s.measures(0, 30)
    # first_two.show('text')
    # print(len(first_two.parts[0]))

