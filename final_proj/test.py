from music21 import *
import math
from typing import List, Dict
import pprint
from pathlib import Path
from multiprocessing import Process
from threading import Thread, Lock
from tqdm import tqdm
from torch import nn
from pydantic import BaseModel


mutex = Lock()

NUM_SAMPLES = 0

class MusicRNNParams(BaseModel):
    vocab_dim: int = 17
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    chord_dim: int # calculate chord dims by going through every song and checking unique chords


class MusicRNN(nn.Module):
    def __init__(self, params: MusicRNNParams):
        super(MusicRNN, self).__init__()

        self.encoder_decoder = nn.Sequential(
            nn.Embedding(
                num_embeddings=params.vocab_dim, 
                embedding_dim=params.embedding_dim
            ),
            nn.RNN()
        )



def parse_xml(xml_file_paths: List[str], parsed_xml_files: List[stream.Score]):
    for index in tqdm(range(len(xml_file_paths))):
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


if __name__ == "__main__":

    # s = converter.parse("/Users/aharhar/Documents/USC Senior Year/CSCI 467/final_proj/chord-melody-dataset/hey_jude/d.xml")

    xml_file_paths = Path("/Users/aharhar/Documents/USC Senior Year/CSCI 467/final_proj/chord-melody-dataset/").glob('**/*.xml')
    xml_file_paths = [str(path) for path in xml_file_paths]
    measures: List[stream.Measure] = []

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

    print("done")

    for p in threads:
        p.join()

    print("fully done")

    print(len(parsed_xml_files))
    for index, s in enumerate(parsed_xml_files):
        part = s.parts[0]
        measures = []
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
                    process_notes_so_far(notes_so_far, datapoint, note_chord_frequencies)
                    process_chord(datapoint, chord_frequencies)



    # # s.show('text')
    # part = s.parts[0]
    # measures = []
    # for x in part:
    #     if isinstance(x, stream.Measure):
    #         measures.append(x)

    # note_chord_frequencies = {}
    # chord_frequencies = {}

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

    # pprint.pprint(note_chord_frequencies)
    # pprint.pprint(chord_frequencies)
        
    
    
        
    
        
   
    # s.write('midi', fp='/Users/aharhar/Documents/USC Senior Year/CSCI 467/final_proj/chord-melody-dataset/hey_jude/d.mid')

   
    # first_two = s.measures(0, 30)
    # first_two.show('text')
    # print(len(first_two.parts[0]))


