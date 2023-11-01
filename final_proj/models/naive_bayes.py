from typing import List, Dict
from music21 import *
from collections import defaultdict
import re
import pprint
import numpy as np
from tqdm import tqdm

class NaiveBayes():
    def __init__(self, parsed_xml_files: List[stream.Score], lambda_: int = 1):
        self.scores = parsed_xml_files
        self.chord_note_frequencies: Dict[str, Dict[str, int]] = {}
        self.chord_frequencies: Dict[str, int] = defaultdict(int)
        self.chord_key_frequencies: Dict[str, Dict[str, int]] = {}
        self.lambda_ = lambda_
        self.unique_notes = set()

    def process_notes_so_far(self, notes_so_far: List[note.Note], chord_class: chord.Chord):
        for note_class in notes_so_far:
            number_replacement_pattern = r'[0-9]'
            note = re.sub(number_replacement_pattern, '', note_class.pitch.name)
            self.unique_notes.add(note)
            chord = re.sub(number_replacement_pattern, '', f"{chord_class.root()} {chord_class.commonName}")
            if chord not in self.chord_note_frequencies:
                self.chord_note_frequencies[chord] = {}
            if note not in self.chord_note_frequencies[chord]:
                self.chord_note_frequencies[chord][note] = 0
            self.chord_note_frequencies[chord][note] += 1
        notes_so_far = []

    def process_chord(self, chord: chord.Chord):
        number_replacement_pattern = r'[0-9]'
        chord_name = re.sub(number_replacement_pattern, '', f"{chord.root()} {chord.commonName}")
        if chord_name not in self.chord_frequencies:
            self.chord_frequencies[chord_name] = 0
        self.chord_frequencies[chord_name] += 1

    def train(self):
        for i in tqdm(range(len(self.scores))):
            s = self.scores[i]
            part = s.parts[0]
            measures: List[stream.Measure] = []
            key = None
            for x in part:
                if isinstance(x, stream.Measure):
                    measures.append(x)
            
            count = 0
            for measure in measures[1:]:
                # breakpoint()
                notes_so_far = []
                last_chord_in_measure = None
                for datapoint in measure:
                    # breakpoint()
                    if isinstance(datapoint, note.Note):
                        notes_so_far.append(datapoint)
                    elif isinstance(datapoint, chord.Chord):
                        last_chord_in_measure = datapoint
                        self.process_notes_so_far(notes_so_far, datapoint)
                        self.process_chord(datapoint)
                        notes_so_far = []
                if len(notes_so_far) > 0:
                    self.process_notes_so_far(notes_so_far, last_chord_in_measure)
                    notes_so_far = []

    def predict_chord_for_measure(self, measure: stream.Measure):
        measure_notes: List[str] = []
        measure_chords: List[str] = []
        for datapoint in measure:
            if isinstance(datapoint, note.Note):
                number_replacement_pattern = r'[0-9]'
                measure_notes.append(re.sub(number_replacement_pattern, '', datapoint.pitch.name))
            elif isinstance(datapoint, chord.Chord):
                number_replacement_pattern = r'[0-9]'
                measure_chords.append(re.sub(number_replacement_pattern, '', f"{datapoint.root()} {datapoint.commonName}"))

        chords = list(self.chord_frequencies.keys())

        num_unique_notes = len(self.unique_notes)

        most_likely_chord = None
        max_prob = -np.inf
        lambda_ = self.lambda_
        
        for chord_label in chords:
            prob_chord_label = np.log(self.chord_frequencies[chord_label] / sum(self.chord_frequencies.values()))
            prob_chord_label = 0
            num_notes_with_chord_label = 0
            if chord_label in self.chord_note_frequencies:
                num_notes_with_chord_label = sum(self.chord_note_frequencies[chord_label].values())
            denominator = num_notes_with_chord_label + lambda_ * num_unique_notes
            # print(f"P({chord_label}) = {self.chord_frequencies[chord_label]} / {sum(self.chord_frequencies.values())} = {prob_chord_label}")
            for note_data in measure_notes:
                note_chord_frequency = 0
                if chord_label in self.chord_note_frequencies:
                    if note_data in self.chord_note_frequencies[chord_label]:
                        note_chord_frequency = self.chord_note_frequencies[chord_label][note_data]
                prob_note_given_chord = np.log((note_chord_frequency + lambda_) / denominator)
                # print(f"P({note_data} | {chord_label}) = {note_chord_frequency + lambda_} / {denominator} = {prob_note_given_chord}")
                prob_chord_label += prob_note_given_chord
            # print(f"{chord_label}: {prob_chord_label}")
            if prob_chord_label > max_prob:
                max_prob = prob_chord_label
                most_likely_chord = chord_label
                
        return most_likely_chord, measure_chords
    

    def predict_on_song(self, score: stream.Score):
        part = score.parts[0]
        measures: List[stream.Measure] = []
        for x in part:
            if isinstance(x, stream.Measure):
                measures.append(x)

        for index, measure in enumerate(measures):
            # breakpoint()
            print(f"----------- MEASURE {index+1} ------------")
            measure.show('text')
            print(self.predict_chord_for_measure(measure))

    def evaluate_on_song(self, score: stream.Score):
        part = score.parts[0]
        measures: List[stream.Measure] = []
        for x in part:
            if isinstance(x, stream.Measure):
                measures.append(x)

        measures = measures[1:] # remove first unused measure
        # print(len(measures))

        number_correct = 0
        number_total = len(measures)
        for i in range(len(measures)):
            measure = measures[i]
            most_likely_chord, measure_chords = self.predict_chord_for_measure(measure)
            # print(f"----------- MEASURE {i+1} ------------")
            # print(f"Predicted chord: {most_likely_chord}")
            # print(f"Actual chords: {measure_chords}")
            if most_likely_chord in measure_chords:
                number_correct += 1
                # print("CORRECT!")
            # else:
                # print("INCORRECT!")
                
        
        accuracy = number_correct / number_total
        # print(f"Accuracy: {accuracy}")
        return number_correct, number_total
    

    def evaluate_on_dataset(self, scores: List[stream.Score]):
        number_correct = 0
        number_total = 0
        for i in tqdm(range(len(scores))):
            score = scores[i]
            returned_correct, returned_total = self.evaluate_on_song(score)
            number_correct += returned_correct
            number_total += returned_total
        
        accuracy = number_correct / number_total
        print(f"Accuracy: {accuracy}")
        return accuracy
    