from typing import List, Dict
from music21 import *
from collections import defaultdict
import re
import pprint
import numpy as np

class NaiveBayes():
    def __init__(self, parsed_xml_files: List[stream.Score], lambda_: int = 1):
        self.scores = parsed_xml_files
        self.chord_note_frequencies: Dict[str, Dict[str, int]] = {}
        self.chord_frequencies: Dict[str, int] = {}
        self.lambda_ = lambda_
        self.unique_notes = set()

    def process_notes_so_far(self, notes_so_far: List[note.Note], chord_class: chord.Chord):
        for note_class in notes_so_far:
            number_replacement_pattern = r'[0-9]'
            note = re.sub(number_replacement_pattern, '', note_class.pitch.name)
            self.unique_notes.add(note)
            chord = re.sub(number_replacement_pattern, '', f"{chord_class.root()} {chord_class.commonName}")
            if chord not in self.chord_note_frequencies:
                self.chord_note_frequencies[chord] = defaultdict(int)
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
        for s in self.scores:
            part = s.parts[0]
            measures: List[stream.Measure] = []
            key = None
            for x in part:
                if isinstance(x, stream.Measure):
                    measures.append(x)

            notes_so_far = []
            for measure in measures:
                # breakpoint()
                last_chord_in_measure = None
                for datapoint in measure:
                    if isinstance(datapoint, note.Note):
                        notes_so_far.append(datapoint)
                    elif isinstance(datapoint, chord.Chord):
                        last_chord_in_measure = datapoint
                        self.process_notes_so_far(notes_so_far, datapoint)
                        self.process_chord(datapoint)
                if len(notes_so_far) > 0:
                    self.process_notes_so_far(notes_so_far, last_chord_in_measure)

        # pprint.pprint(self.chord_note_frequencies)
        # print("")
        # pprint.pprint(self.chord_frequencies)

    def predict_chord_for_measure(self, measure: stream.Measure):
        measure_notes: List[str] = []
        for datapoint in measure:
            if isinstance(datapoint, note.Note):
                number_replacement_pattern = r'[0-9]'
                measure_notes.append(re.sub(number_replacement_pattern, '', datapoint.pitch.name))

        chords = list(self.chord_frequencies.keys())

        num_unique_notes = len(self.unique_notes)

        most_likely_chord = None
        max_prob = -np.inf
        lambda_ = self.lambda_
        
        for chord_label in chords:
            prob_chord_label = np.log(self.chord_frequencies[chord_label] / sum(self.chord_frequencies.values()))
            num_chord_label = sum(self.chord_note_frequencies[chord_label].values())
            denominator = num_chord_label + lambda_ * num_unique_notes
            for note_data in measure_notes:
                prob_chord_given_note = np.log((self.chord_note_frequencies[chord_label][note_data] + lambda_) / denominator)
                prob_chord_label += prob_chord_given_note
            if prob_chord_label > max_prob:
                max_prob = prob_chord_label
                most_likely_chord = chord_label
        return most_likely_chord
    

    def predict_on_song(self, score: stream.Score):
        part = score.parts[0]
        measures: List[stream.Measure] = []
        for x in part:
            if isinstance(x, stream.Measure):
                measures.append(x)

        for index, measure in enumerate(measures):
            print(f"----------- MEASURE {index+1} ------------")
            measure.show('text')
            print(self.predict_chord_for_measure(measure))