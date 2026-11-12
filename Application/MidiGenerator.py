import midiutil
from midiutil.MidiFile import MIDIFile
from collections import defaultdict
from itertools import chain
from math import log2
import numpy as np

def rle(bar: list, loops: int) -> list:
    """
    A recursive implementation of run-length encoding. Used to convert note positions into note lengths.
    :param bar: The positions within a bar of a single pitch.
    :param loops: The number of interactions left until all positions have been checked.
    :return: A list of note durations.
    """
    if loops == 0:
        return bar
    for i, v in enumerate(bar[:-1]):
        if v != 0:
            bar[i] = bar[i] + bar[i + 1]
            bar[i + 1] = 0
    return rle(bar, loops-1)


def inferBar(bar: list) -> dict:
    """
    Generates midi signals for a single bar of music
    :param bar: A single bars worth of pitches
    :return: A dictionary of pitches and when they occur in the bar.
    """
    notes = set(chain.from_iterable(bar))
    result = defaultdict(list)
    for beat in bar:
        for note in notes:
            if note in beat:
                result[note].append(1)
            else:
                result[note].append(0)

    for key, value in result.items():
        result[key] = rle(value, len(bar))
    return result

class MidiGenerator:
    """
    An object which creates midi objects from model predictions.
    """
    def __init__(self, tags: dict, threshold=0.6):
        """
        The default constructor.
        :param tags: The other features of the file such as tempo, time signature etc.
        :param threshold: The minimum confidence the model should have to approve a note.
        """
        self.threshold = threshold
        self.title = tags['title']
        self.tempo = tags['tempo']
        self.time_signature = tags['time_signature']
        self.division = 2**tags['note_values']

    def filterPredictions(self, predictions: np.ndarray) -> list:
        """
        Converts the inference results into pitches.
        :param predictions: The output from the model.
        :return: A list of pitches for every note.
        """
        return [np.argwhere(prediction > self.threshold).flatten().tolist() for prediction in predictions]


    def generateMidi(self, score: list) -> midiutil.MIDIFile:
        """
        Generates a midifile from a list of pitch values.
        :param score: The pitches for the whole song.
        :return: A midifile containing signal generated from inference.
        """
        mf = MIDIFile(1)

        mf.addTrackName(0, 0, self.title)
        mf.addTempo(0, 0, self.tempo)
        measureCount, noteType = list(map(int, list(self.time_signature)))
        mf.addTimeSignature(0, 0, measureCount, int(log2(noteType)), int(24/self.division))

        for i in range(0, len(score), measureCount):
            bar = score[i:i+measureCount]
            notesInBar = inferBar(bar)
            for pitch, durations in notesInBar.items():
                for j, duration in enumerate(durations):
                    if duration != 0:
                        mf.addNote(0, 0, pitch, (i+j)/self.division, duration, 127)

        return mf
