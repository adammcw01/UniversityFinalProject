from unittest import TestCase
from Application import MidiGenerator
from midiutil.MidiFile import MIDIFile
import numpy as np

class TestMidiGenerator(TestCase):
    def setUp(self) :
        self.tags = {"title": "title", "tempo": 120, "time_signature": "44", "note_values": 0}
        self.testObj = MidiGenerator.MidiGenerator(self.tags)
        self.predictions = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.4, 0.3, 0.2, 0.1], [0.0, 0.3, 0.7, 0.0]])

    def test_constructor_good(self):
        obj = MidiGenerator.MidiGenerator(self.tags)
        self.assertIsInstance(obj, MidiGenerator.MidiGenerator)

    def test_constructor_bad(self):
        self.assertRaises(KeyError, lambda: MidiGenerator.MidiGenerator({}))

    def test_note_division(self):
        self.assertEqual(self.testObj.division, 1)

    def test_filterPredictions_valid(self):
        actual = self.testObj.filterPredictions(self.predictions)
        expected = [[0],[1],[],[2]]
        self.assertEqual(actual, expected)

    def test_filterPredictions_empty(self):
        actual = self.testObj.filterPredictions(np.array((1,)))
        expected = [[]]
        self.assertEqual(actual, expected)

    def test_generateMidi_good(self):
        mf = self.testObj.generateMidi([[0], [1], [2], [3]])
        self.assertIsInstance(mf, MIDIFile)

    def test_generateMidi_empty(self):
        mf = self.testObj.generateMidi([[],[],[],[]])
        self.assertIsInstance(mf, MIDIFile)

    def test_inferBar_good(self):
        actual = MidiGenerator.inferBar([[1, 2],[2, 3],[3, 4],[4, 5]])
        expected = {1: [1, 0, 0, 0], 2: [2, 0, 0, 0], 3: [0, 2, 0, 0], 4: [0, 0, 2, 0], 5: [0, 0, 0, 1]}
        self.assertEqual(actual, expected)

    def test_inferBar_empty(self):
        actual = MidiGenerator.inferBar([[],[],[],[]])
        expected = {}
        self.assertEqual(actual, expected)

    def test_inferBar_type(self):
        result = MidiGenerator.inferBar([[],[],[],[]])
        self.assertIsInstance(result, dict)

    def test_rle_good(self):
        actual = MidiGenerator.rle([1, 1, 1, 1], 4)
        expected = [4, 0, 0, 0]
        self.assertEqual(actual, expected)

    def test_rle_variation(self):
        actual = MidiGenerator.rle([0, 1, 1, 0], 4)
        expected = [0, 2, 0, 0]
        self.assertEqual(actual, expected)

    def test_rle_gap(self):
        actual = MidiGenerator.rle([1, 1, 0, 1], 4)
        expected = [2, 0, 0, 1]
        self.assertEqual(actual, expected)

    def test_rle_empty(self):
        actual = MidiGenerator.rle([], 4)
        expected = []
        self.assertEqual(actual, expected)

    def test_rle_type(self):
        result = MidiGenerator.rle([], 4)
        self.assertIsInstance(result, list)