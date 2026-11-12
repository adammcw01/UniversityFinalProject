from unittest import TestCase
import SpectrogramGenerator
import tensorflow as tf
import numpy as np
import os

class TestSpectrogramGenerator(TestCase):
    def setUp(self) :
        self.goodTestFile = r"Resources\GoodAudio.wav"
        self.badTestFile = r"Resources\NotAudio!.txt"
        self.corruptTestFile = r"Resources\CorruptAudio.wav"

        self.testObj = SpectrogramGenerator.SpectrogramGenerator(self.goodTestFile, (150, 150, 1))

    def test_constructor_good(self):
        obj = SpectrogramGenerator.SpectrogramGenerator(self.goodTestFile, (1, 1))
        self.assertIsInstance(obj, SpectrogramGenerator.SpectrogramGenerator)

    def test_constructor_bad(self):
        obj = SpectrogramGenerator.SpectrogramGenerator(self.badTestFile, (1, 1))
        self.assertTrue(obj.corrupt)

    def test_constructor_corrupt(self):
        obj = SpectrogramGenerator.SpectrogramGenerator(self.corruptTestFile, (1, 1))
        self.assertTrue(obj.corrupt)

    def test_filepath(self):
        self.assertEqual(self.testObj.filepath, self.goodTestFile)

    def test_filepath_exists(self):
        self.assertTrue(os.path.isfile(self.testObj.filepath))

    def test_title(self):
        self.assertEqual(self.testObj.title, "GoodAudio")

    def test_division(self):
        self.assertEqual(self.testObj.division, 0)

    def test_tempo(self):
        self.assertEqual(self.testObj.tempo, 120.0)

    def test_timeSignature(self):
        self.assertEqual(self.testObj.timeSignature, '44')

    def test_targetFrameLength(self):
        self.assertEqual(self.testObj.targetFrameLength, (150, 150))

    def test_getTags_good(self):
        expected = {'title': 'GoodAudio', 'tempo': 120.0, 'note_values': 0, 'time_signature': '44'}
        actual = self.testObj.getTags()
        self.assertEqual(expected, actual)

    def test_getSpectrogram_shape(self):
        sample = np.random.uniform(-44100, 44100, (768000,))
        spectrogram = self.testObj.getSpectrogram(sample)
        self.assertEqual(spectrogram.shape, (150, 150, 1))

    def test_getSpectrogram_smallerDivision(self):
        sample = np.random.uniform(-44100, 44100, (768000,))
        self.testObj.division = 2
        spectrogram = self.testObj.getSpectrogram(sample)
        self.assertEqual(spectrogram.shape, (150, 150, 1))

    def test_getSpectrogram_type(self):
        sample = np.random.uniform(-44100, 44100, (768000,))
        spectrogram = self.testObj.getSpectrogram(sample)
        self.assertTrue(tf.is_tensor(spectrogram))
