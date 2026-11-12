from unittest import TestCase
import tensorflow as tf
import Inferer
import shutil
import Main
import os

class TestMain(TestCase):
    def setUp(self) :
        self.goodModel = r"Resources\ValidModel"
        self.badModel = r"Resources\InvalidModel"

        self.testInferer = Main.loadInferer(self.goodModel)
        self.test_shape = self.testInferer.input_shape

        shutil.copy(r"Resources\GoodAudio.wav", r"Resources\Temp")

        self.goodTestFile = r"Resources\Temp\GoodAudio.wav"
        spectrograms, tags, valid = Main.prepareData(self.goodTestFile, self.test_shape)
        self.spectrograms = spectrograms
        self.tags = tags
        self.prepare_valid = valid

    def test_loadInferer_valid(self):
        inferer = Main.loadInferer(self.goodModel)
        self.assertIsInstance(inferer, Inferer.Inferer)

    def test_loadInferer_invalid(self):
        inferer = Main.loadInferer(self.badModel)
        self.assertFalse(inferer)

    def test_loadInferer_error(self):
        inferer = Main.loadInferer(r"Resources\InvalidDir")
        self.assertFalse(inferer)

    def test_inferer_shape(self):
        inferer = Main.loadInferer(self.goodModel)
        self.assertEqual((129, 153, 1), inferer.input_shape)

    def test_getInputFile(self):
        os.chdir("../Application")
        result = Main.getInputFile()
        os.chdir("../UnitTest")
        self.assertFalse(result)

    def test_prepareData_type(self):
        self.assertTrue(tf.is_tensor(self.spectrograms))

    def test_prepareData_spectrogramShape(self):
        expected = self.test_shape
        actual = self.spectrograms[0].shape
        self.assertEqual(expected, actual)

    def test_prepareData_tags_type(self):
        self.assertIsInstance(self.tags, dict)

    def test_prepareData_tags(self):
        expected = {'title': 'GoodAudio', 'tempo': 120.0, 'note_values': 0, 'time_signature': '44'}
        self.assertEqual(self.tags, expected)

    def test_prepareData_valid(self):
        self.assertTrue(self.prepare_valid)

