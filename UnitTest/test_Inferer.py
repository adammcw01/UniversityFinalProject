from unittest import TestCase
from Application import Inferer
import keras
import numpy as np

class TestInferer(TestCase):
    def setUp(self) :
        self.goodModel = r"Resources\ValidModel"
        self.badModel = r"Resources\InvalidModel"

        self.testObj = Inferer.Inferer(self.goodModel)

    def test_loadModel_Good(self):
        model = Inferer.loadModel(self.goodModel)
        self.assertTrue(isinstance(model, keras.Model))

    def test_loadModel_Bad(self):
        model = Inferer.loadModel(self.badModel)
        self.assertFalse(model)

    def test_loadModel_NotFound(self):
        model = Inferer.loadModel(r"Resources\InvalidDir")
        self.assertFalse(model)

    def test_constructor_good(self):
        inferer = Inferer.Inferer(self.goodModel)
        self.assertIsInstance(inferer, Inferer.Inferer)

    def test_constructor_goodValid(self):
        inferer = Inferer.Inferer(self.goodModel)
        self.assertTrue(inferer.validModel)

    def test_constructor_bad(self):
        inferer = Inferer.Inferer(self.badModel)
        self.assertFalse(inferer.validModel)

    def test_predict_good(self):
        spectrogram = np.random.uniform(0, 1, (1, 129, 153))
        result, valid = self.testObj.predict(spectrogram)
        self.assertIsInstance(result, np.ndarray)

    def test_predict_goodValid(self):
        spectrogram = np.random.uniform(0, 1, (1, 129, 153))
        result, valid = self.testObj.predict(spectrogram)
        self.assertTrue(valid)

    def test_predict_bad(self):
        spectrogram = np.random.uniform(0, 1, 1)
        result, valid = self.testObj.predict(spectrogram)
        self.assertIsInstance(result, ValueError)

    def test_predict_badValid(self):
        spectrogram = np.random.uniform(0, 1, 1)
        result, valid = self.testObj.predict(spectrogram)
        self.assertFalse(valid)
