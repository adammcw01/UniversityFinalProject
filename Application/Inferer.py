import numpy as np
import tensorflow as tf
import os

def loadModel(modelPath: str) -> tf.keras.models.Model | bool:
    """
    Loads a deep learning model using Keras.
    :param modelPath: The filepath of the saved model.
    :return: A saved model object ready for inference.
    :returns: False if the path does not exist or does not contain a model.
    """
    if not os.path.exists(modelPath):
        return False
    try:
        return tf.keras.models.load_model(modelPath)
    except OSError:
        return False

class Inferer:
    """
    An Inferer object used to make predictions
    """
    def __init__(self, modelPath: str):
        """
        The constructor for Inferer.
        :param modelPath: The filepath to saved model.
        """
        self.model = loadModel(modelPath)
        if self.model:
            self.validModel = True
            self.input_shape = tuple(self.model.input.shape[1:])
        else:
            self.validModel = False

    def predict(self, sample:tf.constant) -> (np.ndarray, bool):
        """
        Makes a prediction using the model.
        :param sample: A spectrogram or batch of spectrograms.
        :return: A numpy array of confidence values.
        :returns: A valid flag for error prevention.
        """
        try:
            result = self.model.predict(sample)
            return result, True
        except ValueError as error:
            return error, False