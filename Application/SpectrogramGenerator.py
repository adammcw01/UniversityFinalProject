from scipy.io.wavfile import read
import tensorflow as tf
import numpy as np
import taglib
import os

class SpectrogramGenerator:
    """
    A custom object for pre-processing data before inference.
    """
    def __init__(self, filepath: str, model_shape: tuple):
        """
        The default constructor for the Spectrogram Generator object.
        :param filepath: The filepath for the audio file to be inferred.
        :param model_shape: A tuple with the dimensions for the first layer of the model.
        """
        self.filepath = filepath
        self.title = os.path.splitext(os.path.basename(filepath))[0]
        try:
            with taglib.File(filepath) as file:
                self.sampleRate = file.sampleRate
                tags = file.tags
                self.division = int(tags['NOTE_DIVISION'][0])
                self.tempo = float(tags['TEMPO'][0])
                self.timeSignature = tags['TIME_SIGNATURE'][0]
                self.targetFrameLength = model_shape[:-1]
                self.corrupt = False
        except OSError:
            self.corrupt = True
        except KeyError:
            self.corrupt = True

    def loadSample(self) -> (tf.constant, bool):
        """
        Loads an audio file and splits it into individual notes.
        :returns: A batch of tensors and a flag if the function was successful.
        """
        try:
            """
            Convert stereo to mono
            """
            wavfile = np.mean(read(self.filepath)[-1], axis=1)
        except ValueError:
            """
            Return False if file is corrupt.
            """
            return None, False
        samplesPerNote = self.getWindowSize()
        numSpectrograms = int(wavfile.shape[0] / samplesPerNote)
        samples = np.empty((numSpectrograms, samplesPerNote))
        for i in range(numSpectrograms):
            samples[i] = wavfile[i*samplesPerNote:(i+1)*samplesPerNote]
        return tf.convert_to_tensor(samples), True

    def getWindowSize(self) -> int:
        """
        Calculates the number of samples in a single note.
        :return: The window size of each note.
        """
        durationOfBeat = 60/self.tempo
        samplesPerBeat = self.sampleRate * durationOfBeat
        return int(samplesPerBeat/(2 ** self.division))

    def getSpectrogram(self, sample: tf.constant) -> tf.constant:
        """
        Converts a tensor of audio signals into a spectrogram.
        :param sample: The data for a single note from the audio file
        :return: A spectrogram.
        """
        spectrogram = tf.signal.stft(sample, frame_length=256, frame_step=64)
        spectrogram = tf.abs(spectrogram[:self.targetFrameLength[1]])
        padding = [
            [0, self.targetFrameLength[1]-spectrogram.shape[0]],
            [0, self.targetFrameLength[0]-spectrogram.shape[1]]
        ]
        """
        Padding ensures that the spectrograms are compatible with the model.
        """
        spectrogram = tf.pad(spectrogram, padding, "CONSTANT")
        spectrogram = tf.transpose(spectrogram)
        return tf.expand_dims(spectrogram, axis=2)

    def getTags(self) -> dict:
        """
        Parses the metadata from an audio file.
        :return: A dictionary of the metadata tags and their values.
        """
        return {'title': self.title, 'tempo': self.tempo, 'note_values': self.division, 'time_signature': self.timeSignature}
