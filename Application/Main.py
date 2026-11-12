from SpectrogramGenerator import SpectrogramGenerator
from MidiGenerator import MidiGenerator
from music21.converter import parse
from Inferer import Inferer
import tensorflow as tf
import os


def loadInferer(path=r"model") -> Inferer | bool:
    """
    Loads and validates the deep learning model.
    :param path: The path of the saved model.
    :return: A custom Inferer object.
    :returns: False if the model is invalid.
    """
    inferer = Inferer(path)
    if not inferer.validModel:
        return False
    return inferer


def getInputFile() -> str | bool:
    """
    Finds the most recently uploaded file by the user
    :return: The filepath to be inferred.
    :returns: False if no suitable files can be found.
    """
    AllowedFiles = ['.wav', '.mp3', '.aac', 'ogg']
    audioDirectory = r'variables'
    files = [file for file in os.listdir(audioDirectory) if os.path.splitext(file)[-1] in AllowedFiles]
    try:
        sampleFilepath = os.path.join(audioDirectory, files[-1])
    except IndexError:
        return False
    return os.path.abspath(sampleFilepath)

def prepareData(path: str, input_shape: tuple) -> (tf.constant, dict, bool):
    """
    Generate spectrograms using a generator object.
    :param path: The serverside filepath of the audio recording.
    :param input_shape: The shape of the first layer of the deep learning model.
    :return: A 3-rank tensor of spectrograms.
    :return: A dictionary of the files' metadata.
    :return: A flag to determine if the function was successful or not.
    """
    generator = SpectrogramGenerator(path, input_shape)
    if generator.corrupt:
        return None, None, False
    samples, validSamples = generator.loadSample()
    if not validSamples:
        return None, None, False
    os.remove(path)
    spectrograms = [generator.getSpectrogram(sample) for sample in samples]
    tags = generator.getTags()
    return tf.stack(spectrograms), tags, True

def main():
    """
    Performs inference on the file uploaded to the server by the user.
    :return: A string with an error message if one is applicable.
    """

    """
    Load Model
    """
    inferer = loadInferer()
    if not inferer:
        return "Unable to load model."
    image_shape = inferer.input_shape

    """
    Load Audio Recording
    """
    samplePath = getInputFile()
    if not samplePath:
        return None

    """
    Generate Spectrograms
    """
    spectrograms, tags, validSpectrograms = prepareData(samplePath, image_shape)
    if not validSpectrograms:
        return "Could not read file: Corrupted!"

    """
    Make Predictions and Convert to MIDI
    """
    predictions, valid = inferer.predict(spectrograms)
    if not valid:
        return "Invalid model input shape"
    midi = MidiGenerator(tags)
    notes = midi.filterPredictions(predictions)
    midiFile = midi.generateMidi(notes)
    midiPath = os.path.join(r'variables', f"{tags['title']}.mid")
    with open(midiPath, 'wb') as file:
        midiFile.writeFile(file)

    """
    Visualise MIDI Through Lilypond
    """
    score = parse(midiPath)
    score.metadata.title = tags['title']
    score.metadata.composer = ""
    scorePath = os.path.join(r'static', 'score.pdf')
    score.write('musicxml.pdf', fp=scorePath)

    return None

if __name__ == '__main__':
    """
    The entry point for the Musical Notation Generator Application
    """
    main()
