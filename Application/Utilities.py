from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import shutil
import taglib
import os

def checkFileType(file: str) -> bool:
    """
    Checks if the file uploaded by the user is of a valid type.
    :param file: The filename of the users file.
    :return: A boolean for if the type is valid or not.
    """
    AllowedFiles = ['.wav', '.mp3','.aac', 'ogg']
    if os.path.splitext(file)[-1] in AllowedFiles:
        return True
    return False

def tagMetadata(filepath: str, tags: dict) -> bool:
    """
    Appends metadata to the file uploaded to the server.
    :param filepath: The serverside filepath.
    :param tags: The metadata tags and values to append.
    :return: returns True upon completion.
    """
    with taglib.File(filepath, save_on_exit=True) as file:
        for tag in tags:
            file.tags[tag] = tags[tag]
    return True

def setupFilesystem():
    """
    This function runs on server startup and clears the filesystem from the previous instance.
    :return: None
    """
    try:
        os.remove(r'static\Score.pdf')
    except FileNotFoundError:
        pass
    shutil.copy(r'variables\Score.pdf', 'static')
    for file in os.listdir('variables'):
        if file != "Score.pdf":
            try:
                os.remove(rf'variables\{file}')
            except FileNotFoundError:
                pass
    return None

def getFilePath(file: FileStorage) -> str:
    """
    Generates a serverside filepath for a file uploaded by the user.
    :param file: The file in the http request.
    :return: The serverside filepath.
    """
    filepath = os.path.join('variables', secure_filename(file.filename))
    return filepath
