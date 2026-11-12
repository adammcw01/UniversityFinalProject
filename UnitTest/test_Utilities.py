from unittest import TestCase
from Application import Utilities
from werkzeug.datastructures import FileStorage
import os

class TestUtilities(TestCase):
    def setUp(self) :
        self.goodTestFile = r"Resources\GoodAudio.wav"
        self.badTestFile = r"Resources\NotAudio!.txt"

        with open(self.goodTestFile, 'rb') as f:
            self.goodFile = FileStorage(f)

        with open(self.badTestFile, 'rb') as f:
            self.badFile = FileStorage(f)

    def test_checkFileType_Good(self):
        result = Utilities.checkFileType(self.goodTestFile)
        self.assertTrue(result)

    def test_checkFileType_Bad(self):
        result = Utilities.checkFileType(self.badTestFile)
        self.assertFalse(result)

    def test_getFilePath_SafeFileName(self):
        expected = r"variables\Resources_GoodAudio.wav"
        actual = Utilities.getFilePath(self.goodFile)
        self.assertEqual(expected, actual)

    def test_getFilePath_NotSafeFileName(self):
        expected = r"variables\Resources_NotAudio.txt"
        actual = Utilities.getFilePath(self.badFile)
        self.assertEqual(expected, actual)

    def test_setupFilesystem_static(self):
        os.chdir(r"..\Application")
        Utilities.setupFilesystem()
        os.chdir(r"..\UnitTest")
        result = os.listdir(r"..\Application\static")
        self.assertTrue("Score.pdf" in result)

    def test_setupFilesystem_variables(self):
        os.chdir(r"..\Application")
        Utilities.setupFilesystem()
        os.chdir(r"..\UnitTest")
        expected = ["Score.pdf"]
        actual = os.listdir(r"..\Application\variables")
        self.assertEqual(expected, actual)
