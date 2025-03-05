"""
test_transcription.py - Unit Tests for Transcription Module

This module contains unit tests for the transcription.py module,
testing the functionality of audio preprocessing, text validation,
and transcription with metadata extraction.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock, mock_open
import json

import numpy as np

# Import the module to test
from src.transcription import (
    AudioPreprocessor,
    TextInputValidator,
    TranscriptionEngine,
    TranscriptionSegment
)


class TestTranscriptionSegment(unittest.TestCase):
    """Test the TranscriptionSegment class"""

    def test_to_dict(self):
        """Test conversion of TranscriptionSegment to dictionary"""
        segment = TranscriptionSegment(
            text="Hello world",
            start_time=1.0,
            end_time=2.5,
            confidence=0.95,
            speaker="speaker_1"
        )
        
        expected_dict = {
            'text': "Hello world",
            'start_time': 1.0,
            'end_time': 2.5,
            'confidence': 0.95,
            'speaker': "speaker_1"
        }
        
        self.assertEqual(segment.to_dict(), expected_dict)


class TestAudioPreprocessor(unittest.TestCase):
    """Test the AudioPreprocessor class"""
    
    def setUp(self):
        """Set up the test environment"""
        self.preprocessor = AudioPreprocessor()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
    
    def tearDown(self):
        """Clean up the test environment"""
        # Clean up any temp files
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass
        
        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except:
                pass
    
    @patch('soundfile.write')
    @patch('librosa.resample')
    @patch('librosa.load')
    def test_preprocess(self, mock_load, mock_resample, mock_sf_write):
        """Test audio preprocessing workflow"""
        # Mock audio data and sample rate
        mock_audio = np.zeros(1000)  # Dummy audio data
        mock_sr = 44100  # Different from target sample rate to trigger resampling
        
        # Mock librosa.load to return our test data
        mock_load.return_value = (mock_audio, mock_sr)
        
        # Mock resample to return the same audio (we're testing the flow, not the actual resampling)
        mock_resample.return_value = mock_audio
        
        # Mock soundfile.write to create the file
        def mock_write_effect(path, data, sr):
            # Create an empty file at the path to simulate successful file writing
            with open(path, 'wb') as f:
                f.write(b'dummy audio content')
        
        mock_sf_write.side_effect = mock_write_effect
        
        # Call the method to test
        result = self.preprocessor.preprocess("fake_audio.wav")
        
        # Verify librosa.load was called
        mock_load.assert_called_once_with("fake_audio.wav", sr=None)
        
        # Verify resampling was called (since mock_sr != target_sr)
        mock_resample.assert_called_once()
        
        # Verify sf.write was called
        mock_sf_write.assert_called_once()
        
        # Verify the result is a path
        self.assertTrue(os.path.exists(result))
        
        # Add this file to the temp files to clean up
        self.temp_files.append(result)
    
    def test_normalize_audio(self):
        """Test audio normalization"""
        # Test with non-zero audio
        audio = np.array([0.1, 0.2, -0.3, 0.4, -0.5])
        normalized = self.preprocessor._normalize_audio(audio)
        self.assertAlmostEqual(np.max(np.abs(normalized)), 0.95, places=6)
        
        # Test with zero audio (edge case)
        zero_audio = np.zeros(5)
        normalized_zero = self.preprocessor._normalize_audio(zero_audio)
        np.testing.assert_array_equal(normalized_zero, zero_audio)


class TestTextInputValidator(unittest.TestCase):
    """Test the TextInputValidator class"""
    
    def setUp(self):
        """Set up the test environment"""
        self.validator = TextInputValidator(min_chars=10)
    
    @patch("builtins.open", new_callable=mock_open, read_data="This is a test text file with more than 10 characters.")
    def test_validate_success(self, mock_file):
        """Test successful text validation"""
        result = self.validator.validate("fake_text.txt")
        
        # Verify file was opened
        mock_file.assert_called_once_with("fake_text.txt", 'r', encoding='utf-8')
        
        # Verify the cleaned text was returned
        self.assertEqual(result, "This is a test text file with more than 10 characters.")
    
    @patch("builtins.open", new_callable=mock_open, read_data="Too short")
    def test_validate_too_short(self, mock_file):
        """Test validation of text that's too short"""
        with self.assertRaises(ValueError) as context:
            self.validator.validate("fake_text.txt")
        
        self.assertTrue("Text content is too short" in str(context.exception))
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        messy_text = "  Line with spaces  \r\n\r\n  Another line \t with tabs  \r\n\r\n"
        expected_clean = "Line with spaces\nAnother line with tabs"
        
        cleaned = self.validator._clean_text(messy_text)
        self.assertEqual(cleaned, expected_clean)


class TestTranscriptionEngine(unittest.TestCase):
    """Test the TranscriptionEngine class"""
    
    def setUp(self):
        """Set up the test environment"""
        self.engine = TranscriptionEngine(api_key="fake_api_key")
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
    
    def tearDown(self):
        """Clean up the test environment"""
        # Clean up any temp files
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass
        
        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except:
                pass
    
    @patch.object(AudioPreprocessor, 'preprocess')
    @patch('openai.Audio.transcribe')
    def test_transcribe_audio(self, mock_transcribe, mock_preprocess):
        """Test audio transcription workflow"""
        # Create a temporary directory and file for testing
        self.temp_dir = tempfile.mkdtemp()
        fake_processed_path = os.path.join(self.temp_dir, "processed.wav")
        
        # Create an empty file to ensure it exists
        with open(fake_processed_path, 'wb') as f:
            f.write(b'dummy audio content')
            
        # Mock preprocessor to return this fake path
        mock_preprocess.return_value = fake_processed_path
        
        # Mock OpenAI response
        mock_transcribe.return_value = {
            "segments": [
                {"text": "This is segment one.", "start": 0.0, "end": 2.5, "confidence": 0.98},
                {"text": "This is segment two.", "start": 2.5, "end": 5.0, "confidence": 0.95}
            ]
        }
        
        try:
            # Call the method to test
            segments = self.engine.transcribe_audio("fake_audio.wav")
            
            # Verify preprocessor was called
            mock_preprocess.assert_called_once_with("fake_audio.wav")
            
            # Verify OpenAI API was called
            mock_transcribe.assert_called_once()
            
            # Verify segments were created correctly
            self.assertEqual(len(segments), 2)
            self.assertEqual(segments[0].text, "This is segment one.")
            self.assertEqual(segments[0].start_time, 0.0)
            self.assertEqual(segments[0].end_time, 2.5)
            self.assertEqual(segments[0].confidence, 0.98)
            
            self.assertEqual(segments[1].text, "This is segment two.")
            self.assertEqual(segments[1].start_time, 2.5)
            self.assertEqual(segments[1].end_time, 5.0)
            self.assertEqual(segments[1].confidence, 0.95)
        finally:
            # Clean up the temporary file and directory
            if os.path.exists(fake_processed_path):
                os.unlink(fake_processed_path)
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
    
    @patch.object(TextInputValidator, 'validate')
    def test_process_text(self, mock_validate):
        """Test text processing workflow"""
        # Mock text validator
        mock_validate.return_value = "This is a test sentence. This is another sentence. And a third one!"
        
        # Call the method to test
        segments = self.engine.process_text("fake_text.txt")
        
        # Verify validator was called
        mock_validate.assert_called_once_with("fake_text.txt")
        
        # Verify text was split into segments
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0].text, "This is a test sentence.")
        self.assertEqual(segments[1].text, "This is another sentence.")
        self.assertEqual(segments[2].text, "And a third one!")
        
        # Verify timestamps were generated
        self.assertGreater(segments[0].end_time, segments[0].start_time)
        self.assertEqual(segments[1].start_time, segments[0].end_time)
    
    def test_process_input_audio(self):
        """Test input processing for audio files"""
        with patch.object(TranscriptionEngine, 'transcribe_audio') as mock_transcribe:
            # Setup mock
            mock_transcribe.return_value = [
                TranscriptionSegment(text="Test", start_time=0.0, end_time=1.0)
            ]
            
            # Call the method with an audio file
            self.engine.process_input("test.mp3")
            
            # Verify transcribe_audio was called
            mock_transcribe.assert_called_once_with("test.mp3")
    
    def test_process_input_text(self):
        """Test input processing for text files"""
        with patch.object(TranscriptionEngine, 'process_text') as mock_process:
            # Setup mock
            mock_process.return_value = [
                TranscriptionSegment(text="Test", start_time=0.0, end_time=1.0)
            ]
            
            # Call the method with a text file
            self.engine.process_input("test.txt")
            
            # Verify process_text was called
            mock_process.assert_called_once_with("test.txt")
    
    def test_process_input_unsupported(self):
        """Test input processing for unsupported file types"""
        with self.assertRaises(ValueError) as context:
            self.engine.process_input("test.pdf")
        
        self.assertTrue("Unsupported file format" in str(context.exception))
    
    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_transcript(self, mock_file, mock_json_dump):
        """Test saving transcript to file"""
        # Create test segments
        segments = [
            TranscriptionSegment(text="Segment 1", start_time=0.0, end_time=1.0, confidence=0.9),
            TranscriptionSegment(text="Segment 2", start_time=1.0, end_time=2.0, confidence=0.8)
        ]
        
        # Call the method to test
        self.engine.save_transcript(segments, "output.json")
        
        # Verify file was opened
        mock_file.assert_called_once_with("output.json", 'w', encoding='utf-8')
        
        # Verify json.dump was called
        mock_json_dump.assert_called_once()
        
        # Verify structure of the data passed to json.dump
        args, _ = mock_json_dump.call_args
        data = args[0]
        self.assertIn("segments", data)
        self.assertIn("metadata", data)
        self.assertEqual(len(data["segments"]), 2)
        self.assertEqual(data["metadata"]["segment_count"], 2)
        self.assertEqual(data["metadata"]["total_duration"], 2.0)


if __name__ == "__main__":
    unittest.main() 