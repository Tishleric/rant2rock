"""
transcription.py - Audio Processing and Transcription Module

This module handles the processing of both audio files and text inputs.
For audio files, it applies preprocessing techniques and integrates with 
a transcription engine to generate transcripts with metadata.
For text files, it validates and structures the input.
"""

import os
import re
import json
import wave
import tempfile
from typing import Dict, Any, Union, List, Optional, Tuple
from dataclasses import dataclass
import logging

# Third-party imports - these would need to be installed
import numpy as np
# For audio processing
import librosa
import soundfile as sf
# For transcription (assuming OpenAI Whisper is used)
from openai import OpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TranscriptionSegment:
    """Data class to store transcription segments with metadata"""
    text: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None
    speaker: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary"""
        return {
            'text': self.text,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'confidence': self.confidence,
            'speaker': self.speaker
        }


class AudioPreprocessor:
    """Handles preprocessing of audio files before transcription"""
    
    def __init__(self, 
                 target_sr: int = 16000,
                 apply_noise_reduction: bool = True,
                 apply_normalization: bool = True):
        """
        Initialize the audio preprocessor
        
        Args:
            target_sr: Target sample rate (16kHz is common for speech recognition)
            apply_noise_reduction: Whether to apply noise reduction
            apply_normalization: Whether to apply audio normalization
        """
        self.target_sr = target_sr
        self.apply_noise_reduction = apply_noise_reduction
        self.apply_normalization = apply_normalization
    
    def preprocess(self, audio_path: str) -> str:
        """
        Preprocess audio file for improved transcription
        
        Args:
            audio_path: Path to the input audio file
        
        Returns:
            Path to the preprocessed audio file
        """
        logger.info(f"Preprocessing audio file: {audio_path}")
        
        # Load audio file
        try:
            audio, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            raise ValueError(f"Failed to load audio file: {e}")
        
        # Resample if needed
        if sr != self.target_sr:
            logger.info(f"Resampling from {sr}Hz to {self.target_sr}Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        
        # Apply noise reduction if enabled
        if self.apply_noise_reduction:
            logger.info("Applying noise reduction")
            audio = self._reduce_noise(audio)
        
        # Apply normalization if enabled
        if self.apply_normalization:
            logger.info("Applying audio normalization")
            audio = self._normalize_audio(audio)
        
        # Create a temporary directory if it doesn't exist
        temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate a unique filename in the temp directory
        preprocessed_path = os.path.join(temp_dir, f"preprocessed_{os.path.basename(audio_path)}")
        
        # Save preprocessed audio to the file
        try:
            sf.write(preprocessed_path, audio, sr)
            
            # Verify the file was created successfully
            if not os.path.exists(preprocessed_path):
                raise IOError(f"Failed to create preprocessed file at {preprocessed_path}")
                
            logger.info(f"Preprocessed audio saved to: {preprocessed_path}")
            return preprocessed_path
            
        except Exception as e:
            logger.error(f"Error saving preprocessed audio: {e}")
            raise ValueError(f"Failed to save preprocessed audio: {e}")
    
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply simple noise reduction
        
        Args:
            audio: Audio data as numpy array
        
        Returns:
            Noise-reduced audio
        """
        # Simple noise reduction by removing low amplitude signals
        # A more sophisticated approach would use spectral gating or external libraries
        noise_threshold = 0.005  # This threshold would need tuning
        audio_denoised = np.copy(audio)
        audio_denoised[np.abs(audio) < noise_threshold] = 0
        return audio_denoised
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to have a maximum absolute value of 0.95
        
        Args:
            audio: Audio data as numpy array
        
        Returns:
            Normalized audio
        """
        # Prevent division by zero
        if np.max(np.abs(audio)) > 0:
            normalization_factor = 0.95 / np.max(np.abs(audio))
            return audio * normalization_factor
        return audio


class TextInputValidator:
    """Validates and processes text file inputs"""
    
    def __init__(self, min_chars: int = 10):
        """
        Initialize the text validator
        
        Args:
            min_chars: Minimum number of characters for valid input
        """
        self.min_chars = min_chars
    
    def validate(self, text_path: str) -> str:
        """
        Validate a text file input
        
        Args:
            text_path: Path to the text file
        
        Returns:
            Validated and cleaned text content
        """
        logger.info(f"Validating text file: {text_path}")
        
        try:
            with open(text_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            raise ValueError(f"Failed to read text file: {e}")
        
        # Check if the content meets minimum requirements
        if len(content.strip()) < self.min_chars:
            logger.warning(f"Text content too short: {len(content.strip())} chars")
            raise ValueError(f"Text content is too short. Minimum required: {self.min_chars} characters")
        
        # Clean the text (remove excessive whitespace, normalize line endings)
        cleaned_text = self._clean_text(content)
        
        logger.info(f"Text validation successful: {len(cleaned_text)} characters")
        return cleaned_text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by normalizing whitespace and line endings
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Replace multiple whitespace with a single space, except for line breaks
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove whitespace at the beginning and end of lines
        text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)
        
        # Remove empty lines
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()


class TranscriptionEngine:
    """
    Handles the transcription of audio files using Whisper or similar
    and processes text inputs into segmented formats
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the transcription engine
        
        Args:
            api_key: API key for the transcription service, if required
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        
        self.audio_preprocessor = AudioPreprocessor()
        self.text_validator = TextInputValidator()
    
    def process_input(self, input_path: str) -> List[TranscriptionSegment]:
        """
        Process either audio file or text file input
        
        Args:
            input_path: Path to input file (audio or text)
        
        Returns:
            List of transcription segments
        """
        file_ext = os.path.splitext(input_path)[1].lower()
        
        # Audio file extensions
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        
        if file_ext in audio_extensions:
            return self.transcribe_audio(input_path)
        elif file_ext in ['.txt', '.md']:
            return self.process_text(input_path)
        else:
            logger.error(f"Unsupported file format: {file_ext}")
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats are audio files ({', '.join(audio_extensions)}) and text files (.txt, .md)")
    
    def process_audio(self, audio_path: str) -> List[TranscriptionSegment]:
        """
        Process audio file - wrapper for transcribe_audio for API consistency
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of transcription segments
        """
        return self.transcribe_audio(audio_path)
    
    def transcribe_audio(self, audio_path: str) -> List[TranscriptionSegment]:
        """
        Transcribe audio file and extract segments with metadata
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            List of transcription segments
        """
        logger.info(f"Transcribing audio file: {audio_path}")
        
        # Preprocess the audio file
        preprocessed_path = None
        try:
            # Preprocess the audio file
            preprocessed_path = self.audio_preprocessor.preprocess(audio_path)
            
            # Verify the preprocessed file exists before attempting to open it
            if not os.path.exists(preprocessed_path):
                raise FileNotFoundError(f"Preprocessed audio file not found at {preprocessed_path}")
            
            # Use OpenAI Whisper API to transcribe the audio
            # Note: In a real implementation, you might need to handle chunking for large files
            with open(preprocessed_path, "rb") as audio_file:
                transcription_response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",  # Request detailed output with timestamps
                    timestamp_granularities=["segment"]  # Get segment-level timestamps
                )
            
            # Process the response into our segment format
            segments = []
            response_data = json.loads(transcription_response.model_dump_json())
            for segment in response_data.get("segments", []):
                segments.append(TranscriptionSegment(
                    text=segment.get("text", "").strip(),
                    start_time=segment.get("start", 0.0),
                    end_time=segment.get("end", 0.0),
                    confidence=segment.get("confidence", None)
                ))
            
            logger.info(f"Transcription complete: {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise RuntimeError(f"Failed to transcribe audio: {e}")
        finally:
            # Clean up the temporary file in the finally block to ensure it always happens
            if preprocessed_path and os.path.exists(preprocessed_path):
                try:
                    os.unlink(preprocessed_path)
                    logger.debug(f"Cleaned up temporary file: {preprocessed_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary file {preprocessed_path}: {cleanup_error}")
    
    def process_text(self, text_path: str) -> List[TranscriptionSegment]:
        """
        Process a text file into segments
        
        Args:
            text_path: Path to the text file
        
        Returns:
            List of transcription segments
        """
        logger.info(f"Processing text file: {text_path}")
        
        # Validate and clean the text
        text_content = self.text_validator.validate(text_path)
        
        # Split into sentences or paragraphs
        segments = self._split_text_into_segments(text_content)
        
        logger.info(f"Text processing complete: {len(segments)} segments")
        return segments
    
    def _split_text_into_segments(self, text: str) -> List[TranscriptionSegment]:
        """
        Split text into segments based on sentences or paragraphs
        
        Args:
            text: Input text content
        
        Returns:
            List of transcription segments
        """
        # Simple sentence splitter using regex
        # In a real implementation, use a more sophisticated NLP-based approach
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        sentences = re.split(sentence_pattern, text)
        
        # Remove empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Create segments with artificial timestamps
        # In a real implementation, these would be more accurate
        segments = []
        current_time = 0.0
        
        for sentence in sentences:
            # Estimate duration based on word count (rough approximation)
            word_count = len(sentence.split())
            estimated_duration = max(1.0, word_count * 0.3)  # Assumes 3 words per second
            
            segments.append(TranscriptionSegment(
                text=sentence,
                start_time=current_time,
                end_time=current_time + estimated_duration,
                confidence=1.0  # Artificial confidence for text input
            ))
            
            current_time += estimated_duration
        
        return segments
    
    def save_transcript(self, segments: List[TranscriptionSegment], output_path: str) -> None:
        """
        Save transcription segments to a file
        
        Args:
            segments: List of transcription segments
            output_path: Path to save the output file
        """
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        data = {
            "segments": [segment.to_dict() for segment in segments],
            "metadata": {
                "segment_count": len(segments),
                "total_duration": segments[-1].end_time if segments else 0.0
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Transcript saved to: {output_path}")


# Usage example:
if __name__ == "__main__":
    # This would typically be set via environment variable or config file
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    
    transcriber = TranscriptionEngine(api_key=OPENAI_API_KEY)
    
    # Example for processing an audio file
    # segments = transcriber.process_input("path/to/audio.mp3")
    
    # Example for processing a text file
    # segments = transcriber.process_input("path/to/text.txt")
    
    # Save the transcript
    # transcriber.save_transcript(segments, "path/to/output.json") 