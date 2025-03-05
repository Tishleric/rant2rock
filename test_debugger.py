#!/usr/bin/env python3
"""
Test debugger for chunking module

This script runs a minimal test of the chunking module to verify API key detection
and mock mode functionality without hanging or freezing.
"""

import os
import sys
import logging
import unittest
from unittest.mock import patch
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("test_debugger")

def run_minimal_test():
    """Run minimal test of chunking module"""
    logger.info("====== Starting chunking module test ======")
    
    # Check if API key is present
    api_key = os.environ.get("OPENAI_API_KEY")
    logger.info(f"OPENAI_API_KEY present: {bool(api_key)}")
    
    # Import relevant modules
    from src.transcription import TranscriptionSegment
    from src.chunking import (
        TextChunk,
        ChunkingConfig,
        EmbeddingConfig,
        ChunkingProcessor,
        ChunkingUtils
    )
    
    # Create a simple segment for testing
    logger.info("Creating test segment")
    segment = TranscriptionSegment(
        text="This is a very short test segment for debugging.",
        start_time=0.0,
        end_time=1.0,
        confidence=0.9
    )
    
    # Create processor with default config
    logger.info("Creating ChunkingProcessor instance")
    processor = ChunkingProcessor(
        chunking_config=ChunkingConfig(max_chunk_size=200),
        embedding_config=EmbeddingConfig(model_name="text-embedding-3-large")
    )
    
    # Log the detected mode
    logger.info(f"Detected use_mock mode: {processor.embedding_config.use_mock}")
    
    # Process the segment
    logger.info("Processing segments")
    try:
        chunks = processor.process_segments([segment])
        logger.info(f"Successfully processed {len(chunks)} chunks")
        
        # Check chunks
        if chunks:
            embedding = chunks[0].embedding
            if embedding is not None:
                logger.info(f"Embedding shape: {embedding.shape}, type: {type(embedding)}")
                if np.all(embedding == 0):
                    logger.info("Embedding contains all zeros (using zero mock vectors)")
                elif np.isclose(np.linalg.norm(embedding), 1.0):
                    logger.info("Embedding is normalized (unit vector)")
                else:
                    logger.info(f"Embedding norm: {np.linalg.norm(embedding)}")
            else:
                logger.warning("Embedding is None")
    except Exception as e:
        logger.error(f"Error processing segments: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("====== Completed chunking module test ======")
    return True

if __name__ == "__main__":
    success = run_minimal_test()
    logger.info(f"Test {'succeeded' if success else 'failed'}") 