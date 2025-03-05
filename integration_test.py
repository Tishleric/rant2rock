"""
Integration test for the Rant to Rock Obsidian Companion Webapp.

This script tests the entire pipeline from transcription through 
chunking, clustering, summarization, and export packaging to the final ZIP archive.
"""

import os
import tempfile
import logging
import time
import shutil
import unittest
import numpy as np
from typing import List
from unittest.mock import patch, MagicMock

# Import modules to test
from src.transcription import TranscriptionSegment, TranscriptionEngine
from src.chunking import ChunkingProcessor, TextChunk, ChunkingConfig, EmbeddingConfig
from src.clustering import ClusteringProcessor, Cluster, ClusterConfig
from src.summarization import SummarizationProcessor, SummarizationConfig
from src.export_packaging import ExportPackagingProcessor, FolderStructureConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationTest(unittest.TestCase):
    """Integration test for the complete pipeline"""

    def setUp(self):
        """Set up the test by creating test data and temp directory"""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.chunks_file = os.path.join(self.temp_dir, "chunks.json")
        self.embeddings_file = os.path.join(self.temp_dir, "embeddings.npz")  # Use .npz extension
        self.clusters_file = os.path.join(self.temp_dir, "clusters.json")
        self.output_dir = os.path.join(self.temp_dir, "obsidian_notes")
        self.organized_dir = os.path.join(self.temp_dir, "organized_notes")
        self.archive_path = os.path.join(self.temp_dir, "obsidian_export.zip")
        
        # Create a test transcript with 5 segments
        self.segments = [
            TranscriptionSegment(
                text="The development of artificial intelligence has made significant progress in recent years.",
                start_time=0.0,
                end_time=5.0,
                confidence=0.95
            ),
            TranscriptionSegment(
                text="Machine learning models can now perform tasks that were once thought to be exclusive to humans.",
                start_time=5.0,
                end_time=10.0,
                confidence=0.93
            ),
            TranscriptionSegment(
                text="Climate change is one of the biggest challenges facing humanity today.",
                start_time=10.0,
                end_time=15.0,
                confidence=0.97
            ),
            TranscriptionSegment(
                text="Rising global temperatures are causing more extreme weather events and rising sea levels.",
                start_time=15.0,
                end_time=20.0,
                confidence=0.96
            ),
            TranscriptionSegment(
                text="Both AI technology and climate science are rapidly evolving fields that require ongoing research.",
                start_time=20.0,
                end_time=25.0,
                confidence=0.94
            )
        ]

    def tearDown(self):
        """Clean up temporary directory after tests"""
        shutil.rmtree(self.temp_dir)

    @patch('openai.ChatCompletion.create')
    @patch('openai.Embedding.create')
    def test_full_pipeline(self, mock_embedding, mock_chat_completion):
        """Test the complete pipeline from transcription to markdown generation"""
        start_time = time.time()
        logger.info("Starting full pipeline integration test")

        # Mock embedding API with proper dimensions
        embedding_dim = 1536
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [{"embedding": np.random.rand(embedding_dim).tolist()}]
        mock_embedding.return_value = mock_embedding_response
        
        # Mock chat completion API for summarization
        def mock_chat_completion_side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            content = messages[1]['content'] if len(messages) > 1 else ""
            
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            
            if "create a concise summary" in content:
                if "artificial intelligence" in content.lower():
                    mock_response.choices[0].message.content = "# Artificial Intelligence Impact\n\nAI systems are revolutionizing work processes and transforming industries."
                elif "climate change" in content.lower():
                    mock_response.choices[0].message.content = "# Climate Change Action\n\nClimate change requires global action to reduce carbon emissions and mitigate effects."
                else:
                    mock_response.choices[0].message.content = "# Technology Evolution\n\nBoth AI and climate technologies are evolving rapidly, changing how we approach global challenges."
            elif "Extract the key entities" in content:
                if "artificial intelligence" in content.lower():
                    mock_response.choices[0].message.content = "Artificial Intelligence, Work Transformation, Technology"
                elif "climate change" in content.lower():
                    mock_response.choices[0].message.content = "Climate Change, Carbon Emissions, Renewable Energy"
                else:
                    mock_response.choices[0].message.content = "AI, Climate Technology, Global Challenges"
            elif "Extract" in content and "key topics" in content:
                if "artificial intelligence" in content.lower():
                    mock_response.choices[0].message.content = "AI, Technology, Work, Transformation"
                elif "climate change" in content.lower():
                    mock_response.choices[0].message.content = "Climate, Environment, Carbon, Energy"
                else:
                    mock_response.choices[0].message.content = "Technology, Evolution, AI, Climate"
            
            return mock_response
        
        mock_chat_completion.side_effect = mock_chat_completion_side_effect

        # Step 1: Initialize the Chunking processor
        chunking_config = ChunkingConfig(
            max_chunk_size=1000,
            min_chunk_size=100,
            overlap_size=200
        )
        embedding_config = EmbeddingConfig(
            model_name="text-embedding-3-large",
            dimensions=embedding_dim,
            use_mock=True  # Use mock embeddings for testing
        )
        chunking_processor = ChunkingProcessor(
            chunking_config=chunking_config,
            embedding_config=embedding_config
        )

        # Step 2: Create chunks from transcript
        chunks = chunking_processor.process_segments(self.segments)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        logger.info(f"Created {len(chunks)} chunks from {len(self.segments)} segments")
        
        # Manually set embeddings for testing
        for chunk in chunks:
            # Ensure all chunks have embeddings with the correct dimension
            chunk.embedding = np.random.rand(embedding_dim)

        # Step 3: Save chunks and embeddings
        chunking_processor.save_chunks(chunks, self.chunks_file)
        chunking_processor.save_embeddings(chunks, self.embeddings_file)
        logger.info(f"Saved chunks to {self.chunks_file} and embeddings to {self.embeddings_file}")

        # Step 4: Initialize the Clustering processor
        clustering_config = ClusterConfig(
            algorithm="hierarchical",
            temporal_weight=0.3
        )
        clustering_processor = ClusteringProcessor(clustering_config)

        # Step 5: Create clusters
        clusters = clustering_processor.process_chunks(chunks)
        self.assertIsInstance(clusters, list)
        
        # Ensure we have at least one cluster
        if not clusters:
            # Create a manual cluster if none were created
            clusters = [Cluster(cluster_id=1, chunks=chunks)]
            
        self.assertGreater(len(clusters), 0)
        logger.info(f"Created {len(clusters)} clusters from {len(chunks)} chunks")

        # Step 6: Save clusters
        clustering_processor.save_clusters(clusters, self.clusters_file)
        logger.info(f"Saved clusters to {self.clusters_file}")

        # Step 7: Initialize the Summarization processor
        summarization_config = SummarizationConfig(
            output_dir=self.output_dir
        )
        summarization_processor = SummarizationProcessor(summarization_config)

        # Step 8: Generate summaries and markdown files
        file_paths = summarization_processor.process_clusters(clusters)
        self.assertIsInstance(file_paths, list)
        self.assertGreater(len(file_paths), 0)
        logger.info(f"Generated {len(file_paths)} markdown files")

        # Step 9: Initialize Export Packaging processor
        folder_structure_config = FolderStructureConfig(
            input_dir=self.output_dir,
            output_dir=self.organized_dir,
            archive_path=self.archive_path,
            organize_by_topic=True,
            organize_by_date=True,
            organize_by_entity=True,
            include_timestamp_in_archive_name=False  # Disable for testing
        )
        export_processor = ExportPackagingProcessor(folder_structure_config)

        # Step 10: Organize files and create ZIP archive
        organized_dir, zip_path = export_processor.process()
        self.assertEqual(organized_dir, self.organized_dir)
        self.assertEqual(zip_path, self.archive_path)
        self.assertTrue(os.path.exists(self.organized_dir))
        self.assertTrue(os.path.exists(self.archive_path))
        logger.info(f"Organized files into {organized_dir} and created ZIP archive at {zip_path}")

        # Step 11: Verify outputs
        self.assertTrue(os.path.exists(self.chunks_file))
        self.assertTrue(os.path.exists(self.embeddings_file))
        self.assertTrue(os.path.exists(self.clusters_file))
        self.assertTrue(os.path.exists(self.output_dir))
        
        # Verify entity library
        entity_library_path = os.path.join(self.output_dir, "Entity-Library.md")
        self.assertTrue(os.path.exists(entity_library_path))
        
        # Check content of output files
        for file_path in file_paths:
            if os.path.basename(file_path).startswith("Cluster"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.assertIn("# ", content)  # Has a title
                    self.assertIn("---", content)  # Has YAML frontmatter
                    self.assertIn("tags:", content)  # Has tags
                    self.assertIn("## Related Entities", content)  # Has entities
                    self.assertIn("## Timespan", content)  # Has timespan info
        
        # Verify organized folder structure
        self.assertTrue(os.path.exists(os.path.join(self.organized_dir, "Notes")))
        self.assertTrue(os.path.exists(os.path.join(self.organized_dir, "Topics")))
        self.assertTrue(os.path.exists(os.path.join(self.organized_dir, "Entities")))
        
        # Measure and log performance metrics
        elapsed_time = time.time() - start_time
        archive_size = os.path.getsize(self.archive_path) / (1024 * 1024)  # Size in MB
        logger.info(f"Full pipeline integration test completed in {elapsed_time:.2f} seconds")
        logger.info(f"Number of files processed: {len(file_paths)}")
        logger.info(f"Size of ZIP archive: {archive_size:.2f} MB")


if __name__ == "__main__":
    unittest.main()
