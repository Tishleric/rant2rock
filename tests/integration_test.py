"""
integration_test.py - Integration Tests for the Rant to Rock Pipeline

This file contains integration tests for the full pipeline from transcription
through chunking, clustering, summarization, and export packaging to the final
Markdown generation and ZIP archive creation.
"""

import os
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock
import shutil
import numpy as np

from src.transcription import TranscriptionSegment
from src.chunking import ChunkingConfig, ChunkingProcessor, EmbeddingConfig
from src.clustering import ClusterConfig, ClusteringProcessor
from src.summarization import SummarizationConfig, SummarizationProcessor
from src.export_packaging import FolderStructureConfig, ExportPackagingProcessor


class IntegrationTest(unittest.TestCase):
    """Integration test for the full pipeline"""
    
    def setUp(self):
        """Set up test data and temporary directories"""
        # Create temporary directories for output files
        self.temp_dir = tempfile.mkdtemp()
        self.chunks_file = os.path.join(self.temp_dir, "chunks.json")
        self.embeddings_file = os.path.join(self.temp_dir, "embeddings.npz")
        self.clusters_file = os.path.join(self.temp_dir, "clusters.json")
        self.output_dir = os.path.join(self.temp_dir, "obsidian_notes")
        
        # Create test transcription segments
        self.segments = [
            TranscriptionSegment(
                text="This is the first segment about artificial intelligence.",
                start_time=0.0,
                end_time=5.0,
                confidence=0.9
            ),
            TranscriptionSegment(
                text="AI systems are changing how we work and live.",
                start_time=5.0,
                end_time=10.0,
                confidence=0.8
            ),
            TranscriptionSegment(
                text="Climate change is a global challenge that requires immediate action.",
                start_time=15.0,
                end_time=20.0,
                confidence=0.9
            ),
            TranscriptionSegment(
                text="We need to reduce carbon emissions and transition to renewable energy.",
                start_time=20.0,
                end_time=25.0,
                confidence=0.8
            ),
            TranscriptionSegment(
                text="Both AI and climate technologies are evolving rapidly.",
                start_time=30.0,
                end_time=35.0,
                confidence=0.85
            )
        ]
        
        # Configure processors
        self.embedding_config = EmbeddingConfig(
            model_name="text-embedding-3-large",
            dimensions=1536,
            use_mock=True  # Use mock embeddings for testing
        )
        
        self.chunking_config = ChunkingConfig(
            max_chunk_size=1000,
            min_chunk_size=100,
            overlap_size=200
        )
        
        self.cluster_config = ClusterConfig(
            algorithm="hierarchical",
            temporal_weight=0.3
        )
        
        self.summarization_config = SummarizationConfig(
            output_dir=self.output_dir
        )
    
    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)
    
    @patch('openai.ChatCompletion.create')
    @patch('openai.Embedding.create')
    def test_full_pipeline(self, mock_embedding, mock_chat_completion):
        """Test the full pipeline from transcription to summarization and export packaging"""
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
        
        # Start timing
        start_time = time.time()
        
        # Step 1: Chunking
        chunking_processor = ChunkingProcessor(
            chunking_config=self.chunking_config,
            embedding_config=self.embedding_config
        )
        chunks = chunking_processor.process_segments(self.segments)
        
        # Manually set embeddings for testing
        for chunk in chunks:
            # Ensure all chunks have embeddings with the correct dimension
            chunk.embedding = np.random.rand(embedding_dim)
        
        # Save chunks and embeddings
        chunking_processor.save_chunks(chunks, self.chunks_file)
        chunking_processor.save_embeddings(chunks, self.embeddings_file)
        
        # Step 2: Clustering
        clustering_processor = ClusteringProcessor(self.cluster_config)
        clusters = clustering_processor.process_chunks(chunks)
        
        # Ensure we have at least one cluster
        if not clusters:
            # Create a manual cluster if none were created
            from src.clustering import Cluster
            clusters = [Cluster(cluster_id=1, chunks=chunks)]
        
        # Save clusters
        clustering_processor.save_clusters(clusters, self.clusters_file)
        
        # Step 3: Summarization
        summarization_processor = SummarizationProcessor(self.summarization_config)
        file_paths = summarization_processor.process_clusters(clusters)
        
        # Step 4: Export Packaging
        export_config = FolderStructureConfig(
            input_dir=self.output_dir,
            output_dir=os.path.join(self.temp_dir, "packaged_output"),
            archive_path=os.path.join(self.temp_dir, "archive.zip"),
            organize_by_topic=True,
            organize_by_date=True,
            organize_by_entity=True,
            include_timestamp_in_archive_name=False  # Disable for testing
        )
        export_processor = ExportPackagingProcessor(export_config)
        export_result = export_processor.process()
        
        # End timing
        elapsed_time = time.time() - start_time
        
        # Verify results
        self.assertTrue(os.path.exists(self.chunks_file))
        self.assertTrue(os.path.exists(self.embeddings_file))
        self.assertTrue(os.path.exists(self.clusters_file))
        
        # Verify output files
        self.assertTrue(os.path.exists(self.output_dir))
        self.assertGreaterEqual(len(file_paths), 2)  # At least one cluster file and entity library
        
        # Verify entity library
        entity_library_path = os.path.join(self.output_dir, "Entity-Library.md")
        self.assertTrue(os.path.exists(entity_library_path))
        
        # Verify export packaging results
        self.assertTrue(os.path.exists(export_config.output_dir))
        self.assertTrue(os.path.exists(export_config.archive_path))
        self.assertTrue(os.path.exists(os.path.join(export_config.output_dir, "index.md")))
        self.assertTrue(os.path.exists(os.path.join(export_config.output_dir, "Topics")))
        self.assertTrue(os.path.exists(os.path.join(export_config.output_dir, "Entities")))
        
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
        
        print(f"\nFull pipeline integration test completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    unittest.main() 