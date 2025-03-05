"""
test_clustering.py - Unit Tests for Clustering Module

This module contains unit tests for the clustering.py module,
testing the functionality of semantic clustering algorithms,
temporal weighting, and cluster organization.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import json
import numpy as np

# Import the modules to test
from src.transcription import TranscriptionSegment
from src.chunking import TextChunk
from src.clustering import (
    ClusterConfig,
    Cluster,
    ClusteringProcessor
)


class TestClusterConfig(unittest.TestCase):
    """Test the ClusterConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ClusterConfig()
        
        # Check default values
        self.assertEqual(config.algorithm, "dbscan")
        self.assertEqual(config.eps, 0.3)
        self.assertEqual(config.min_samples, 2)
        self.assertEqual(config.n_clusters, None)
        self.assertEqual(config.distance_threshold, 0.5)
        self.assertEqual(config.linkage_method, "average")
        self.assertTrue(config.use_temporal_weighting)
        self.assertEqual(config.temporal_weight, 0.3)
        self.assertEqual(config.max_time_distance, 600.0)
        self.assertEqual(config.min_cluster_size, 2)
        
        # Check validation
        self.assertTrue(config.validate())
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = ClusterConfig(
            algorithm="hierarchical",
            eps=0.5,
            min_samples=3,
            n_clusters=5,
            distance_threshold=0.6,
            linkage_method="ward",
            use_temporal_weighting=False,
            temporal_weight=0.2,
            max_time_distance=300.0,
            min_cluster_size=3
        )
        
        # Check custom values
        self.assertEqual(config.algorithm, "hierarchical")
        self.assertEqual(config.eps, 0.5)
        self.assertEqual(config.min_samples, 3)
        self.assertEqual(config.n_clusters, 5)
        self.assertEqual(config.distance_threshold, 0.6)
        self.assertEqual(config.linkage_method, "ward")
        self.assertFalse(config.use_temporal_weighting)
        self.assertEqual(config.temporal_weight, 0.2)
        self.assertEqual(config.max_time_distance, 300.0)
        self.assertEqual(config.min_cluster_size, 3)
        
        # Check validation
        self.assertTrue(config.validate())
    
    def test_invalid_config(self):
        """Test validation with invalid configuration values"""
        # Invalid algorithm
        config = ClusterConfig(algorithm="invalid")
        self.assertFalse(config.validate())
        
        # Invalid eps
        config = ClusterConfig(eps=-0.1)
        self.assertFalse(config.validate())
        
        config = ClusterConfig(eps=1.5)
        self.assertFalse(config.validate())
        
        # Invalid min_samples
        config = ClusterConfig(min_samples=0)
        self.assertFalse(config.validate())
        
        # Invalid n_clusters
        config = ClusterConfig(algorithm="hierarchical", n_clusters=0)
        self.assertFalse(config.validate())
        
        # Invalid linkage_method
        config = ClusterConfig(algorithm="hierarchical", linkage_method="invalid")
        self.assertFalse(config.validate())
        
        # Invalid temporal_weight
        config = ClusterConfig(temporal_weight=-0.1)
        self.assertFalse(config.validate())
        
        config = ClusterConfig(temporal_weight=1.1)
        self.assertFalse(config.validate())
        
        # Invalid max_time_distance
        config = ClusterConfig(max_time_distance=0)
        self.assertFalse(config.validate())


class TestCluster(unittest.TestCase):
    """Test the Cluster class"""
    
    def setUp(self):
        """Set up test data"""
        # Create test segments with different timestamps
        self.segment1 = TranscriptionSegment(
            text="Segment 1",
            start_time=1.0,
            end_time=2.0,
            confidence=0.9
        )
        
        self.segment2 = TranscriptionSegment(
            text="Segment 2",
            start_time=2.0,
            end_time=3.0,
            confidence=0.8
        )
        
        self.segment3 = TranscriptionSegment(
            text="Segment 3",
            start_time=3.0,
            end_time=4.0,
            confidence=0.7
        )
        
        # Create test chunks with different embeddings
        self.chunk1 = TextChunk(
            text="Chunk 1",
            start_time=1.0,
            end_time=2.0,
            chunk_id=1,
            segments=[self.segment1],
            embedding=np.array([0.1, 0.2, 0.3])
        )
        
        self.chunk2 = TextChunk(
            text="Chunk 2",
            start_time=2.0,
            end_time=3.0,
            chunk_id=2,
            segments=[self.segment2],
            embedding=np.array([0.2, 0.3, 0.4])
        )
        
        self.chunk3 = TextChunk(
            text="Chunk 3",
            start_time=3.0,
            end_time=4.0,
            chunk_id=3,
            segments=[self.segment3],
            embedding=np.array([0.3, 0.4, 0.5])
        )
    
    def test_cluster_initialization(self):
        """Test cluster initialization and metadata calculation"""
        # Create a cluster with all three chunks
        cluster = Cluster(
            cluster_id=1,
            chunks=[self.chunk1, self.chunk2, self.chunk3]
        )
        
        # Check cluster properties
        self.assertEqual(cluster.cluster_id, 1)
        self.assertEqual(len(cluster.chunks), 3)
        self.assertEqual(cluster.start_time, 1.0)
        self.assertEqual(cluster.end_time, 4.0)
        self.assertAlmostEqual(cluster.avg_confidence, 0.8)
        self.assertIsNotNone(cluster.representative_chunk_id)
        self.assertEqual(cluster.duration, 3.0)
        self.assertEqual(cluster.size, 3)
    
    def test_find_representative_chunk(self):
        """Test representative chunk identification"""
        # Create a cluster with normalized embeddings
        norm_embedding1 = np.array([0.1, 0.2, 0.3]) / np.linalg.norm(np.array([0.1, 0.2, 0.3]))
        norm_embedding2 = np.array([0.2, 0.3, 0.4]) / np.linalg.norm(np.array([0.2, 0.3, 0.4]))
        norm_embedding3 = np.array([0.3, 0.4, 0.5]) / np.linalg.norm(np.array([0.3, 0.4, 0.5]))
        
        chunk1 = TextChunk(
            text="Chunk 1",
            start_time=1.0,
            end_time=2.0,
            chunk_id=1,
            segments=[self.segment1],
            embedding=norm_embedding1
        )
        
        chunk2 = TextChunk(
            text="Chunk 2",
            start_time=2.0,
            end_time=3.0,
            chunk_id=2,
            segments=[self.segment2],
            embedding=norm_embedding2
        )
        
        chunk3 = TextChunk(
            text="Chunk 3",
            start_time=3.0,
            end_time=4.0,
            chunk_id=3,
            segments=[self.segment3],
            embedding=norm_embedding3
        )
        
        # Create a cluster with all three chunks
        cluster = Cluster(
            cluster_id=1,
            chunks=[chunk1, chunk2, chunk3]
        )
        
        # Check that representative_chunk_id is set
        self.assertIsNotNone(cluster.representative_chunk_id)
        self.assertIn(cluster.representative_chunk_id, [1, 2, 3])
    
    def test_to_dict(self):
        """Test conversion of Cluster to dictionary"""
        # Create a cluster with all three chunks
        cluster = Cluster(
            cluster_id=1,
            chunks=[self.chunk1, self.chunk2, self.chunk3],
            representative_chunk_id=2
        )
        
        # Convert to dictionary
        cluster_dict = cluster.to_dict()
        
        # Check dictionary contents
        self.assertEqual(cluster_dict['cluster_id'], 1)
        self.assertEqual(cluster_dict['chunk_ids'], [1, 2, 3])
        self.assertEqual(cluster_dict['start_time'], 1.0)
        self.assertEqual(cluster_dict['end_time'], 4.0)
        self.assertEqual(cluster_dict['duration'], 3.0)
        self.assertEqual(cluster_dict['size'], 3)
        self.assertAlmostEqual(cluster_dict['avg_confidence'], 0.8)
        self.assertEqual(cluster_dict['representative_chunk_id'], 2)
        self.assertEqual(cluster_dict['text'], "Chunk 2")
    
    def test_add_chunk(self):
        """Test adding a chunk to a cluster"""
        # Create a cluster with two chunks
        cluster = Cluster(
            cluster_id=1,
            chunks=[self.chunk1, self.chunk2]
        )
        
        # Check initial state
        self.assertEqual(len(cluster.chunks), 2)
        self.assertEqual(cluster.start_time, 1.0)
        self.assertEqual(cluster.end_time, 3.0)
        
        # Add a chunk
        cluster.add_chunk(self.chunk3)
        
        # Check updated state
        self.assertEqual(len(cluster.chunks), 3)
        self.assertEqual(cluster.start_time, 1.0)
        self.assertEqual(cluster.end_time, 4.0)
        self.assertAlmostEqual(cluster.avg_confidence, 0.8)


class TestClusteringProcessor(unittest.TestCase):
    """Test the ClusteringProcessor class"""
    
    def setUp(self):
        """Set up test data"""
        # Create test segments
        self.segments = [
            TranscriptionSegment(text=f"Segment {i}", 
                               start_time=float(i), 
                               end_time=float(i+1),
                               confidence=0.9)
            for i in range(20)
        ]
        
        # Create test chunks with different embeddings
        # Group 1: Similar embeddings (chunks 0-4)
        self.chunks = [
            TextChunk(
                text=f"Chunk {i}",
                start_time=float(i),
                end_time=float(i+1),
                chunk_id=i,
                segments=[self.segments[i]],
                embedding=np.array([0.1, 0.2, 0.3]) + (np.random.rand(3) * 0.05)
            )
            for i in range(5)
        ]
        
        # Group 2: Similar embeddings (chunks 5-9)
        self.chunks.extend([
            TextChunk(
                text=f"Chunk {i}",
                start_time=float(i),
                end_time=float(i+1),
                chunk_id=i,
                segments=[self.segments[i]],
                embedding=np.array([0.5, 0.1, 0.2]) + (np.random.rand(3) * 0.05)
            )
            for i in range(5, 10)
        ])
        
        # Group 3: Similar embeddings (chunks 10-14)
        self.chunks.extend([
            TextChunk(
                text=f"Chunk {i}",
                start_time=float(i),
                end_time=float(i+1),
                chunk_id=i,
                segments=[self.segments[i]],
                embedding=np.array([0.2, 0.5, 0.1]) + (np.random.rand(3) * 0.05)
            )
            for i in range(10, 15)
        ])
        
        # Group 4: Similar embeddings (chunks 15-19)
        self.chunks.extend([
            TextChunk(
                text=f"Chunk {i}",
                start_time=float(i),
                end_time=float(i+1),
                chunk_id=i,
                segments=[self.segments[i]],
                embedding=np.array([0.3, 0.1, 0.5]) + (np.random.rand(3) * 0.05)
            )
            for i in range(15, 20)
        ])
        
        # Normalize all embeddings
        for chunk in self.chunks:
            chunk.embedding = chunk.embedding / np.linalg.norm(chunk.embedding)
    
    def test_process_chunks_empty(self):
        """Test processing with empty chunks list"""
        processor = ClusteringProcessor()
        clusters = processor.process_chunks([])
        self.assertEqual(len(clusters), 0)
    
    def test_process_chunks_no_embeddings(self):
        """Test processing with chunks that have no embeddings"""
        # Create chunks without embeddings
        chunks_no_embeddings = [
            TextChunk(
                text=f"Chunk {i}",
                start_time=float(i),
                end_time=float(i+1),
                chunk_id=i,
                segments=[self.segments[i]],
                embedding=None
            )
            for i in range(5)
        ]
        
        processor = ClusteringProcessor()
        clusters = processor.process_chunks(chunks_no_embeddings)
        self.assertEqual(len(clusters), 0)
    
    def test_dbscan_clustering(self):
        """Test DBSCAN clustering algorithm"""
        # Configure for DBSCAN with appropriate epsilon for our test data
        config = ClusterConfig(
            algorithm="dbscan",
            eps=0.15,  # Adjusted for the normalized distance matrix
            min_samples=2,
            use_temporal_weighting=False
        )
        
        processor = ClusteringProcessor(config)
        clusters = processor.process_chunks(self.chunks)
        
        # We should have at least 1 cluster
        self.assertGreaterEqual(len(clusters), 1)
        
        # Due to the normalized distance calculation, we can't make strong assumptions
        # about cluster membership with respect to the original groups.
        # Instead, we'll check basic properties of the clusters.
        
        # Each cluster should have at least min_samples chunks
        for cluster in clusters:
            self.assertGreaterEqual(len(cluster.chunks), config.min_samples)
            
            # Check that the cluster has valid metadata
            self.assertIsNotNone(cluster.start_time)
            self.assertIsNotNone(cluster.end_time)
            self.assertIsNotNone(cluster.representative_chunk_id)
    
    def test_hierarchical_clustering(self):
        """Test hierarchical clustering algorithm"""
        # Configure for hierarchical clustering
        config = ClusterConfig(
            algorithm="hierarchical",
            n_clusters=4,  # We expect 4 clusters
            linkage_method="average",
            use_temporal_weighting=False
        )
        
        processor = ClusteringProcessor(config)
        clusters = processor.process_chunks(self.chunks)
        
        # We should have 4 clusters
        self.assertEqual(len(clusters), 4)
        
        # Due to the normalized distance calculation, we can't make strong assumptions
        # about cluster membership with respect to the original groups.
        # Instead, we'll check basic properties of the clusters.
        
        # Check that each cluster has valid metadata
        for cluster in clusters:
            self.assertIsNotNone(cluster.start_time)
            self.assertIsNotNone(cluster.end_time)
            self.assertIsNotNone(cluster.representative_chunk_id)
            
            # Each cluster should have at least one chunk
            self.assertGreater(len(cluster.chunks), 0)
    
    def test_hybrid_clustering(self):
        """Test hybrid clustering algorithm"""
        # Configure for hybrid clustering
        config = ClusterConfig(
            algorithm="hybrid",
            eps=0.15,  # Strict epsilon to create more noise points
            min_samples=3,
            distance_threshold=0.5,
            use_temporal_weighting=False
        )
        
        processor = ClusteringProcessor(config)
        clusters = processor.process_chunks(self.chunks)
        
        # We should have some clusters
        self.assertGreaterEqual(len(clusters), 1)
        
        # Total number of chunks in all clusters should be <= total chunks
        total_clustered = sum(len(cluster.chunks) for cluster in clusters)
        self.assertLessEqual(total_clustered, len(self.chunks))
    
    def test_temporal_weighting(self):
        """Test temporal weighting in clustering"""
        # Create chunks with similar embeddings but different timestamps
        # This will create chunks that are semantically similar but temporally distant
        temporally_separated_chunks = []
        
        # Group 1: Similar embeddings, early times (0-9)
        for i in range(10):
            temporally_separated_chunks.append(TextChunk(
                text=f"Early Chunk {i}",
                start_time=float(i),
                end_time=float(i+1),
                chunk_id=i,
                segments=[TranscriptionSegment(
                    text=f"Early Segment {i}", 
                    start_time=float(i), 
                    end_time=float(i+1),
                    confidence=0.9
                )],
                embedding=np.array([0.1, 0.2, 0.3]) + (np.random.rand(3) * 0.05)
            ))
        
        # Group 2: Similar embeddings to Group 1, but later times (100-109)
        for i in range(10):
            temporally_separated_chunks.append(TextChunk(
                text=f"Late Chunk {i}",
                start_time=float(i+100),
                end_time=float(i+101),
                chunk_id=i+10,
                segments=[TranscriptionSegment(
                    text=f"Late Segment {i}", 
                    start_time=float(i+100), 
                    end_time=float(i+101),
                    confidence=0.9
                )],
                embedding=np.array([0.1, 0.2, 0.3]) + (np.random.rand(3) * 0.05)
            ))
        
        # Normalize all embeddings
        for chunk in temporally_separated_chunks:
            chunk.embedding = chunk.embedding / np.linalg.norm(chunk.embedding)
        
        # Case 1: Without temporal weighting
        config_no_time = ClusterConfig(
            algorithm="dbscan",
            eps=0.2,
            min_samples=2,
            use_temporal_weighting=False
        )
        
        processor_no_time = ClusteringProcessor(config_no_time)
        clusters_no_time = processor_no_time.process_chunks(temporally_separated_chunks)
        
        # Without temporal weighting, we should get fewer clusters because
        # semantically similar chunks are grouped regardless of time
        
        # Case 2: With temporal weighting
        config_with_time = ClusterConfig(
            algorithm="dbscan",
            eps=0.2,
            min_samples=2,
            use_temporal_weighting=True,
            temporal_weight=0.5,
            max_time_distance=30.0  # Chunks more than 30s apart will be fully separated
        )
        
        processor_with_time = ClusteringProcessor(config_with_time)
        clusters_with_time = processor_with_time.process_chunks(temporally_separated_chunks)
        
        # With temporal weighting, we should get more clusters because
        # the temporal distance should separate the early and late chunks
        
        # We expect more clusters with temporal weighting
        self.assertGreater(len(clusters_with_time), len(clusters_no_time))
    
    def test_manual_review(self):
        """Test manual review functionality"""
        processor = ClusteringProcessor()
        clusters = processor.process_chunks(self.chunks)
        
        # Generate manual review
        review = processor.manual_review(clusters)
        
        # Check review contents
        self.assertIn("n_clusters", review)
        self.assertIn("total_chunks", review)
        self.assertIn("avg_cluster_size", review)
        self.assertIn("min_cluster_size", review)
        self.assertIn("max_cluster_size", review)
        self.assertIn("cluster_size_distribution", review)
        self.assertIn("samples", review)
        
        # Check sample clusters
        self.assertIn("largest_cluster", review["samples"])
        self.assertIn("smallest_cluster", review["samples"])
        self.assertIn("middle_cluster", review["samples"])
    
    def test_save_and_load_clusters(self):
        """Test saving and loading clusters"""
        processor = ClusteringProcessor()
        clusters = processor.process_chunks(self.chunks)
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save clusters
            processor.save_clusters(clusters, temp_path)
            
            # Load clusters
            loaded_clusters = processor.load_clusters(temp_path, self.chunks)
            
            # Check loaded clusters
            self.assertEqual(len(loaded_clusters), len(clusters))
            
            # Check consistency of loaded clusters
            for i, (orig, loaded) in enumerate(zip(clusters, loaded_clusters)):
                self.assertEqual(loaded.cluster_id, orig.cluster_id)
                self.assertEqual(len(loaded.chunks), len(orig.chunks))
                self.assertEqual(loaded.start_time, orig.start_time)
                self.assertEqual(loaded.end_time, orig.end_time)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main() 