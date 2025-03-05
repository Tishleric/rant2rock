"""
clustering.py - Semantic Clustering & Topic Grouping Module

This module handles:
1. Clustering of text chunks based on semantic similarity (embeddings)
2. Incorporation of temporal metadata to refine clusters
3. Topic identification and grouping
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
from collections import defaultdict

# Clustering algorithms
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# Local imports
from src.chunking import TextChunk

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """Configuration for clustering algorithm"""
    # Algorithm selection
    algorithm: str = "dbscan"  # Options: "dbscan", "hierarchical", "hybrid"
    
    # DBSCAN parameters
    eps: float = 0.3  # Maximum distance between samples for DBSCAN (reduced from 0.4 to account for normalized distances)
    min_samples: int = 2  # Minimum number of samples in a DBSCAN cluster
    
    # Hierarchical clustering parameters
    n_clusters: Optional[int] = None  # Number of clusters for hierarchical clustering
    distance_threshold: Optional[float] = 0.5  # Distance threshold for hierarchical clustering (adjusted from 0.7)
    linkage_method: str = "average"  # Options: "single", "complete", "average", "ward"
    
    # Temporal weighting parameters
    use_temporal_weighting: bool = True  # Whether to incorporate temporal information
    temporal_weight: float = 0.3  # Weight of temporal similarity (0.0 to 1.0)
    max_time_distance: float = 600.0  # Maximum time distance to consider (in seconds)
    
    # Output parameters
    min_cluster_size: int = 2  # Minimum size of a cluster to keep
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        # Check algorithm is valid
        valid_algorithms = ["dbscan", "hierarchical", "hybrid"]
        if self.algorithm not in valid_algorithms:
            logger.error(f"Invalid algorithm: {self.algorithm}. Must be one of {valid_algorithms}")
            return False
        
        # Check DBSCAN parameters
        if self.eps <= 0 or self.eps > 1:
            logger.error(f"Invalid eps value: {self.eps}. Must be between 0 and 1")
            return False
        
        if self.min_samples < 1:
            logger.error(f"Invalid min_samples: {self.min_samples}. Must be at least 1")
            return False
        
        # Check hierarchical clustering parameters
        if self.algorithm in ["hierarchical", "hybrid"]:
            if self.n_clusters is not None and self.n_clusters < 1:
                logger.error(f"Invalid n_clusters: {self.n_clusters}. Must be at least 1")
                return False
            
            valid_linkage_methods = ["single", "complete", "average", "ward"]
            if self.linkage_method not in valid_linkage_methods:
                logger.error(f"Invalid linkage_method: {self.linkage_method}. Must be one of {valid_linkage_methods}")
                return False
        
        # Check temporal weighting parameters
        if self.temporal_weight < 0 or self.temporal_weight > 1:
            logger.error(f"Invalid temporal_weight: {self.temporal_weight}. Must be between 0 and 1")
            return False
        
        if self.max_time_distance <= 0:
            logger.error(f"Invalid max_time_distance: {self.max_time_distance}. Must be greater than 0")
            return False
        
        return True


@dataclass
class Cluster:
    """Data class to store a cluster of text chunks with metadata"""
    cluster_id: int
    chunks: List[TextChunk]
    start_time: float = field(init=False)
    end_time: float = field(init=False)
    avg_confidence: float = field(init=False)
    representative_chunk_id: Optional[int] = None
    
    def __post_init__(self):
        """Calculate aggregate metadata after initialization"""
        # Set start_time to the earliest start_time of any chunk
        self.start_time = min(chunk.start_time for chunk in self.chunks) if self.chunks else 0
        
        # Set end_time to the latest end_time of any chunk
        self.end_time = max(chunk.end_time for chunk in self.chunks) if self.chunks else 0
        
        # Calculate average confidence across all segments in all chunks
        all_confidences = []
        for chunk in self.chunks:
            for segment in chunk.segments:
                if segment.confidence is not None:
                    all_confidences.append(segment.confidence)
        
        self.avg_confidence = np.mean(all_confidences) if all_confidences else None
        
        # Find most representative chunk (closest to centroid) if not provided
        if self.representative_chunk_id is None and self.chunks:
            self._find_representative_chunk()
    
    def _find_representative_chunk(self):
        """Find the chunk closest to the centroid of the cluster"""
        if not self.chunks or len(self.chunks) == 1:
            self.representative_chunk_id = self.chunks[0].chunk_id if self.chunks else None
            return
        
        # Collect all embeddings
        embeddings = np.vstack([chunk.embedding for chunk in self.chunks if chunk.embedding is not None])
        
        if len(embeddings) == 0:
            # No embeddings available, use the first chunk
            self.representative_chunk_id = self.chunks[0].chunk_id
            return
        
        # Calculate centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Find chunk closest to centroid
        min_distance = float('inf')
        closest_chunk_id = None
        
        for chunk in self.chunks:
            if chunk.embedding is not None:
                # Use cosine distance (1 - cosine similarity)
                distance = 1 - cosine_similarity([chunk.embedding], [centroid])[0][0]
                if distance < min_distance:
                    min_distance = distance
                    closest_chunk_id = chunk.chunk_id
        
        self.representative_chunk_id = closest_chunk_id if closest_chunk_id is not None else self.chunks[0].chunk_id
    
    @property
    def duration(self) -> float:
        """Return the duration of the cluster in seconds"""
        return self.end_time - self.start_time
    
    @property
    def size(self) -> int:
        """Return the number of chunks in the cluster"""
        return len(self.chunks)
    
    @property
    def text(self) -> str:
        """Return the text of the representative chunk"""
        for chunk in self.chunks:
            if chunk.chunk_id == self.representative_chunk_id:
                return chunk.text
        return "" if not self.chunks else self.chunks[0].text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster to dictionary"""
        return {
            'cluster_id': self.cluster_id,
            'chunk_ids': [chunk.chunk_id for chunk in self.chunks],
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'size': self.size,
            'avg_confidence': self.avg_confidence,
            'representative_chunk_id': self.representative_chunk_id,
            'text': self.text
        }
    
    def add_chunk(self, chunk: TextChunk) -> None:
        """Add a chunk to the cluster and recalculate metadata"""
        self.chunks.append(chunk)
        self.__post_init__()  # Recalculate metadata


class ClusteringProcessor:
    """Process text chunks and group them into semantic clusters with temporal weighting"""
    
    def __init__(self, config: Optional[ClusterConfig] = None):
        """Initialize the clustering processor with configuration"""
        self.config = config if config is not None else ClusterConfig()
        
        if not self.config.validate():
            raise ValueError("Invalid clustering configuration")
            
        logger.info(f"Initialized ClusteringProcessor with algorithm: {self.config.algorithm}")
    
    def process_chunks(self, chunks: List[TextChunk]) -> List[Cluster]:
        """
        Process chunks to create topic-based clusters
        
        Args:
            chunks: List of TextChunk objects with embeddings
            
        Returns:
            List of Cluster objects
        """
        if not chunks:
            logger.warning("No chunks provided for clustering")
            return []
        
        # Ensure all chunks have embeddings
        chunks_with_embeddings = [chunk for chunk in chunks if chunk.embedding is not None]
        if len(chunks_with_embeddings) < len(chunks):
            logger.warning(f"Skipping {len(chunks) - len(chunks_with_embeddings)} chunks without embeddings")
        
        if not chunks_with_embeddings:
            logger.error("No chunks with embeddings found")
            return []
        
        # Create embeddings matrix
        embeddings = np.vstack([chunk.embedding for chunk in chunks_with_embeddings])
        
        # Normalize embeddings if they aren't already
        if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
            logger.info("Normalizing embeddings")
            embeddings = normalize(embeddings)
        
        # Apply temporal weighting if configured
        distance_matrix = None
        if self.config.use_temporal_weighting:
            distance_matrix = self._create_weighted_distance_matrix(chunks_with_embeddings, embeddings)
        
        # Perform clustering based on the selected algorithm
        labels = self._perform_clustering(chunks_with_embeddings, embeddings, distance_matrix)
        
        # Group chunks by cluster label
        return self._create_clusters(chunks_with_embeddings, labels)
    
    def _create_weighted_distance_matrix(self, chunks: List[TextChunk], embeddings: np.ndarray) -> np.ndarray:
        """
        Create a distance matrix that incorporates both semantic and temporal distance
        
        Args:
            chunks: List of TextChunk objects
            embeddings: Matrix of embeddings (n_chunks, embedding_dim)
            
        Returns:
            Weighted distance matrix
        """
        # Create semantic distance matrix (1 - cosine similarity)
        cosine_sim = cosine_similarity(embeddings)
        # Ensure cosine similarity is in range [-1, 1]
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        # Convert to distance (range [0, 2])
        semantic_dist = 1 - cosine_sim
        # Ensure all distances are non-negative (normalizing to [0, 1])
        semantic_dist = semantic_dist / 2.0
        
        # Create temporal distance matrix
        n_chunks = len(chunks)
        temporal_dist = np.zeros((n_chunks, n_chunks))
        
        for i in range(n_chunks):
            for j in range(i + 1, n_chunks):
                # Calculate midpoint times for each chunk
                time_i = (chunks[i].start_time + chunks[i].end_time) / 2
                time_j = (chunks[j].start_time + chunks[j].end_time) / 2
                
                # Calculate absolute time difference
                time_diff = abs(time_i - time_j)
                
                # Normalize time difference to [0, 1] using max_time_distance
                norm_time_diff = min(time_diff / self.config.max_time_distance, 1.0)
                
                # Store in matrix (symmetric)
                temporal_dist[i, j] = norm_time_diff
                temporal_dist[j, i] = norm_time_diff
        
        # Combine distances using temporal_weight
        w = self.config.temporal_weight
        weighted_dist = (1 - w) * semantic_dist + w * temporal_dist
        
        # Ensure diagonal is explicitly set to zero
        np.fill_diagonal(weighted_dist, 0.0)
        
        # Validate the distance matrix
        if np.any(weighted_dist < 0):
            logger.warning("Negative values found in distance matrix, adjusting to non-negative")
            weighted_dist = np.abs(weighted_dist)
            
        if not np.allclose(np.diagonal(weighted_dist), 0.0):
            logger.warning("Non-zero diagonal found in distance matrix, forcing diagonal to zero")
            np.fill_diagonal(weighted_dist, 0.0)
        
        logger.info(f"Created weighted distance matrix with temporal weight: {w}")
        logger.info(f"Distance matrix min: {np.min(weighted_dist)}, max: {np.max(weighted_dist)}")
        return weighted_dist
    
    def _perform_clustering(self, chunks: List[TextChunk], embeddings: np.ndarray, 
                           distance_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform clustering using the selected algorithm
        
        Args:
            chunks: List of TextChunk objects
            embeddings: Matrix of embeddings (n_chunks, embedding_dim)
            distance_matrix: Optional precomputed distance matrix
            
        Returns:
            Array of cluster labels for each chunk
        """
        if self.config.algorithm == "dbscan":
            return self._dbscan_clustering(distance_matrix if distance_matrix is not None else embeddings)
        elif self.config.algorithm == "hierarchical":
            return self._hierarchical_clustering(distance_matrix if distance_matrix is not None else embeddings)
        elif self.config.algorithm == "hybrid":
            return self._hybrid_clustering(chunks, embeddings, distance_matrix)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.config.algorithm}")
    
    def _dbscan_clustering(self, data: np.ndarray) -> np.ndarray:
        """
        Perform DBSCAN clustering
        
        Args:
            data: Embeddings matrix or distance matrix
            
        Returns:
            Array of cluster labels
        """
        # Check if we have only one sample
        if data.shape[0] <= 1:
            logger.info("Only one sample provided for DBSCAN clustering, returning single cluster")
            return np.zeros(data.shape[0], dtype=int)
            
        # Check if data is a distance matrix
        is_distance_matrix = (data.shape[0] == data.shape[1])
        
        if is_distance_matrix:
            # Ensure distance matrix is valid for DBSCAN
            if np.any(data < 0):
                logger.warning("Negative values in distance matrix detected. Converting to absolute values.")
                data = np.abs(data)
            
            # Ensure the diagonal is zero
            np.fill_diagonal(data, 0.0)
            
            # Use precomputed distance matrix
            dbscan = DBSCAN(
                eps=self.config.eps,
                min_samples=self.config.min_samples,
                metric='precomputed'
            )
            labels = dbscan.fit_predict(data)
        else:
            # Use embeddings directly
            dbscan = DBSCAN(
                eps=self.config.eps,
                min_samples=self.config.min_samples,
                metric='cosine'
            )
            labels = dbscan.fit_predict(data)
        
        logger.info(f"DBSCAN clustering complete. Found {len(np.unique(labels[labels >= 0]))} clusters")
        return labels
    
    def _hierarchical_clustering(self, data: np.ndarray) -> np.ndarray:
        """
        Perform hierarchical clustering
        
        Args:
            data: Embeddings matrix or distance matrix
            
        Returns:
            Array of cluster labels
        """
        # Check if we have only one sample
        if data.shape[0] <= 1:
            logger.info("Only one sample provided for hierarchical clustering, returning single cluster")
            return np.zeros(data.shape[0], dtype=int)
            
        # Check if data is a distance matrix
        is_distance_matrix = (data.shape[0] == data.shape[1])
        
        if is_distance_matrix:
            # Ensure distance matrix is valid
            if np.any(data < 0):
                logger.warning("Negative values in distance matrix detected. Converting to absolute values.")
                data = np.abs(data)
            
            # Ensure the diagonal is zero
            np.fill_diagonal(data, 0.0)
            
            # Convert distance matrix to condensed form for scipy
            condensed_dist = squareform(data)
        else:
            # Calculate pairwise distances
            condensed_dist = pdist(data, metric='cosine')
        
        # Perform hierarchical clustering
        Z = linkage(condensed_dist, method=self.config.linkage_method)
        
        # Cut the dendrogram to get clusters
        if self.config.n_clusters is not None:
            labels = fcluster(Z, self.config.n_clusters, criterion='maxclust') - 1
        else:
            labels = fcluster(Z, self.config.distance_threshold, criterion='distance') - 1
        
        logger.info(f"Hierarchical clustering complete. Found {len(np.unique(labels))} clusters")
        return labels
    
    def _hybrid_clustering(self, chunks: List[TextChunk], embeddings: np.ndarray,
                          distance_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform hybrid clustering (DBSCAN followed by hierarchical clustering of outliers)
        
        Args:
            chunks: List of TextChunk objects
            embeddings: Matrix of embeddings
            distance_matrix: Optional precomputed distance matrix
            
        Returns:
            Array of cluster labels
        """
        # Check if we have only one sample
        if len(chunks) <= 1:
            logger.info("Only one sample provided for hybrid clustering, returning single cluster")
            return np.zeros(len(chunks), dtype=int)
            
        # First, apply DBSCAN
        dbscan_labels = self._dbscan_clustering(distance_matrix if distance_matrix is not None else embeddings)
        
        # Identify noise points (label -1)
        noise_indices = np.where(dbscan_labels == -1)[0]
        
        if len(noise_indices) == 0:
            # No noise points, just return DBSCAN results
            return dbscan_labels
        
        # Extract noise points
        noise_chunks = [chunks[i] for i in noise_indices]
        noise_embeddings = embeddings[noise_indices]
        
        # Create a distance matrix for noise points if needed
        noise_distance_matrix = None
        if distance_matrix is not None:
            noise_distance_matrix = distance_matrix[np.ix_(noise_indices, noise_indices)]
        
        # Apply hierarchical clustering to noise points
        if len(noise_indices) > 1:
            # We need at least 2 points for hierarchical clustering
            hierarchical_labels = self._hierarchical_clustering(
                noise_distance_matrix if noise_distance_matrix is not None else noise_embeddings
            )
            
            # Offset hierarchical labels to avoid conflicts with DBSCAN labels
            max_dbscan_label = np.max(dbscan_labels) if np.max(dbscan_labels) >= 0 else -1
            hierarchical_labels = hierarchical_labels + max_dbscan_label + 1
            
            # Merge labels
            merged_labels = dbscan_labels.copy()
            merged_labels[noise_indices] = hierarchical_labels
        else:
            # Only one noise point, assign it to a new cluster
            merged_labels = dbscan_labels.copy()
            merged_labels[noise_indices] = np.max(dbscan_labels) + 1 if np.max(dbscan_labels) >= 0 else 0
        
        logger.info(f"Hybrid clustering complete. Found {len(np.unique(merged_labels[merged_labels >= 0]))} clusters")
        return merged_labels
    
    def _create_clusters(self, chunks: List[TextChunk], labels: np.ndarray) -> List[Cluster]:
        """
        Create Cluster objects from clustering results
        
        Args:
            chunks: List of TextChunk objects
            labels: Array of cluster labels
            
        Returns:
            List of Cluster objects
        """
        # Group chunks by label
        cluster_map = defaultdict(list)
        for i, label in enumerate(labels):
            # Skip noise points (label -1)
            if label >= 0:
                cluster_map[label].append(chunks[i])
        
        # Filter out clusters that are too small
        cluster_map = {label: chunks for label, chunks in cluster_map.items() 
                      if len(chunks) >= self.config.min_cluster_size}
        
        # Create Cluster objects
        clusters = [Cluster(cluster_id=i, chunks=chunks) 
                   for i, (_, chunks) in enumerate(sorted(cluster_map.items()))]
        
        logger.info(f"Created {len(clusters)} clusters")
        return clusters
    
    def save_clusters(self, clusters: List[Cluster], output_path: str) -> None:
        """
        Save clusters to a JSON file
        
        Args:
            clusters: List of Cluster objects
            output_path: Path to save the JSON file
        """
        with open(output_path, 'w') as f:
            json.dump({
                'clusters': [cluster.to_dict() for cluster in clusters],
                'config': {k: v for k, v in vars(self.config).items()
                          if not k.startswith('_')}
            }, f, indent=2)
        
        logger.info(f"Saved {len(clusters)} clusters to {output_path}")
    
    def load_clusters(self, input_path: str, chunks: List[TextChunk]) -> List[Cluster]:
        """
        Load clusters from a JSON file
        
        Args:
            input_path: Path to the JSON file
            chunks: List of TextChunk objects to reference
            
        Returns:
            List of Cluster objects
        """
        # Create a mapping of chunk_id to TextChunk
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        clusters = []
        for cluster_data in data.get('clusters', []):
            # Get chunks for this cluster
            cluster_chunks = [chunk_map[chunk_id] for chunk_id in cluster_data.get('chunk_ids', [])
                             if chunk_id in chunk_map]
            
            # Create Cluster object
            if cluster_chunks:
                cluster = Cluster(
                    cluster_id=cluster_data.get('cluster_id'),
                    chunks=cluster_chunks,
                    representative_chunk_id=cluster_data.get('representative_chunk_id')
                )
                clusters.append(cluster)
        
        logger.info(f"Loaded {len(clusters)} clusters from {input_path}")
        return clusters
    
    def manual_review(self, clusters: List[Cluster]) -> Dict[str, Any]:
        """
        Generate a summary of clusters for manual review
        
        Args:
            clusters: List of Cluster objects
            
        Returns:
            Dictionary with cluster statistics and samples
        """
        if not clusters:
            return {"error": "No clusters to review"}
        
        # Calculate basic statistics
        n_clusters = len(clusters)
        total_chunks = sum(cluster.size for cluster in clusters)
        avg_cluster_size = total_chunks / n_clusters if n_clusters > 0 else 0
        
        # Get cluster sizes
        cluster_sizes = [cluster.size for cluster in clusters]
        
        # Sample clusters (largest, smallest, middle)
        largest_cluster = max(clusters, key=lambda c: c.size)
        smallest_cluster = min(clusters, key=lambda c: c.size)
        
        # Sort clusters by size and get middle one
        sorted_clusters = sorted(clusters, key=lambda c: c.size)
        middle_cluster = sorted_clusters[len(sorted_clusters) // 2]
        
        # Create summary
        return {
            "n_clusters": n_clusters,
            "total_chunks": total_chunks,
            "avg_cluster_size": avg_cluster_size,
            "min_cluster_size": min(cluster_sizes),
            "max_cluster_size": max(cluster_sizes),
            "cluster_size_distribution": {
                "1-2": sum(1 for size in cluster_sizes if 1 <= size <= 2),
                "3-5": sum(1 for size in cluster_sizes if 3 <= size <= 5),
                "6-10": sum(1 for size in cluster_sizes if 6 <= size <= 10),
                "11+": sum(1 for size in cluster_sizes if size > 10)
            },
            "samples": {
                "largest_cluster": {
                    "id": largest_cluster.cluster_id,
                    "size": largest_cluster.size,
                    "duration": largest_cluster.duration,
                    "text": largest_cluster.text,
                    "chunk_ids": [chunk.chunk_id for chunk in largest_cluster.chunks[:5]]  # First 5 chunks
                },
                "smallest_cluster": {
                    "id": smallest_cluster.cluster_id,
                    "size": smallest_cluster.size,
                    "duration": smallest_cluster.duration,
                    "text": smallest_cluster.text,
                    "chunk_ids": [chunk.chunk_id for chunk in smallest_cluster.chunks]
                },
                "middle_cluster": {
                    "id": middle_cluster.cluster_id,
                    "size": middle_cluster.size,
                    "duration": middle_cluster.duration,
                    "text": middle_cluster.text,
                    "chunk_ids": [chunk.chunk_id for chunk in middle_cluster.chunks[:3]]  # First 3 chunks
                }
            }
        } 