#!/usr/bin/env python3
"""
main.py - Rant to Rock Main Script

This script demonstrates how to use all the modules in the Rant to Rock pipeline
to process audio or text input into organized Markdown files for Obsidian.
It also provides a FastAPI server for UI integration.
"""

import os
import time
import argparse
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Import modules
from src.transcription import TranscriptionEngine, TranscriptionSegment
from src.chunking import ChunkingProcessor, ChunkingConfig, EmbeddingConfig
from src.clustering import ClusteringProcessor, ClusterConfig
from src.summarization import SummarizationProcessor, SummarizationConfig
from src.export_packaging import ExportPackagingProcessor, FolderStructureConfig

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Rant to Rock API")

# Add CORS middleware to allow requests from the UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from the frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to track processing status
processing_status = {
    "stage": "idle",
    "progress": 0,
    "message": None,
    "error": None
}

# Currently processing file info
current_file_info = None

# Latest processing results
latest_results = {
    "transcript_path": None,
    "chunks_path": None,
    "embeddings_path": None,
    "clusters_path": None,
    "notes_dir": None,
    "archive_path": None
}

# Define API models
class ProcessingOptions(BaseModel):
    fileType: str
    advancedClustering: bool = True
    generateEntities: bool = True
    includeTimestamps: bool = True

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Rant to Rock - Process audio or text into organized Markdown files")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio", type=str, help="Path to audio file for processing")
    input_group.add_argument("--text", type=str, help="Path to text file for processing")
    input_group.add_argument("--transcript", type=str, help="Path to existing transcript JSON file")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for output files")
    parser.add_argument("--organized-dir", type=str, help="Directory for organized output files (default: output_dir/organized)")
    parser.add_argument("--archive-path", type=str, help="Path for ZIP archive (default: output_dir/archive.zip)")
    
    # Configuration options
    parser.add_argument("--max-chunk-size", type=int, default=1000, help="Maximum chunk size in characters")
    parser.add_argument("--min-chunk-size", type=int, default=100, help="Minimum chunk size in characters")
    parser.add_argument("--overlap-size", type=int, default=200, help="Overlap size between chunks")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-large", help="Embedding model to use")
    parser.add_argument("--clustering-algorithm", type=str, default="hierarchical", help="Clustering algorithm to use")
    parser.add_argument("--summarization-model", type=str, default="gpt-4o", help="Summarization model to use")
    parser.add_argument("--max-filename-length", type=int, default=100, help="Maximum filename length")
    
    # Organization options
    parser.add_argument("--organize-by-topic", action="store_true", help="Organize files by topic")
    parser.add_argument("--organize-by-entity", action="store_true", help="Organize files by entity")
    parser.add_argument("--organize-by-date", action="store_true", help="Organize files by date")
    
    return parser.parse_args()

async def process_file_task(file_path: str, file_type: str, options: ProcessingOptions):
    """Background task to process a file through the pipeline"""
    global processing_status, latest_results
    
    try:
        # Create output directory
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Set organized_dir and archive_path
        organized_dir = os.path.join(output_dir, "organized")
        archive_path = os.path.join(output_dir, f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
        
        # Step 1: Transcription
        processing_status = {
            "stage": "transcribing",
            "progress": 10,
            "message": "Converting audio to text..." if file_type == "audio" else "Processing text input...",
            "error": None
        }
        
        transcription_engine = TranscriptionEngine()
        
        if file_type == "audio":
            segments = transcription_engine.process_audio(file_path)
        else:  # text or transcript
            segments = transcription_engine.process_text(file_path)
        
        # Save transcript
        transcript_path = os.path.join(output_dir, "transcript.json")
        transcription_engine.save_transcript(segments, transcript_path)
        latest_results["transcript_path"] = transcript_path
        
        processing_status["progress"] = 20
        
        # Step 2: Chunking
        processing_status = {
            "stage": "chunking",
            "progress": 30,
            "message": "Segmenting transcript into semantic chunks...",
            "error": None
        }
        
        chunking_config = ChunkingConfig(
            max_chunk_size=1000,
            min_chunk_size=100,
            overlap_size=200
        )
        
        embedding_config = EmbeddingConfig(
            model_name="text-embedding-3-large"
        )
        
        chunking_processor = ChunkingProcessor(
            chunking_config=chunking_config,
            embedding_config=embedding_config
        )
        
        chunks = chunking_processor.process_segments(segments)
        
        # Save chunks and embeddings
        chunks_path = os.path.join(output_dir, "chunks.json")
        embeddings_path = os.path.join(output_dir, "embeddings.npz")
        chunking_processor.save_chunks(chunks, chunks_path)
        chunking_processor.save_embeddings(chunks, embeddings_path)
        latest_results["chunks_path"] = chunks_path
        latest_results["embeddings_path"] = embeddings_path
        
        processing_status["progress"] = 45
        
        # Step 3: Clustering
        processing_status = {
            "stage": "clustering",
            "progress": 50,
            "message": "Applying hybrid semantic-temporal clustering..." if options.advancedClustering else "Clustering content by topic...",
            "error": None
        }
        
        cluster_config = ClusterConfig(
            algorithm="hierarchical",
            distance_threshold=0.5,
            temporal_weight=0.3 if options.advancedClustering else 0.0
        )
        
        clustering_processor = ClusteringProcessor(cluster_config)
        clusters = clustering_processor.process_chunks(chunks)
        
        # Save clusters
        clusters_path = os.path.join(output_dir, "clusters.json")
        clustering_processor.save_clusters(clusters, clusters_path)
        latest_results["clusters_path"] = clusters_path
        
        processing_status["progress"] = 65
        
        # Step 4: Summarization
        processing_status = {
            "stage": "summarizing",
            "progress": 70,
            "message": "Generating summaries and extracting entities..." if options.generateEntities else "Creating topic summaries...",
            "error": None
        }
        
        notes_dir = os.path.join(output_dir, "obsidian_notes")
        os.makedirs(notes_dir, exist_ok=True)
        
        summarization_config = SummarizationConfig(
            model_name="gpt-4o",
            output_dir=notes_dir,
            generate_entities=options.generateEntities,
            include_timestamps=options.includeTimestamps and file_type == "audio"
        )
        
        summarization_processor = SummarizationProcessor(summarization_config)
        file_paths = summarization_processor.process_clusters(clusters)
        latest_results["notes_dir"] = notes_dir
        
        processing_status["progress"] = 85
        
        # Step 5: Export Packaging
        processing_status = {
            "stage": "packaging",
            "progress": 90,
            "message": "Creating Obsidian-compatible files and folder structure...",
            "error": None
        }
        
        export_config = FolderStructureConfig(
            input_dir=notes_dir,
            output_dir=organized_dir,
            archive_path=archive_path,
            organize_by_topic=True,
            organize_by_entity=options.generateEntities,
            organize_by_date=False,
            max_filename_length=100
        )
        
        export_processor = ExportPackagingProcessor(export_config)
        export_result = export_processor.process()
        latest_results["archive_path"] = archive_path
        
        # Complete
        processing_status = {
            "stage": "complete",
            "progress": 100,
            "message": "Processing complete!",
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        processing_status = {
            "stage": "error",
            "progress": 0,
            "message": "An error occurred during processing. Please try again.",
            "error": str(e)
        }

def main():
    """Main function to run the pipeline"""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set default paths if not provided
    if not args.organized_dir:
        args.organized_dir = os.path.join(args.output_dir, "organized")
    
    if not args.archive_path:
        args.archive_path = os.path.join(args.output_dir, f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
    
    # Start timing
    start_time = time.time()
    
    # Step 1: Transcription
    logger.info("Step 1: Transcription")
    transcription_engine = TranscriptionEngine()
    
    if args.audio:
        logger.info(f"Processing audio file: {args.audio}")
        segments = transcription_engine.process_audio(args.audio)
    elif args.text:
        logger.info(f"Processing text file: {args.text}")
        segments = transcription_engine.process_text(args.text)
    elif args.transcript:
        logger.info(f"Loading transcript file: {args.transcript}")
        segments = transcription_engine.load_transcript(args.transcript)
    
    # Save transcript
    transcript_path = os.path.join(args.output_dir, "transcript.json")
    transcription_engine.save_transcript(segments, transcript_path)
    logger.info(f"Transcript saved to: {transcript_path}")
    
    # Step 2: Chunking
    logger.info("Step 2: Chunking")
    chunking_config = ChunkingConfig(
        max_chunk_size=args.max_chunk_size,
        min_chunk_size=args.min_chunk_size,
        overlap_size=args.overlap_size
    )
    
    embedding_config = EmbeddingConfig(
        model_name=args.embedding_model
    )
    
    chunking_processor = ChunkingProcessor(
        chunking_config=chunking_config,
        embedding_config=embedding_config
    )
    
    chunks = chunking_processor.process_segments(segments)
    
    # Save chunks and embeddings
    chunks_path = os.path.join(args.output_dir, "chunks.json")
    embeddings_path = os.path.join(args.output_dir, "embeddings.npz")
    chunking_processor.save_chunks(chunks, chunks_path)
    chunking_processor.save_embeddings(chunks, embeddings_path)
    logger.info(f"Chunks saved to: {chunks_path}")
    logger.info(f"Embeddings saved to: {embeddings_path}")
    
    # Step 3: Clustering
    logger.info("Step 3: Clustering")
    cluster_config = ClusterConfig(
        algorithm=args.clustering_algorithm,
        distance_threshold=0.5,
        temporal_weight=0.3
    )
    
    clustering_processor = ClusteringProcessor(cluster_config)
    clusters = clustering_processor.process_chunks(chunks)
    
    # Save clusters
    clusters_path = os.path.join(args.output_dir, "clusters.json")
    clustering_processor.save_clusters(clusters, clusters_path)
    logger.info(f"Clusters saved to: {clusters_path}")
    
    # Step 4: Summarization
    logger.info("Step 4: Summarization")
    notes_dir = os.path.join(args.output_dir, "obsidian_notes")
    os.makedirs(notes_dir, exist_ok=True)
    
    summarization_config = SummarizationConfig(
        model_name=args.summarization_model,
        output_dir=notes_dir
    )
    
    summarization_processor = SummarizationProcessor(summarization_config)
    file_paths = summarization_processor.process_clusters(clusters)
    
    logger.info(f"Markdown files generated in: {notes_dir}")
    
    # Step 5: Export Packaging
    logger.info("Step 5: Export Packaging")
    export_config = FolderStructureConfig(
        input_dir=notes_dir,
        output_dir=args.organized_dir,
        archive_path=args.archive_path,
        organize_by_topic=args.organize_by_topic,
        organize_by_entity=args.organize_by_entity,
        organize_by_date=args.organize_by_date,
        max_filename_length=args.max_filename_length
    )
    
    export_processor = ExportPackagingProcessor(export_config)
    export_result = export_processor.process()
    
    # End timing
    elapsed_time = time.time() - start_time
    
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
    logger.info(f"Organized files saved to: {args.organized_dir}")
    logger.info(f"ZIP archive created at: {args.archive_path}")

# API routes
@app.post("/api/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    options: str = Form(...)
):
    """Upload a file and start processing"""
    global current_file_info, processing_status
    
    # Reset processing status
    processing_status = {
        "stage": "idle",
        "progress": 0,
        "message": None,
        "error": None
    }
    
    try:
        # Parse options
        options_dict = json.loads(options)
        processing_options = ProcessingOptions(**options_dict)
        
        # Save file temporarily
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Store file info
        current_file_info = {
            "name": file.filename,
            "size": len(content),
            "type": file.content_type,
            "path": file_path
        }
        
        # Start processing in background
        background_tasks.add_task(
            process_file_task,
            file_path,
            processing_options.fileType,
            processing_options
        )
        
        return {"message": "File upload successful, processing started"}
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    """Get current processing status"""
    return processing_status

@app.get("/api/file-info")
async def get_file_info():
    """Get information about the currently processing file"""
    if not current_file_info:
        raise HTTPException(status_code=404, detail="No file is currently being processed")
    return current_file_info

@app.get("/api/cluster")
async def get_clusters():
    """Get clustering results"""
    if not latest_results["clusters_path"] or processing_status["stage"] not in ["summarizing", "packaging", "complete"]:
        raise HTTPException(status_code=404, detail="Clusters not available")
    
    try:
        with open(latest_results["clusters_path"], "r") as f:
            clusters = json.load(f)
        return clusters
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/summarize")
async def get_summaries():
    """Get summarization results"""
    if not latest_results["notes_dir"] or processing_status["stage"] not in ["packaging", "complete"]:
        raise HTTPException(status_code=404, detail="Summaries not available")
    
    try:
        summaries = []
        for filename in os.listdir(latest_results["notes_dir"]):
            if filename.endswith(".md"):
                with open(os.path.join(latest_results["notes_dir"], filename), "r") as f:
                    content = f.read()
                summaries.append({
                    "filename": filename,
                    "content": content
                })
        return {"summaries": summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export/zip")
async def download_zip():
    """Download the final ZIP archive"""
    if not latest_results["archive_path"] or processing_status["stage"] != "complete":
        raise HTTPException(status_code=404, detail="ZIP archive not available")
    
    try:
        return FileResponse(
            path=latest_results["archive_path"],
            filename=os.path.basename(latest_results["archive_path"]),
            media_type="application/zip"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    if os.environ.get("API_MODE", "").lower() == "true":
        # Run as API server
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run as CLI application
        main() 