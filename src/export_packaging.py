"""
export_packaging.py - Folder Structure & Export Packaging Module

This module handles:
1. Organizing the generated Markdown files into a well-structured folder hierarchy
2. Providing configurable options for folder organization based on YAML metadata
3. Bundling the folders and files into a ZIP archive for easy distribution
"""

import os
import re
import sys
import yaml
import shutil
import logging
import zipfile
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import fnmatch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FolderStructureConfig:
    """Configuration for folder structure organization"""
    # Base configuration
    input_dir: str = "obsidian_notes"  # Directory containing generated Markdown files
    output_dir: str = "organized_notes"  # Directory for organized structure
    archive_path: str = "obsidian_export.zip"  # Path for the ZIP archive
    
    # Organization strategies
    organize_by_topic: bool = True  # Create folders based on topics/tags
    organize_by_date: bool = False  # Create folders based on dates
    organize_by_entity: bool = True  # Create folders for entity references
    
    # Naming and structure configuration
    topic_folder_name: str = "Topics"  # Name of the root folder for topics
    date_folder_format: str = "%Y/%m/%d"  # Format for date-based folders (Year/Month/Day)
    entity_folder_name: str = "Entities"  # Name of the root folder for entities
    include_attachments_folder: bool = True  # Create an Attachments folder
    add_index_files: bool = True  # Create index.md files in folders
    
    # File naming
    rename_files: bool = True  # Whether to rename files (removing cluster IDs)
    max_filename_length: int = 100  # Maximum length for filenames
    
    # ZIP configuration
    compress_level: int = 9  # Compression level (0-9, higher = more compression)
    include_timestamp_in_archive_name: bool = True  # Add timestamp to ZIP filename
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        # Check directories
        if not self.input_dir:
            logger.error("Input directory cannot be empty")
            return False
        
        if not self.output_dir:
            logger.error("Output directory cannot be empty")
            return False
        
        if not self.archive_path:
            logger.error("Archive path cannot be empty")
            return False
        
        # Check max filename length (operating system constraints)
        if self.max_filename_length <= 0 or self.max_filename_length > 255:
            logger.error(f"Invalid max_filename_length: {self.max_filename_length}. Must be between 1 and 255")
            return False
        
        # Check compression level
        if self.compress_level < 0 or self.compress_level > 9:
            logger.error(f"Invalid compress_level: {self.compress_level}. Must be between 0 and 9")
            return False
        
        return True


class FolderOrganizer:
    """Organizes Markdown files into a structured folder hierarchy"""
    
    def __init__(self, config: Optional[FolderStructureConfig] = None):
        """Initialize the folder organizer with configuration"""
        self.config = config or FolderStructureConfig()
        if not self.config.validate():
            raise ValueError("Invalid folder structure configuration")
        
        self.topics_map: Dict[str, List[str]] = {}  # Maps topics to file paths
        self.dates_map: Dict[str, List[str]] = {}  # Maps dates to file paths
        self.entities_map: Dict[str, List[str]] = {}  # Maps entities to file paths
        self.entity_library_path: Optional[str] = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def organize_files(self) -> str:
        """
        Organize files from input directory into structured output directory
        Returns the path to the organized directory
        """
        logger.info(f"Organizing files from {self.config.input_dir} to {self.config.output_dir}")
        
        # Ensure input directory exists
        if not os.path.exists(self.config.input_dir):
            raise FileNotFoundError(f"Input directory not found: {self.config.input_dir}")
        
        # Clear output directory if it exists
        if os.path.exists(self.config.output_dir):
            shutil.rmtree(self.config.output_dir)
        os.makedirs(self.config.output_dir)
        
        # Process files and build mapping
        self._process_input_files()
        
        # Create structured folders and copy files
        self._create_folder_structure()
        
        # Create index files for navigation
        if self.config.add_index_files:
            self._create_index_files()
        
        logger.info(f"Successfully organized files into {self.config.output_dir}")
        return self.config.output_dir
    
    def _process_input_files(self):
        """Process all markdown files in input directory to build mapping"""
        logger.info("Processing input files and extracting metadata")
        
        # Reset mappings
        self.topics_map = {}
        self.dates_map = {}
        self.entities_map = {}
        
        # Find all markdown files
        md_files = [f for f in os.listdir(self.config.input_dir) 
                  if f.endswith('.md') and os.path.isfile(os.path.join(self.config.input_dir, f))]
        
        # Process entity library separately
        entity_library_files = [f for f in md_files if f.startswith("Entity-Library")]
        if entity_library_files:
            self.entity_library_path = os.path.join(self.config.input_dir, entity_library_files[0])
            # Remove from regular processing
            md_files = [f for f in md_files if not f.startswith("Entity-Library")]
        
        for filename in md_files:
            file_path = os.path.join(self.config.input_dir, filename)
            try:
                # Extract metadata from file
                frontmatter = self._extract_frontmatter(file_path)
                
                # Map by topics/tags
                if self.config.organize_by_topic and 'tags' in frontmatter:
                    for tag in frontmatter['tags']:
                        if tag not in self.topics_map:
                            self.topics_map[tag] = []
                        self.topics_map[tag].append(file_path)
                
                # Map by date
                if self.config.organize_by_date and 'date' in frontmatter:
                    date_str = frontmatter['date']
                    try:
                        # Handle date string that might be wrapped in quotes
                        if isinstance(date_str, str):
                            # Remove quotes if present
                            date_str = date_str.strip('"\'')
                            date_obj = datetime.fromisoformat(date_str)
                            date_folder = date_obj.strftime(self.config.date_folder_format)
                            if date_folder not in self.dates_map:
                                self.dates_map[date_folder] = []
                            self.dates_map[date_folder].append(file_path)
                    except ValueError:
                        logger.warning(f"Invalid date format in {file_path}: {date_str}")
                
                # Map by entities (if present in frontmatter)
                if self.config.organize_by_entity and 'entities' in frontmatter:
                    for entity in frontmatter['entities']:
                        if entity not in self.entities_map:
                            self.entities_map[entity] = []
                        self.entities_map[entity].append(file_path)
                
                # Also extract entities from the "Related Entities" section
                entities = self._extract_entities_from_content(file_path)
                for entity in entities:
                    if entity not in self.entities_map:
                        self.entities_map[entity] = []
                    if file_path not in self.entities_map[entity]:
                        self.entities_map[entity].append(file_path)
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
    
    def _extract_frontmatter(self, file_path: str) -> Dict[str, Any]:
        """Extract YAML frontmatter from a markdown file"""
        frontmatter = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file has YAML frontmatter (between --- markers)
            if content.startswith('---'):
                # Extract frontmatter content
                frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
                if frontmatter_match:
                    frontmatter_content = frontmatter_match.group(1)
                    frontmatter = yaml.safe_load(frontmatter_content)
            
            # Add creation date if not present
            if 'date' not in frontmatter:
                file_stats = os.stat(file_path)
                create_date = datetime.fromtimestamp(file_stats.st_ctime).isoformat()
                frontmatter['date'] = create_date
            
            return frontmatter
        except Exception as e:
            logger.warning(f"Error extracting frontmatter from {file_path}: {str(e)}")
            return {}
    
    def _extract_entities_from_content(self, file_path: str) -> List[str]:
        """Extract entities from the 'Related Entities' section of a markdown file"""
        entities = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for Related Entities section
            entity_section_match = re.search(r'## Related Entities\s*\n(.*?)(?:\n##|\Z)', 
                                            content, re.DOTALL)
            if entity_section_match:
                entity_section = entity_section_match.group(1)
                # Extract entity links with format [[Entity Name]]
                entity_links = re.findall(r'\[\[(.*?)\]\]', entity_section)
                entities.extend(entity_links)
            
            return entities
        except Exception as e:
            logger.warning(f"Error extracting entities from {file_path}: {str(e)}")
            return []
    
    def _create_folder_structure(self):
        """Create folder structure and copy files according to the mappings"""
        logger.info("Creating folder structure based on mappings")
        
        # Create base output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create default Notes folder for all notes
        notes_dir = os.path.join(self.config.output_dir, "Notes")
        os.makedirs(notes_dir, exist_ok=True)
        
        # Track which files have been copied to avoid duplication
        copied_files = set()
        
        # Copy all markdown files to the Notes directory first
        md_files = [f for f in os.listdir(self.config.input_dir) 
                  if f.endswith('.md') and os.path.isfile(os.path.join(self.config.input_dir, f))]
        
        for filename in md_files:
            src_path = os.path.join(self.config.input_dir, filename)
            
            # Skip entity library (will handle separately)
            if src_path == self.entity_library_path:
                continue
            
            # Rename file if configured
            if self.config.rename_files:
                # Remove Cluster ID from filename (Cluster-123-SomeTitle.md -> SomeTitle.md)
                new_filename = re.sub(r'^Cluster-\d+-', '', filename)
                # Ensure filename length is within limits
                if len(new_filename) > self.config.max_filename_length:
                    new_filename = new_filename[:self.config.max_filename_length-3] + ".md"
            else:
                new_filename = filename
            
            # Copy file to Notes directory
            dest_path = os.path.join(notes_dir, new_filename)
            shutil.copy2(src_path, dest_path)
            copied_files.add(src_path)
        
        # Create and populate Topics folder
        if self.config.organize_by_topic and self.topics_map:
            topics_dir = os.path.join(self.config.output_dir, self.config.topic_folder_name)
            os.makedirs(topics_dir, exist_ok=True)
            
            for topic, file_paths in self.topics_map.items():
                # Create safe folder name for topic
                safe_topic = self._create_safe_folder_name(topic)
                topic_path = os.path.join(topics_dir, safe_topic)
                os.makedirs(topic_path, exist_ok=True)
                
                # Copy files to topic folder
                for src_path in file_paths:
                    filename = os.path.basename(src_path)
                    if self.config.rename_files:
                        new_filename = re.sub(r'^Cluster-\d+-', '', filename)
                        if len(new_filename) > self.config.max_filename_length:
                            new_filename = new_filename[:self.config.max_filename_length-3] + ".md"
                    else:
                        new_filename = filename
                    
                    dest_path = os.path.join(topic_path, new_filename)
                    # Create symlink instead of copying to save space
                    relative_path = os.path.relpath(
                        os.path.join(notes_dir, new_filename), 
                        os.path.dirname(dest_path)
                    )
                    try:
                        if sys.platform == 'win32':
                            # Windows requires admin privileges for symlinks
                            # Use a hardlink or copy instead
                            shutil.copy2(os.path.join(notes_dir, new_filename), dest_path)
                        else:
                            os.symlink(relative_path, dest_path)
                    except Exception as e:
                        logger.warning(f"Error creating link for {dest_path}: {str(e)}. Using copy instead.")
                        shutil.copy2(os.path.join(notes_dir, new_filename), dest_path)
        
        # Create and populate Dates folder
        if self.config.organize_by_date and self.dates_map:
            for date_path, file_paths in self.dates_map.items():
                # Create folder path for date
                date_dir = os.path.join(self.config.output_dir, date_path)
                os.makedirs(date_dir, exist_ok=True)
                
                # Copy files to date folder
                for src_path in file_paths:
                    filename = os.path.basename(src_path)
                    if self.config.rename_files:
                        new_filename = re.sub(r'^Cluster-\d+-', '', filename)
                        if len(new_filename) > self.config.max_filename_length:
                            new_filename = new_filename[:self.config.max_filename_length-3] + ".md"
                    else:
                        new_filename = filename
                    
                    dest_path = os.path.join(date_dir, new_filename)
                    # Create symlink instead of copying to save space
                    relative_path = os.path.relpath(
                        os.path.join(notes_dir, new_filename), 
                        os.path.dirname(dest_path)
                    )
                    try:
                        if sys.platform == 'win32':
                            shutil.copy2(os.path.join(notes_dir, new_filename), dest_path)
                        else:
                            os.symlink(relative_path, dest_path)
                    except Exception as e:
                        logger.warning(f"Error creating link for {dest_path}: {str(e)}. Using copy instead.")
                        shutil.copy2(os.path.join(notes_dir, new_filename), dest_path)
        
        # Create and populate Entities folder
        if self.config.organize_by_entity and self.entities_map:
            entities_dir = os.path.join(self.config.output_dir, self.config.entity_folder_name)
            os.makedirs(entities_dir, exist_ok=True)
            
            # Copy Entity Library if it exists
            if self.entity_library_path:
                dest_path = os.path.join(entities_dir, os.path.basename(self.entity_library_path))
                shutil.copy2(self.entity_library_path, dest_path)
            
            for entity, file_paths in self.entities_map.items():
                # Create safe folder name for entity
                safe_entity = self._create_safe_folder_name(entity)
                entity_path = os.path.join(entities_dir, safe_entity)
                os.makedirs(entity_path, exist_ok=True)
                
                # Copy files to entity folder
                for src_path in file_paths:
                    filename = os.path.basename(src_path)
                    if self.config.rename_files:
                        new_filename = re.sub(r'^Cluster-\d+-', '', filename)
                        if len(new_filename) > self.config.max_filename_length:
                            new_filename = new_filename[:self.config.max_filename_length-3] + ".md"
                    else:
                        new_filename = filename
                    
                    dest_path = os.path.join(entity_path, new_filename)
                    # Create symlink instead of copying to save space
                    relative_path = os.path.relpath(
                        os.path.join(notes_dir, new_filename), 
                        os.path.dirname(dest_path)
                    )
                    try:
                        if sys.platform == 'win32':
                            shutil.copy2(os.path.join(notes_dir, new_filename), dest_path)
                        else:
                            os.symlink(relative_path, dest_path)
                    except Exception as e:
                        logger.warning(f"Error creating link for {dest_path}: {str(e)}. Using copy instead.")
                        shutil.copy2(os.path.join(notes_dir, new_filename), dest_path)
        
        # Create Attachments folder if configured
        if self.config.include_attachments_folder:
            attachments_dir = os.path.join(self.config.output_dir, "Attachments")
            os.makedirs(attachments_dir, exist_ok=True)
    
    def _create_safe_folder_name(self, name: str) -> str:
        """Create a safe folder name from a string"""
        # Replace invalid characters with underscore
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)
        # Remove leading/trailing periods or spaces
        safe_name = safe_name.strip('. ')
        # Limit length
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        return safe_name
    
    def _create_index_files(self):
        """Create index.md files in each folder for better navigation"""
        logger.info("Creating index files for navigation")
        
        # Create main index file
        main_index_path = os.path.join(self.config.output_dir, "index.md")
        with open(main_index_path, 'w', encoding='utf-8') as f:
            f.write("# Obsidian Notes - Table of Contents\n\n")
            
            # Add all sections
            f.write("## Main Sections\n\n")
            
            # Add Notes link
            f.write("- [[Notes/index|All Notes]]\n")
            
            # Add Topics link if exists
            if self.config.organize_by_topic and self.topics_map:
                f.write(f"- [[{self.config.topic_folder_name}/index|Topics]]\n")
            
            # Add Dates link if exists
            if self.config.organize_by_date and self.dates_map:
                f.write("- [[Dates|Dates]]\n")
            
            # Add Entities link if exists
            if self.config.organize_by_entity and self.entities_map:
                f.write(f"- [[{self.config.entity_folder_name}/index|Entities]]\n")
            
            # Add Attachments link if exists
            if self.config.include_attachments_folder:
                f.write("- [[Attachments|Attachments]]\n")
        
        # Create Notes index
        notes_index_path = os.path.join(self.config.output_dir, "Notes", "index.md")
        with open(notes_index_path, 'w', encoding='utf-8') as f:
            f.write("# All Notes\n\n")
            
            # List all notes
            notes_dir = os.path.join(self.config.output_dir, "Notes")
            note_files = [f for f in os.listdir(notes_dir) 
                        if f.endswith('.md') and f != "index.md"]
            
            for note_file in sorted(note_files):
                note_name = note_file[:-3]  # Remove .md extension
                f.write(f"- [[{note_name}]]\n")
        
        # Create Topics index if exists
        if self.config.organize_by_topic and self.topics_map:
            topics_index_path = os.path.join(self.config.output_dir, self.config.topic_folder_name, "index.md")
            os.makedirs(os.path.dirname(topics_index_path), exist_ok=True)
            
            with open(topics_index_path, 'w', encoding='utf-8') as f:
                f.write(f"# Topics\n\n")
                
                # List all topics
                for topic in sorted(self.topics_map.keys()):
                    safe_topic = self._create_safe_folder_name(topic)
                    f.write(f"- [[{safe_topic}/index|{topic}]]\n")
            
            # Create index file for each topic
            for topic in self.topics_map:
                safe_topic = self._create_safe_folder_name(topic)
                topic_index_path = os.path.join(self.config.output_dir, self.config.topic_folder_name, 
                                             safe_topic, "index.md")
                os.makedirs(os.path.dirname(topic_index_path), exist_ok=True)
                
                with open(topic_index_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Topic: {topic}\n\n")
                    
                    # List all notes in this topic
                    topic_dir = os.path.dirname(topic_index_path)
                    note_files = [f for f in os.listdir(topic_dir) 
                                if f.endswith('.md') and f != "index.md"]
                    
                    for note_file in sorted(note_files):
                        note_name = note_file[:-3]  # Remove .md extension
                        f.write(f"- [[{note_name}]]\n")
        
        # Create Entities index if exists
        if self.config.organize_by_entity and self.entities_map:
            entities_index_path = os.path.join(self.config.output_dir, self.config.entity_folder_name, "index.md")
            os.makedirs(os.path.dirname(entities_index_path), exist_ok=True)
            
            with open(entities_index_path, 'w', encoding='utf-8') as f:
                f.write(f"# Entities\n\n")
                
                # Link to Entity Library if it exists
                if self.entity_library_path:
                    library_name = os.path.basename(self.entity_library_path)[:-3]  # Remove .md
                    f.write(f"## [[{library_name}|Entity Library]]\n\n")
                
                # List all entities
                f.write("## Entity Folders\n\n")
                for entity in sorted(self.entities_map.keys()):
                    safe_entity = self._create_safe_folder_name(entity)
                    f.write(f"- [[{safe_entity}/index|{entity}]]\n")
            
            # Create index file for each entity
            for entity in self.entities_map:
                safe_entity = self._create_safe_folder_name(entity)
                entity_index_path = os.path.join(self.config.output_dir, self.config.entity_folder_name, 
                                             safe_entity, "index.md")
                os.makedirs(os.path.dirname(entity_index_path), exist_ok=True)
                
                with open(entity_index_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Entity: {entity}\n\n")
                    
                    # List all notes for this entity
                    entity_dir = os.path.dirname(entity_index_path)
                    note_files = [f for f in os.listdir(entity_dir) 
                                if f.endswith('.md') and f != "index.md"]
                    
                    for note_file in sorted(note_files):
                        note_name = note_file[:-3]  # Remove .md extension
                        f.write(f"- [[{note_name}]]\n")


class ZipPackager:
    """Creates a ZIP archive of the organized folder structure"""
    
    def __init__(self, config: Optional[FolderStructureConfig] = None):
        """Initialize the ZIP packager with configuration"""
        self.config = config or FolderStructureConfig()
        if not self.config.validate():
            raise ValueError("Invalid folder structure configuration")
    
    def create_zip_archive(self, dir_to_zip: Optional[str] = None) -> str:
        """Create a ZIP archive of the specified directory"""
        # Use the specified directory or the configured output directory
        source_dir = dir_to_zip or self.config.output_dir
        
        # Ensure source directory exists
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        
        # Create archive path with timestamp if configured
        archive_path = self.config.archive_path
        if self.config.include_timestamp_in_archive_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(archive_path)
            archive_path = f"{name}_{timestamp}{ext}"
        
        logger.info(f"Creating ZIP archive at {archive_path}")
        
        # Create ZIP file
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, 
                           compresslevel=self.config.compress_level) as zipf:
            # Walk through the directory structure
            for root, _, files in os.walk(source_dir):
                for file in files:
                    # Get the full file path
                    file_path = os.path.join(root, file)
                    
                    # Skip symlinks to avoid duplicates
                    if os.path.islink(file_path):
                        continue
                    
                    # Get relative path for the archive
                    rel_path = os.path.relpath(file_path, source_dir)
                    
                    # Add file to archive with relative path
                    zipf.write(file_path, rel_path)
                    logger.debug(f"Added to ZIP: {rel_path}")
        
        archive_size = os.path.getsize(archive_path) / (1024 * 1024)  # Size in MB
        logger.info(f"ZIP archive created successfully: {archive_path} ({archive_size:.2f} MB)")
        
        return archive_path


class ExportPackagingProcessor:
    """Main class for processing and packaging exported Markdown files"""
    
    def __init__(self, config: Optional[FolderStructureConfig] = None):
        """Initialize the export packaging processor with configuration"""
        self.config = config or FolderStructureConfig()
        if not self.config.validate():
            raise ValueError("Invalid folder structure configuration")
        
        self.folder_organizer = FolderOrganizer(self.config)
        self.zip_packager = ZipPackager(self.config)
    
    def process(self) -> Tuple[str, str]:
        """
        Process markdown files, organize them, and create a ZIP archive
        Returns a tuple with (organized_dir_path, zip_archive_path)
        """
        start_time = datetime.now()
        logger.info(f"Starting export packaging process at {start_time}")
        
        # Step 1: Organize files into folder structure
        try:
            organized_dir = self.folder_organizer.organize_files()
            logger.info(f"Files organized into {organized_dir}")
        except Exception as e:
            logger.error(f"Error organizing files: {str(e)}")
            raise
        
        # Step 2: Create ZIP archive
        try:
            zip_path = self.zip_packager.create_zip_archive(organized_dir)
            logger.info(f"ZIP archive created at {zip_path}")
        except Exception as e:
            logger.error(f"Error creating ZIP archive: {str(e)}")
            raise
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Export packaging completed in {duration.total_seconds():.2f} seconds")
        
        # Log folder structure
        self._log_folder_structure(organized_dir)
        
        return organized_dir, zip_path
    
    def _log_folder_structure(self, directory: str, indent: int = 0):
        """Log the folder structure recursively"""
        if indent == 0:
            logger.info("Folder structure created:")
        
        # Get directory contents
        try:
            contents = os.listdir(directory)
            
            # Sort directories first, then files
            dirs = [d for d in contents if os.path.isdir(os.path.join(directory, d))]
            files = [f for f in contents if os.path.isfile(os.path.join(directory, f))]
            
            # Log current directory name
            if indent == 0:
                logger.info(f"{' ' * indent}└── {os.path.basename(directory)}/")
            
            # Log directories
            for i, d in enumerate(sorted(dirs)):
                is_last = (i == len(dirs) - 1 and not files)
                prefix = "└── " if is_last else "├── "
                logger.info(f"{' ' * (indent+4)}{prefix}{d}/")
                
                # Recursively log subdirectories with increased indent
                self._log_folder_structure(os.path.join(directory, d), indent + 8)
            
            # Log files (limit to 5 per directory to avoid excessive logging)
            if files:
                if len(files) <= 5:
                    for i, f in enumerate(sorted(files)):
                        is_last = (i == len(files) - 1)
                        prefix = "└── " if is_last else "├── "
                        logger.info(f"{' ' * (indent+4)}{prefix}{f}")
                else:
                    # Show first 3 files
                    for i, f in enumerate(sorted(files)[:3]):
                        logger.info(f"{' ' * (indent+4)}├── {f}")
                    
                    # Show file count
                    logger.info(f"{' ' * (indent+4)}├── ... ({len(files) - 3} more files)")
                    
                    # Show last file
                    logger.info(f"{' ' * (indent+4)}└── {sorted(files)[-1]}")
        
        except Exception as e:
            logger.error(f"Error logging folder structure for {directory}: {str(e)}")


# For testing or direct usage
if __name__ == "__main__":
    # Example usage
    config = FolderStructureConfig(
        input_dir="obsidian_notes",
        output_dir="organized_notes",
        archive_path="obsidian_export.zip",
        organize_by_topic=True,
        organize_by_date=True,
        organize_by_entity=True
    )
    
    processor = ExportPackagingProcessor(config)
    try:
        organized_dir, zip_path = processor.process()
        print(f"Successfully organized files into {organized_dir}")
        print(f"ZIP archive created at {zip_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 