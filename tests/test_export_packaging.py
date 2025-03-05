"""
Unit tests for the Export Packaging module.

This file tests:
1. Folder hierarchy creation from Markdown files
2. ZIP archive packaging
3. Edge cases such as empty folders and long file names
"""

import os
import sys
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
from src.export_packaging import (
    FolderStructureConfig,
    FolderOrganizer,
    ZipPackager,
    ExportPackagingProcessor
)


class TestFolderStructureConfig(unittest.TestCase):
    """Test the folder structure configuration class"""

    def test_default_config(self):
        """Test default configuration values"""
        config = FolderStructureConfig()
        self.assertEqual(config.input_dir, "obsidian_notes")
        self.assertEqual(config.output_dir, "organized_notes")
        self.assertEqual(config.archive_path, "obsidian_export.zip")
        self.assertTrue(config.organize_by_topic)
        self.assertFalse(config.organize_by_date)
        self.assertTrue(config.organize_by_entity)
        self.assertEqual(config.compress_level, 9)

    def test_custom_config(self):
        """Test custom configuration values"""
        config = FolderStructureConfig(
            input_dir="custom_input",
            output_dir="custom_output",
            archive_path="custom_archive.zip",
            organize_by_topic=False,
            organize_by_date=True,
            organize_by_entity=False,
            compress_level=5
        )
        self.assertEqual(config.input_dir, "custom_input")
        self.assertEqual(config.output_dir, "custom_output")
        self.assertEqual(config.archive_path, "custom_archive.zip")
        self.assertFalse(config.organize_by_topic)
        self.assertTrue(config.organize_by_date)
        self.assertFalse(config.organize_by_entity)
        self.assertEqual(config.compress_level, 5)

    def test_validation(self):
        """Test configuration validation"""
        # Valid config
        config = FolderStructureConfig()
        self.assertTrue(config.validate())

        # Invalid input_dir
        config = FolderStructureConfig(input_dir="")
        self.assertFalse(config.validate())

        # Invalid output_dir
        config = FolderStructureConfig(output_dir="")
        self.assertFalse(config.validate())

        # Invalid archive_path
        config = FolderStructureConfig(archive_path="")
        self.assertFalse(config.validate())

        # Invalid max_filename_length
        config = FolderStructureConfig(max_filename_length=0)
        self.assertFalse(config.validate())
        config = FolderStructureConfig(max_filename_length=300)
        self.assertFalse(config.validate())

        # Invalid compress_level
        config = FolderStructureConfig(compress_level=-1)
        self.assertFalse(config.validate())
        config = FolderStructureConfig(compress_level=10)
        self.assertFalse(config.validate())


class TestExportPackaging(unittest.TestCase):
    """Test the Export Packaging functionality with actual files"""

    def setUp(self):
        """Set up temporary directories and sample files for testing"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, "input")
        self.output_dir = os.path.join(self.temp_dir, "output")
        self.archive_path = os.path.join(self.temp_dir, "archive.zip")
        
        # Create input directory
        os.makedirs(self.input_dir, exist_ok=True)
        
        # Create sample Markdown files with YAML frontmatter
        self._create_sample_files()

    def tearDown(self):
        """Clean up temporary directories"""
        shutil.rmtree(self.temp_dir)

    def _create_sample_files(self):
        """Create sample Markdown files for testing"""
        # Cluster file with topics and entities
        with open(os.path.join(self.input_dir, "Cluster-1-AI-Technology.md"), "w", encoding="utf-8") as f:
            f.write("""---
title: AI Technology
date: "2023-03-01T10:15:30"
tags:
  - AI
  - Technology
  - Machine Learning
---

# AI Technology

This is a sample note about AI technology.

## Related Entities

- [[Machine Learning]]
- [[Neural Networks]]

## Timespan

- Start: 00:05:10
- End: 00:15:20
- Duration: 00:10:10
""")

        # Another cluster file with different topics
        with open(os.path.join(self.input_dir, "Cluster-2-Climate-Change.md"), "w", encoding="utf-8") as f:
            f.write("""---
title: Climate Change
date: "2023-03-02T14:25:40"
tags:
  - Environment
  - Climate
  - Science
---

# Climate Change

This is a sample note about climate change.

## Related Entities

- [[Global Warming]]
- [[Carbon Emissions]]

## Timespan

- Start: 00:20:30
- End: 00:32:45
- Duration: 00:12:15
""")

        # Entity library file
        with open(os.path.join(self.input_dir, "Entity-Library.md"), "w", encoding="utf-8") as f:
            f.write("""---
title: Entity Library
date: "2023-03-03T09:00:00"
---

# Entity Library

## Machine Learning

Found in:
- [[Cluster-1-AI-Technology]]

## Neural Networks

Found in:
- [[Cluster-1-AI-Technology]]

## Global Warming

Found in:
- [[Cluster-2-Climate-Change]]

## Carbon Emissions

Found in:
- [[Cluster-2-Climate-Change]]
""")

        # File with very long name
        long_name = "Cluster-3-" + "A" * 200 + ".md"
        with open(os.path.join(self.input_dir, long_name), "w", encoding="utf-8") as f:
            f.write("""---
title: Very Long File Name
date: "2023-03-04T11:22:33"
tags:
  - Test
---

# Very Long File Name

This is a file with a very long name.

## Related Entities

- [[Test Entity]]

## Timespan

- Start: 00:40:00
- End: 00:45:00
- Duration: 00:05:00
""")

    def test_folder_organizer(self):
        """Test folder organizer functionality"""
        # Create config
        config = FolderStructureConfig(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            archive_path=self.archive_path,
            organize_by_topic=True,
            organize_by_date=True,
            organize_by_entity=True,
            include_timestamp_in_archive_name=False  # Disable for testing
        )
        
        # Create folder organizer
        organizer = FolderOrganizer(config)
        
        # Organize files
        result_dir = organizer.organize_files()
        
        # Check that output directory exists
        self.assertTrue(os.path.exists(self.output_dir))
        self.assertEqual(result_dir, self.output_dir)
        
        # Check that main folders exist
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "Notes")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "Topics")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "Entities")))
        
        # Check that index files were created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "index.md")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "Notes", "index.md")))
        
        # Check that topic folders were created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "Topics", "AI")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "Topics", "Technology")))
        # Note: the folder name might be "Machine Learning" or "Machine_Learning" depending on the implementation
        machine_learning_path = os.path.join(self.output_dir, "Topics", "Machine Learning")
        if not os.path.exists(machine_learning_path):
            machine_learning_path = os.path.join(self.output_dir, "Topics", "Machine_Learning")
        self.assertTrue(os.path.exists(machine_learning_path))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "Topics", "Environment")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "Topics", "Climate")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "Topics", "Science")))
        
        # Check that entity folders were created - use the same flexible approach
        ml_entity_path = os.path.join(self.output_dir, "Entities", "Machine Learning")
        if not os.path.exists(ml_entity_path):
            ml_entity_path = os.path.join(self.output_dir, "Entities", "Machine_Learning")
        self.assertTrue(os.path.exists(ml_entity_path))
        
        nn_entity_path = os.path.join(self.output_dir, "Entities", "Neural Networks")
        if not os.path.exists(nn_entity_path):
            nn_entity_path = os.path.join(self.output_dir, "Entities", "Neural_Networks")
        self.assertTrue(os.path.exists(nn_entity_path))
        
        gw_entity_path = os.path.join(self.output_dir, "Entities", "Global Warming")
        if not os.path.exists(gw_entity_path):
            gw_entity_path = os.path.join(self.output_dir, "Entities", "Global_Warming")
        self.assertTrue(os.path.exists(gw_entity_path))
        
        ce_entity_path = os.path.join(self.output_dir, "Entities", "Carbon Emissions")
        if not os.path.exists(ce_entity_path):
            ce_entity_path = os.path.join(self.output_dir, "Entities", "Carbon_Emissions")
        self.assertTrue(os.path.exists(ce_entity_path))
        
        # Check that files were renamed correctly
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "Notes", "AI-Technology.md")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "Notes", "Climate-Change.md")))
        
        # Check that long filename was truncated
        files = os.listdir(os.path.join(self.output_dir, "Notes"))
        long_files = [f for f in files if f.startswith("A") and not f.startswith("AI-Technology")]
        self.assertEqual(len(long_files), 1)
        self.assertLessEqual(len(long_files[0]), config.max_filename_length)
        
        # Check that date-based folders were created
        year_folder = os.path.join(self.output_dir, "2023")
        self.assertTrue(os.path.exists(year_folder))
        month_folder = os.path.join(year_folder, "03")
        self.assertTrue(os.path.exists(month_folder))
        
        # Check that entity library was copied
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "Entities", "Entity-Library.md")))

    def test_zip_packager(self):
        """Test ZIP packager functionality"""
        # Create config
        config = FolderStructureConfig(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            archive_path=self.archive_path,
            include_timestamp_in_archive_name=False  # Don't include timestamp for testing
        )
        
        # First organize the files
        organizer = FolderOrganizer(config)
        organizer.organize_files()
        
        # Create ZIP packager
        packager = ZipPackager(config)
        
        # Create ZIP archive
        result_path = packager.create_zip_archive()
        
        # Check that archive exists
        self.assertTrue(os.path.exists(self.archive_path))
        self.assertEqual(result_path, self.archive_path)
        
        # Check that archive is a valid ZIP file
        self.assertTrue(zipfile.is_zipfile(self.archive_path))
        
        # Extract archive to temporary directory and verify contents
        extract_dir = os.path.join(self.temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(self.archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Check that main folders exist in extracted archive
        self.assertTrue(os.path.exists(os.path.join(extract_dir, "Notes")))
        self.assertTrue(os.path.exists(os.path.join(extract_dir, "Topics")))
        self.assertTrue(os.path.exists(os.path.join(extract_dir, "Entities")))
        
        # Check that files exist in extracted archive
        self.assertTrue(os.path.exists(os.path.join(extract_dir, "Notes", "AI-Technology.md")))
        self.assertTrue(os.path.exists(os.path.join(extract_dir, "Notes", "Climate-Change.md")))

    def test_export_packaging_processor(self):
        """Test the complete export packaging process"""
        # Create config
        config = FolderStructureConfig(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            archive_path=self.archive_path,
            include_timestamp_in_archive_name=False  # Don't include timestamp for testing
        )
        
        # Create processor
        processor = ExportPackagingProcessor(config)
        
        # Process files
        organized_dir, zip_path = processor.process()
        
        # Check results
        self.assertEqual(organized_dir, self.output_dir)
        self.assertEqual(zip_path, self.archive_path)
        self.assertTrue(os.path.exists(self.output_dir))
        self.assertTrue(os.path.exists(self.archive_path))
        
        # Check ZIP file
        self.assertTrue(zipfile.is_zipfile(self.archive_path))

    def test_empty_input_directory(self):
        """Test handling of empty input directory"""
        # Create empty input directory
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        
        # Create config
        config = FolderStructureConfig(
            input_dir=empty_dir,
            output_dir=self.output_dir,
            archive_path=self.archive_path,
            include_timestamp_in_archive_name=False  # Don't include timestamp for testing
        )
        
        # Create processor
        processor = ExportPackagingProcessor(config)
        
        # Process files (should not raise an error)
        organized_dir, zip_path = processor.process()
        
        # Check that output directory exists with base structure
        self.assertTrue(os.path.exists(self.output_dir))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "Notes")))
        
        # Check that ZIP file exists
        self.assertTrue(os.path.exists(zip_path))  # Use the returned zip_path instead of self.archive_path

    def test_invalid_input_directory(self):
        """Test handling of non-existent input directory"""
        # Create config with non-existent input directory
        config = FolderStructureConfig(
            input_dir="nonexistent_directory",
            output_dir=self.output_dir,
            archive_path=self.archive_path
        )
        
        # Create processor
        processor = ExportPackagingProcessor(config)
        
        # Process files (should raise FileNotFoundError)
        with self.assertRaises(FileNotFoundError):
            processor.process()

    def test_very_long_filenames(self):
        """Test handling of extremely long filenames"""
        # Create a file with a name at the OS limit
        max_length = 255  # Maximum filename length on most filesystems
        long_name = "Cluster-4-" + "B" * (max_length - 20) + ".md"
        with open(os.path.join(self.input_dir, long_name), "w", encoding="utf-8") as f:
            f.write("""---
title: Extremely Long File Name
date: "2023-03-05T15:30:45"
tags:
  - LongName
---

# Extremely Long File Name

This is a file with an extremely long name, approaching OS limits.
""")
        
        # Create config
        config = FolderStructureConfig(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            archive_path=self.archive_path,
            max_filename_length=100,  # Limit to 100 characters
            include_timestamp_in_archive_name=False  # Don't include timestamp for testing
        )
        
        # Create processor
        processor = ExportPackagingProcessor(config)
        
        # Process files
        organized_dir, zip_path = processor.process()
        
        # Check that the file was renamed correctly
        notes_dir = os.path.join(self.output_dir, "Notes")
        files = os.listdir(notes_dir)
        b_files = [f for f in files if f.startswith("B")]
        self.assertEqual(len(b_files), 1)
        self.assertLessEqual(len(b_files[0]), config.max_filename_length)


if __name__ == "__main__":
    unittest.main() 