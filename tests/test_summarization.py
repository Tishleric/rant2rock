"""
test_summarization.py - Tests for the Summarization & Markdown Generation Module

This file contains unit and integration tests for the summarization module,
including tests for summary generation, markdown conversion, and entity extraction.
"""

import os
import shutil
import tempfile
import unittest
import json
import re
import yaml
from unittest.mock import patch, MagicMock, Mock
import pytest
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Import modules to test
from src.summarization import (
    SummarizationConfig,
    EntityLibrary,
    SummarizationProcessor
)

# Import dependent modules
from src.clustering import Cluster
from src.chunking import TextChunk
from src.transcription import TranscriptionSegment


class TestSummarizationConfig(unittest.TestCase):
    """Test the SummarizationConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = SummarizationConfig()
        self.assertEqual(config.model_name, "gpt-4o")
        self.assertEqual(config.max_tokens, 1000)
        self.assertEqual(config.temperature, 0.3)
        self.assertTrue(config.include_yaml_frontmatter)
        self.assertTrue(config.include_timestamps)
        self.assertTrue(config.extract_topics)
        self.assertEqual(config.max_topics, 5)
        self.assertTrue(config.create_entity_library)
        self.assertEqual(config.entity_detection_threshold, 0.7)
        self.assertEqual(config.output_dir, "obsidian_notes")
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = SummarizationConfig(
            model_name="gpt-3.5-turbo",
            max_tokens=500,
            temperature=0.5,
            include_yaml_frontmatter=False,
            include_timestamps=False,
            extract_topics=False,
            max_topics=3,
            create_entity_library=False,
            entity_detection_threshold=0.5,
            output_dir="custom_notes"
        )
        
        self.assertEqual(config.model_name, "gpt-3.5-turbo")
        self.assertEqual(config.max_tokens, 500)
        self.assertEqual(config.temperature, 0.5)
        self.assertFalse(config.include_yaml_frontmatter)
        self.assertFalse(config.include_timestamps)
        self.assertFalse(config.extract_topics)
        self.assertEqual(config.max_topics, 3)
        self.assertFalse(config.create_entity_library)
        self.assertEqual(config.entity_detection_threshold, 0.5)
        self.assertEqual(config.output_dir, "custom_notes")
    
    def test_config_validation(self):
        """Test validation of configuration parameters"""
        # Valid configuration
        config = SummarizationConfig()
        self.assertTrue(config.validate())
        
        # Invalid model name
        config = SummarizationConfig(model_name="")
        self.assertFalse(config.validate())
        
        # Invalid max_tokens
        config = SummarizationConfig(max_tokens=0)
        self.assertFalse(config.validate())
        
        # Invalid temperature
        config = SummarizationConfig(temperature=1.5)
        self.assertFalse(config.validate())
        
        # Invalid max_topics
        config = SummarizationConfig(max_topics=0)
        self.assertFalse(config.validate())
        
        # Invalid entity_detection_threshold
        config = SummarizationConfig(entity_detection_threshold=1.5)
        self.assertFalse(config.validate())


class TestEntityLibrary(unittest.TestCase):
    """Test the EntityLibrary class"""
    
    def setUp(self):
        """Set up common objects for all tests"""
        self.library = EntityLibrary()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)
    
    def test_add_entity(self):
        """Test adding an entity to the library"""
        self.library.add_entity("Test Entity", "doc1", "This is a test context")
        
        # Verify entity was added
        self.assertIn("Test Entity", self.library.entities)
        self.assertEqual(self.library.entities["Test Entity"]["documents"], {"doc1"})
        self.assertEqual(len(self.library.entities["Test Entity"]["contexts"]), 1)
        self.assertEqual(self.library.entities["Test Entity"]["contexts"][0]["document_id"], "doc1")
        self.assertEqual(self.library.entities["Test Entity"]["contexts"][0]["text"], "This is a test context")
    
    def test_add_duplicate_context(self):
        """Test adding a duplicate context"""
        self.library.add_entity("Test Entity", "doc1", "This is a test context")
        self.library.add_entity("Test Entity", "doc1", "This is a test context")
        
        # Verify duplicate context was not added
        self.assertEqual(len(self.library.entities["Test Entity"]["contexts"]), 1)
    
    def test_add_multiple_documents(self):
        """Test adding an entity with multiple documents"""
        self.library.add_entity("Test Entity", "doc1", "Context 1")
        self.library.add_entity("Test Entity", "doc2", "Context 2")
        
        # Verify both documents were added
        self.assertEqual(self.library.entities["Test Entity"]["documents"], {"doc1", "doc2"})
        self.assertEqual(len(self.library.entities["Test Entity"]["contexts"]), 2)
    
    def test_get_related_entities(self):
        """Test getting related entities"""
        # Add entities with overlapping documents
        self.library.add_entity("Entity1", "doc1", "Context 1")
        self.library.add_entity("Entity2", "doc1", "Context 2")
        self.library.add_entity("Entity3", "doc2", "Context 3")
        self.library.add_entity("Entity1", "doc2", "Context 4")
        
        # Get related entities for Entity1
        related = self.library.get_related_entities("Entity1")
        
        # Entity2 and Entity3 should be related to Entity1
        self.assertIn("Entity2", related)
        self.assertIn("Entity3", related)
        
        # Get related entities for Entity2
        related = self.library.get_related_entities("Entity2")
        
        # Only Entity1 should be related to Entity2
        self.assertIn("Entity1", related)
        self.assertNotIn("Entity3", related)
    
    def test_to_markdown(self):
        """Test converting the library to Markdown"""
        # Add some entities
        self.library.add_entity("Entity1", "doc1", "Context 1")
        self.library.add_entity("Entity2", "doc1", "Context 2")
        
        # Get the Markdown output
        markdown = self.library.to_markdown()
        
        # Verify the Markdown contains basic elements
        self.assertIn("# Entity Cross-Reference", markdown)
        self.assertIn("## [[Entity1]]", markdown)
        self.assertIn("## [[Entity2]]", markdown)
        self.assertIn("From [[doc1]]:", markdown)
        self.assertIn("> Context 1", markdown)
        self.assertIn("> Context 2", markdown)
        self.assertIn("### Related Entities", markdown)
    
    def test_save_to_file(self):
        """Test saving the library to a file"""
        # Add an entity
        self.library.add_entity("Test Entity", "doc1", "This is a test context")
        
        # Save to file
        file_path = self.library.save_to_file(self.temp_dir)
        
        # Verify file exists
        self.assertTrue(os.path.exists(file_path))
        
        # Verify content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("# Entity Cross-Reference", content)
            self.assertIn("## [[Test Entity]]", content)
            self.assertIn("> This is a test context", content)


class TestSummarizationProcessor(unittest.TestCase):
    """Test the SummarizationProcessor class"""
    
    def setUp(self):
        """Set up common objects for all tests"""
        # Create test chunks and clusters
        self.segments = [
            TranscriptionSegment(text="This is segment one.", start_time=0.0, end_time=5.0, confidence=0.9),
            TranscriptionSegment(text="This is segment two.", start_time=5.0, end_time=10.0, confidence=0.8)
        ]
        
        self.chunks = [
            TextChunk(
                text="This is segment one. This is segment two.",
                start_time=0.0,
                end_time=10.0,
                chunk_id=1,
                segments=self.segments,
                embedding=np.random.rand(128)
            )
        ]
        
        self.cluster = Cluster(
            cluster_id=1,
            chunks=self.chunks
        )
        
        # Create temporary directory for output
        self.temp_dir = tempfile.mkdtemp()
        
        # Create processor with mock mode for testing
        self.config = SummarizationConfig(
            model_name="gpt-4o",
            output_dir=self.temp_dir
        )
        self.processor = SummarizationProcessor(self.config)
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)
    
    @patch('openai.ChatCompletion.create')
    def test_generate_summary(self, mock_chat_completion):
        """Test generating a summary"""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a test summary."
        mock_chat_completion.return_value = mock_response
        
        # Generate summary
        summary = self.processor._generate_summary(self.cluster)
        
        # Verify result
        self.assertEqual(summary, "This is a test summary.")
        
        # Verify API call
        mock_chat_completion.assert_called_once()
        call_args = mock_chat_completion.call_args[1]
        self.assertEqual(call_args['model'], "gpt-4o")
        self.assertEqual(call_args['max_tokens'], 1000)
        self.assertEqual(call_args['temperature'], 0.3)
        self.assertEqual(len(call_args['messages']), 2)
    
    @patch('openai.ChatCompletion.create')
    def test_generate_summary_api_error(self, mock_chat_completion):
        """Test handling API errors in summary generation"""
        # Mock API error
        mock_chat_completion.side_effect = Exception("API Error")
        
        # Generate summary with error
        summary = self.processor._generate_summary(self.cluster)
        
        # Verify fallback to partial content
        self.assertIn("[Error generating summary: API Error]", summary)
        self.assertIn("This is segment one. This is segment two.", summary)
    
    @patch('openai.ChatCompletion.create')
    def test_extract_entities(self, mock_chat_completion):
        """Test extracting entities"""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Entity1, Entity2, Entity3"
        mock_chat_completion.return_value = mock_response
        
        # Extract entities
        entities = self.processor._extract_entities("This is a test summary.", self.cluster)
        
        # Verify result
        self.assertEqual(entities, ["Entity1", "Entity2", "Entity3"])
        
        # Verify API call
        mock_chat_completion.assert_called_once()
        call_args = mock_chat_completion.call_args[1]
        self.assertEqual(call_args['model'], "gpt-4o")
        self.assertEqual(call_args['temperature'], 0.1)
    
    @patch('openai.ChatCompletion.create')
    def test_extract_topics(self, mock_chat_completion):
        """Test extracting topics"""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Topic1, Topic2, Topic3, Topic4, Topic5, Topic6"
        mock_chat_completion.return_value = mock_response
        
        # Extract topics
        topics = self.processor._extract_topics("This is a test summary.")
        
        # Verify result is limited to max_topics (5)
        self.assertEqual(topics, ["Topic1", "Topic2", "Topic3", "Topic4", "Topic5"])
        
        # Verify API call
        mock_chat_completion.assert_called_once()
        call_args = mock_chat_completion.call_args[1]
        self.assertEqual(call_args['model'], "gpt-4o")
        self.assertEqual(call_args['temperature'], 0.1)
    
    def test_format_time(self):
        """Test time formatting"""
        # Format various times
        self.assertEqual(self.processor._format_time(0), "00:00")
        self.assertEqual(self.processor._format_time(61), "01:01")
        self.assertEqual(self.processor._format_time(3661), "01:01:01")
    
    def test_convert_to_markdown(self):
        """Test converting to Markdown"""
        # Create a summary with title
        summary = "# Test Title\n\nThis is a test summary."
        entities = ["Entity1", "Entity2"]
        
        # Patch extract_topics to return predictable topics
        with patch.object(self.processor, '_extract_topics', return_value=["Topic1", "Topic2"]):
            # Convert to Markdown
            content = self.processor._convert_to_markdown(self.cluster, summary, entities)
            
            # Verify content
            self.assertEqual(content["title"], "Test Title")
            self.assertEqual(content["summary"], summary)
            self.assertIn("entity_links", content)
            self.assertIn("frontmatter", content)
            
            # Verify frontmatter contains expected keys
            frontmatter = content["frontmatter"]
            self.assertIn("title", frontmatter)
            self.assertIn("date", frontmatter)
            self.assertIn("cluster_id", frontmatter)
            self.assertIn("start_time", frontmatter)
            self.assertIn("end_time", frontmatter)
            self.assertIn("duration", frontmatter)
            self.assertIn("tags", frontmatter)
            
            # Verify entity links
            self.assertIn("[[Entity1]]", content["entity_links"])
            self.assertIn("[[Entity2]]", content["entity_links"])
    
    def test_save_markdown(self):
        """Test saving Markdown to a file"""
        # Create markdown content
        markdown_content = {
            "title": "Test Title",
            "summary": "# Test Title\n\nThis is a test summary.",
            "frontmatter": {
                "title": "Test Title",
                "date": "2023-11-07",
                "tags": ["Topic1", "Topic2"]
            },
            "entity_links": "\n## Related Entities\n\n- [[Entity1]]\n- [[Entity2]]",
            "timespan_info": "\n## Timespan\n\n- **Start:** 00:00\n- **End:** 10:00\n- **Duration:** 10:00\n"
        }
        
        # Save to file
        file_path = self.processor._save_markdown(markdown_content, self.cluster, self.temp_dir)
        
        # Verify file exists
        self.assertTrue(os.path.exists(file_path))
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Verify YAML frontmatter
            self.assertIn("---", content)
            self.assertIn("title: Test Title", content)
            self.assertIn("tags:", content)
            self.assertIn("- Topic1", content)
            self.assertIn("- Topic2", content)
            
            # Verify content elements
            self.assertIn("# Test Title", content)
            self.assertIn("This is a test summary.", content)
            self.assertIn("## Related Entities", content)
            self.assertIn("[[Entity1]]", content)
            self.assertIn("[[Entity2]]", content)
            self.assertIn("## Timespan", content)
    
    @patch('openai.ChatCompletion.create')
    def test_process_clusters(self, mock_chat_completion):
        """Test processing multiple clusters"""
        # Mock chat.completions.create with side effect
        def mock_completions_side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            content = messages[1]['content'] if len(messages) > 1 else ""
            
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            if "create a concise summary" in content:
                mock_response.choices[0].message.content = "This is a test summary."
            elif "Extract the key entities" in content:
                mock_response.choices[0].message.content = "Entity1, Entity2"
            elif "Extract" in content and "key topics" in content:
                mock_response.choices[0].message.content = "Topic1, Topic2"
            return mock_response
            
        mock_chat_completion.side_effect = mock_completions_side_effect
        
        # Process clusters
        clusters = [self.cluster]
        file_paths = self.processor.process_clusters(clusters, self.temp_dir)
        
        # Verify output
        self.assertEqual(len(file_paths), 2)  # One for the cluster, one for the entity library
        for path in file_paths:
            self.assertTrue(os.path.exists(path))
            
        # Verify entity library was created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "Entity-Library.md")))
    
    def test_get_mock_summary(self):
        """Test generating a mock summary for testing"""
        summary = self.processor.get_mock_summary(self.cluster)
        
        # Verify the summary structure
        self.assertIn(f"# Summary of Cluster {self.cluster.cluster_id}", summary)
        self.assertIn("This is segment one. This is segment two.", summary)
        
    def test_generate_sample_markdown(self):
        """Test generating a sample markdown file"""
        file_path = self.processor.generate_sample_markdown(self.cluster, self.temp_dir)
        
        # Verify file exists
        self.assertTrue(os.path.exists(file_path))
        
        # Check file contents
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn(f"# Summary of Cluster {self.cluster.cluster_id}", content)
            self.assertIn("This is segment one. This is segment two.", content)
            self.assertIn("## Related Entities", content)
            self.assertIn("[[Entity1]]", content)


# Integration test for the summarization module
class TestSummarizationIntegration:
    """Integration tests for the summarization module"""
    
    @pytest.fixture
    def setup_test_data(self):
        """Set up test data for integration tests"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create test segments
        segments1 = [
            TranscriptionSegment(text="This is the first segment about AI.", start_time=0.0, end_time=5.0, confidence=0.9),
            TranscriptionSegment(text="AI systems are changing how we work.", start_time=5.0, end_time=10.0, confidence=0.8)
        ]
        
        segments2 = [
            TranscriptionSegment(text="Climate change is a global challenge.", start_time=15.0, end_time=20.0, confidence=0.9),
            TranscriptionSegment(text="We need to reduce carbon emissions.", start_time=20.0, end_time=25.0, confidence=0.8)
        ]
        
        # Create chunks
        chunk1 = TextChunk(
            text="This is the first segment about AI. AI systems are changing how we work.",
            start_time=0.0,
            end_time=10.0,
            chunk_id=1,
            segments=segments1,
            embedding=np.random.rand(128)
        )
        
        chunk2 = TextChunk(
            text="Climate change is a global challenge. We need to reduce carbon emissions.",
            start_time=15.0,
            end_time=25.0,
            chunk_id=2,
            segments=segments2,
            embedding=np.random.rand(128)
        )
        
        # Create clusters
        cluster1 = Cluster(cluster_id=1, chunks=[chunk1])
        cluster2 = Cluster(cluster_id=2, chunks=[chunk2])
        
        yield temp_dir, [cluster1, cluster2]
        
        # Clean up
        shutil.rmtree(temp_dir)
    
    @patch('openai.ChatCompletion.create')
    def test_end_to_end_summarization(self, mock_chat_completion, setup_test_data):
        """Test end-to-end summarization process"""
        temp_dir, clusters = setup_test_data
        
        # Mock chat.completions.create with side effect
        def mock_completions_side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            content = messages[1]['content'] if len(messages) > 1 else ""
            
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            if "create a concise summary" in content:
                if "AI" in content:
                    mock_response.choices[0].message.content = "# Artificial Intelligence Impact\n\nAI systems are revolutionizing work processes and transforming industries."
                else:
                    mock_response.choices[0].message.content = "# Climate Change Action\n\nClimate change requires global action to reduce carbon emissions and mitigate effects."
            elif "Extract the key entities" in content:
                if "AI" in content:
                    mock_response.choices[0].message.content = "Artificial Intelligence, Work Transformation, Technology"
                else:
                    mock_response.choices[0].message.content = "Climate Change, Carbon Emissions, Global Challenge"
            elif "Extract" in content and "key topics" in content:
                if "AI" in content:
                    mock_response.choices[0].message.content = "AI, Technology, Work, Transformation"
                else:
                    mock_response.choices[0].message.content = "Climate, Environment, Carbon, Global"
            return mock_response
            
        mock_chat_completion.side_effect = mock_completions_side_effect
        
        # Create processor
        config = SummarizationConfig(output_dir=temp_dir)
        processor = SummarizationProcessor(config)
        
        # Process clusters
        file_paths = processor.process_clusters(clusters)
        
        # Verify output files
        assert len(file_paths) == 3  # Two cluster files plus entity library
        
        # Check first cluster file
        ai_file = [f for f in file_paths if "Cluster-1" in f][0]
        with open(ai_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "# Artificial Intelligence Impact" in content
            assert "AI systems" in content
            assert "tags:" in content
            assert "- AI" in content
            assert "[[Artificial Intelligence]]" in content
        
        # Check second cluster file
        climate_file = [f for f in file_paths if "Cluster-2" in f][0]
        with open(climate_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "# Climate Change Action" in content
            assert "carbon emissions" in content
            assert "tags:" in content
            assert "- Climate" in content
            assert "[[Climate Change]]" in content
        
        # Check entity library
        entity_file = [f for f in file_paths if "Entity-Library" in f][0]
        with open(entity_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "# Entity Cross-Reference" in content
            assert "## [[Artificial Intelligence]]" in content
            assert "## [[Climate Change]]" in content
    
    def test_markdown_syntax_validation(self, setup_test_data):
        """Test that generated Markdown adheres to Obsidian syntax"""
        temp_dir, clusters = setup_test_data
        
        # Create processor with mock mode to avoid API calls
        config = SummarizationConfig(output_dir=temp_dir)
        processor = SummarizationProcessor(config)
        
        # Use mock summary generation
        with patch.object(processor, '_generate_summary', side_effect=processor.get_mock_summary), \
             patch.object(processor, '_extract_entities', return_value=["Entity1", "Entity2"]), \
             patch.object(processor, '_extract_topics', return_value=["Topic1", "Topic2"]):
            
            # Process clusters
            file_paths = processor.process_clusters(clusters)
            
            # Check each output file for valid Obsidian syntax
            for file_path in file_paths:
                if not os.path.basename(file_path).startswith("Entity-Library"):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for YAML frontmatter
                        assert re.search(r'^---\n.*\n---\n', content, re.DOTALL) is not None, "Missing valid YAML frontmatter"
                        
                        # Check for internal links
                        assert re.search(r'\[\[Entity\d+\]\]', content) is not None, "Missing Obsidian internal links"
                        
                        # Check for headers
                        assert re.search(r'# [^\n]+\n', content) is not None, "Missing headers"
                        
                        # Validate YAML syntax
                        yaml_match = re.search(r'---\n(.*?)\n---', content, re.DOTALL)
                        if yaml_match:
                            yaml_content = yaml_match.group(1)
                            try:
                                parsed_yaml = yaml.safe_load(yaml_content)
                                assert isinstance(parsed_yaml, dict), "YAML content is not a valid dictionary"
                            except yaml.YAMLError:
                                pytest.fail("Invalid YAML syntax in frontmatter") 