"""
summarization.py - Summarization & Markdown Generation Module

This module handles:
1. Processing clusters to generate cohesive summaries using NLP models
2. Converting the summaries into Obsidian-friendly Markdown files
3. Generating metadata and cross-references for enhanced organization
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
import datetime
from pathlib import Path
import yaml
import openai
from dotenv import load_dotenv

# Local imports
from src.clustering import Cluster

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()  # Load from .env file if present

# Securely load the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    logger.info("OpenAI API key loaded successfully. Using real API for summarization.")
else:
    logger.warning("No OPENAI_API_KEY found in environment variables. Please set it to use the OpenAI API.")
    logger.warning("For summarization, an API key is required.")


@dataclass
class SummarizationConfig:
    """Configuration for summarization and markdown generation"""
    # Summarization parameters
    model_name: str = "gpt-4o"  # Model for summarization
    max_tokens: int = 1000  # Maximum tokens for summary generation
    temperature: float = 0.3  # Controls randomness (lower = more deterministic)
    
    # Markdown parameters
    include_yaml_frontmatter: bool = True  # Whether to include YAML frontmatter
    include_timestamps: bool = True  # Whether to include timestamps
    extract_topics: bool = True  # Whether to extract topics for tags
    max_topics: int = 5  # Maximum number of topics to extract
    
    # Entity linking parameters
    create_entity_library: bool = True  # Whether to create an entity cross-reference
    entity_detection_threshold: float = 0.7  # Confidence threshold for entity detection
    
    # Output parameters
    output_dir: str = "obsidian_notes"  # Directory to save Markdown files
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        # Check model name
        if not self.model_name:
            logger.error("Model name cannot be empty")
            return False
        
        # Check tokens
        if self.max_tokens <= 0:
            logger.error(f"Invalid max_tokens: {self.max_tokens}. Must be positive")
            return False
        
        # Check temperature
        if self.temperature < 0 or self.temperature > 1:
            logger.error(f"Invalid temperature: {self.temperature}. Must be between 0 and 1")
            return False
        
        # Check topic count
        if self.max_topics <= 0:
            logger.error(f"Invalid max_topics: {self.max_topics}. Must be positive")
            return False
        
        # Check entity detection threshold
        if self.entity_detection_threshold < 0 or self.entity_detection_threshold > 1:
            logger.error(f"Invalid entity_detection_threshold: {self.entity_detection_threshold}. Must be between 0 and 1")
            return False
        
        return True


class EntityLibrary:
    """Manages entity cross-referencing across multiple summaries"""
    
    def __init__(self):
        self.entities: Dict[str, Dict[str, Any]] = {}
        
    def add_entity(self, entity_name: str, document_id: str, context: str):
        """Add an entity reference to the library"""
        entity_name = entity_name.strip()
        if not entity_name:
            return
        
        if entity_name not in self.entities:
            self.entities[entity_name] = {
                "documents": set(),
                "contexts": []
            }
        
        self.entities[entity_name]["documents"].add(document_id)
        # Avoid duplicate contexts
        for existing_context in self.entities[entity_name]["contexts"]:
            if existing_context["document_id"] == document_id and existing_context["text"] == context:
                return
        
        self.entities[entity_name]["contexts"].append({
            "document_id": document_id,
            "text": context
        })
    
    def get_related_entities(self, entity_name: str) -> List[str]:
        """Get entities that co-occur in documents with the given entity"""
        if entity_name not in self.entities:
            return []
        
        related_entities = []
        documents = self.entities[entity_name]["documents"]
        
        for other_entity, data in self.entities.items():
            if other_entity != entity_name:
                # Check if there's document overlap
                if documents.intersection(data["documents"]):
                    related_entities.append(other_entity)
        
        return related_entities
    
    def to_markdown(self) -> str:
        """Convert the entity library to Markdown format"""
        output = "# Entity Cross-Reference\n\n"
        
        # Sort entities alphabetically
        for entity in sorted(self.entities.keys()):
            doc_count = len(self.entities[entity]["documents"])
            output += f"## [[{entity}]]\n\n"
            output += f"Appears in {doc_count} document{'s' if doc_count != 1 else ''}.\n\n"
            
            # Add related entities
            related = self.get_related_entities(entity)
            if related:
                output += "### Related Entities\n\n"
                for related_entity in sorted(related):
                    output += f"- [[{related_entity}]]\n"
                output += "\n"
            
            # Add contexts
            output += "### Contexts\n\n"
            for context in self.entities[entity]["contexts"]:
                doc_id = context["document_id"]
                output += f"From [[{doc_id}]]:\n"
                output += f"> {context['text']}\n\n"
        
        return output
    
    def save_to_file(self, output_dir: str) -> str:
        """Save the entity library to a file"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "Entity-Library.md")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.to_markdown())
        
        return output_path


class SummarizationProcessor:
    """Processes clusters to generate summaries and Markdown files"""
    
    def __init__(self, config: Optional[SummarizationConfig] = None):
        """Initialize with configuration"""
        self.config = config or SummarizationConfig()
        if not self.config.validate():
            raise ValueError("Invalid configuration for summarization processor")
        
        self.entity_library = EntityLibrary() if self.config.create_entity_library else None
        logger.info(f"Initialized SummarizationProcessor with model: {self.config.model_name}")
    
    def process_clusters(self, clusters: List[Cluster], output_dir: Optional[str] = None) -> List[str]:
        """Process all clusters and generate Markdown files"""
        if not clusters:
            logger.warning("No clusters provided for summarization")
            return []
        
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Processing {len(clusters)} clusters for summarization")
        output_files = []
        
        for cluster in clusters:
            try:
                # Generate summary
                summary = self._generate_summary(cluster)
                
                # Extract entities if enabled
                entities = self._extract_entities(summary, cluster) if self.config.create_entity_library else []
                
                # Convert to Markdown
                markdown = self._convert_to_markdown(cluster, summary, entities)
                
                # Save to file
                file_path = self._save_markdown(markdown, cluster, output_dir)
                output_files.append(file_path)
                
                logger.info(f"Generated summary and markdown for cluster {cluster.cluster_id}")
            except Exception as e:
                logger.error(f"Error processing cluster {cluster.cluster_id}: {str(e)}")
        
        # Save entity library if enabled
        if self.config.create_entity_library and self.entity_library:
            entity_file = self.entity_library.save_to_file(output_dir)
            output_files.append(entity_file)
            logger.info(f"Generated entity library at {entity_file}")
        
        return output_files
    
    def _generate_summary(self, cluster: Cluster) -> str:
        """Generate a cohesive summary for the cluster using an NLP model"""
        # Collect all text from the cluster
        all_text = "\n\n".join([chunk.text for chunk in cluster.chunks])
        
        # Use prompt engineering to ensure summary stays faithful to original content
        prompt = f"""
Please create a concise summary of the following content. The summary should:
1. Reorganize and synthesize the information faithfully
2. NOT introduce any new facts or information not present in the original
3. Maintain the original tone and perspective
4. Focus on the main points and key details
5. Be coherent and well-structured

Content to summarize:
{all_text}
"""
        
        try:
            # Call API for summarization
            response = openai.ChatCompletion.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are a skilled summarizer. Your task is to create concise, accurate summaries that faithfully represent the original content without adding new information."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            
            # Extract and return the summary
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            # Fallback to a simple concatenation with a warning
            return f"[Error generating summary: {str(e)}]\n\n{all_text[:1000]}..."
    
    def _extract_entities(self, summary: str, cluster: Cluster) -> List[str]:
        """Extract key entities from the summary for cross-referencing"""
        try:
            # Use API to extract entities
            prompt = f"""
Extract the key entities (people, places, organizations, concepts, etc.) from the following text.
Return ONLY a comma-separated list of the most important 5-10 entities, with no other text or explanation.

Text:
{summary}
"""
            
            response = openai.ChatCompletion.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are an entity extraction specialist. Extract only the key entities as requested."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1,
            )
            
            # Process the response
            entities_text = response.choices[0].message.content.strip()
            entities = [e.strip() for e in entities_text.split(',')]
            
            # Add to entity library if available
            if self.entity_library:
                document_id = f"Cluster-{cluster.cluster_id}"
                for entity in entities:
                    self.entity_library.add_entity(entity, document_id, summary[:200] + "...")
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def _extract_topics(self, summary: str) -> List[str]:
        """Extract topics from the summary for tags"""
        try:
            # Use API to extract topics
            prompt = f"""
Extract {self.config.max_topics} key topics or themes from the following text.
Return ONLY a comma-separated list of single-word or short phrase topics, with no other text or explanation.
These will be used as tags, so keep them concise and relevant.

Text:
{summary}
"""
            
            response = openai.ChatCompletion.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are a topic extraction specialist. Extract only the key topics as requested."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1,
            )
            
            # Process the response
            topics_text = response.choices[0].message.content.strip()
            topics = [t.strip() for t in topics_text.split(',')]
            
            return topics[:self.config.max_topics]  # Limit to max_topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            return []
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to a human-readable format"""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        else:
            return f"{m:02d}:{s:02d}"
    
    def _convert_to_markdown(self, cluster: Cluster, summary: str, entities: List[str]) -> Dict[str, Any]:
        """Convert a summary to Obsidian-friendly Markdown"""
        # Extract title from the first line of the summary or use a default
        title_match = re.search(r'^#\s+(.+)$', summary, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
        else:
            # Extract a title from the first sentence
            first_sentence = re.split(r'[.!?]', summary.split('\n')[0])[0].strip()
            title = first_sentence[:50] + ('...' if len(first_sentence) > 50 else '')
        
        # Extract topics if enabled
        topics = self._extract_topics(summary) if self.config.extract_topics else []
        
        # Create the markdown content
        content = {}
        
        # Add YAML frontmatter if enabled
        if self.config.include_yaml_frontmatter:
            # Build frontmatter dictionary
            frontmatter = {
                "title": title,
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "cluster_id": cluster.cluster_id,
                "start_time": self._format_time(cluster.start_time) if self.config.include_timestamps else None,
                "end_time": self._format_time(cluster.end_time) if self.config.include_timestamps else None,
                "duration": self._format_time(cluster.duration) if self.config.include_timestamps else None,
                "tags": topics
            }
            
            # Remove None values
            frontmatter = {k: v for k, v in frontmatter.items() if v is not None}
            
            # Store frontmatter
            content["frontmatter"] = frontmatter
        
        # Add the title and summary
        content["title"] = title
        content["summary"] = summary
        
        # Add entity links if available
        if entities:
            entity_links = "\n## Related Entities\n\n" + "\n".join([f"- [[{entity}]]" for entity in entities])
            content["entity_links"] = entity_links
        
        return content
    
    def _save_markdown(self, markdown_content: Dict[str, Any], cluster: Cluster, output_dir: str) -> str:
        """Save the markdown content to a file"""
        # Create a filename based on the title
        safe_title = re.sub(r'[^\w\s-]', '', markdown_content["title"]).strip()
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        
        # Add cluster ID to ensure uniqueness
        filename = f"Cluster-{cluster.cluster_id}-{safe_title[:40]}.md"
        file_path = os.path.join(output_dir, filename)
        
        # Build the full markdown text
        markdown_text = ""
        
        # Add frontmatter if present
        if "frontmatter" in markdown_content:
            frontmatter = markdown_content["frontmatter"]
            frontmatter_yaml = yaml.dump(frontmatter)
            markdown_text += f"---\n{frontmatter_yaml}---\n\n"
        
        # Add title if not already in the summary (avoid duplication)
        if not markdown_content["summary"].startswith(f"# {markdown_content['title']}"):
            markdown_text += f"# {markdown_content['title']}\n\n"
        
        # Add the summary
        markdown_text += markdown_content["summary"]
        
        # Add entity links if present
        if "entity_links" in markdown_content:
            markdown_text += f"\n\n{markdown_content['entity_links']}"
        
        # Add a timestamp section if enabled
        if self.config.include_timestamps:
            markdown_text += f"\n\n## Timespan\n\n"
            markdown_text += f"- Start: {self._format_time(cluster.start_time)}\n"
            markdown_text += f"- End: {self._format_time(cluster.end_time)}\n"
            markdown_text += f"- Duration: {self._format_time(cluster.duration)}\n"
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        
        logger.info(f"Saved markdown to {file_path}")
        return file_path
    
    def get_mock_summary(self, cluster: Cluster) -> str:
        """Generate a mock summary for testing purposes"""
        all_text = "\n\n".join([chunk.text for chunk in cluster.chunks])
        paragraph_count = min(3, max(1, len(all_text) // 500))
        
        mock_summary = f"# Summary of Cluster {cluster.cluster_id}\n\n"
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', all_text)
        
        # Take a sample of sentences for each paragraph
        for i in range(paragraph_count):
            start_idx = i * len(sentences) // paragraph_count
            end_idx = min((i + 1) * len(sentences) // paragraph_count, len(sentences))
            sample = sentences[start_idx:end_idx]
            
            # Limit to first 5 sentences per paragraph
            sample = sample[:5]
            
            if sample:
                mock_summary += " ".join(sample) + "\n\n"
        
        return mock_summary
    
    def generate_sample_markdown(self, cluster: Cluster, output_dir: Optional[str] = None) -> str:
        """Generate a sample Markdown file for the given cluster"""
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a mock summary
        summary = self.get_mock_summary(cluster)
        
        # Extract mock entities
        entities = ["Entity1", "Entity2", "Entity3"]
        
        # Convert to Markdown
        markdown = self._convert_to_markdown(cluster, summary, entities)
        
        # Save to file
        return self._save_markdown(markdown, cluster, output_dir) 