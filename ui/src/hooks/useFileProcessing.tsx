/**
 * useFileProcessing.tsx
 * 
 * This hook manages file processing operations for Rant to Rock.
 * It handles file upload, status monitoring, and retrieval of processing results.
 * The hook connects to the backend API for all operations and provides a clean interface for the UI.
 */

import { useState, useCallback, useEffect } from 'react';
import { ProcessingStage, ProcessingStatus, ProcessingOptions, PreviewData, FileMetadata, FolderStructure, ClusterData } from '@/types';
import { toast } from 'sonner';

// Get the API base URL from environment variables
const baseURL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

// Main hook for file processing
export const useFileProcessing = () => {
  const [status, setStatus] = useState<ProcessingStatus>({
    stage: 'idle',
    progress: 0
  });
  
  const [fileMetadata, setFileMetadata] = useState<FileMetadata | null>(null);
  const [previewData, setPreviewData] = useState<PreviewData | null>(null);
  const [statusPollingInterval, setStatusPollingInterval] = useState<number | null>(null);
  
  // Poll for status updates when processing is active
  useEffect(() => {
    if (status.stage !== 'idle' && status.stage !== 'complete' && status.stage !== 'error') {
      // Start polling if not already polling
      if (!statusPollingInterval) {
        const interval = window.setInterval(async () => {
          try {
            const response = await fetch(`${baseURL}/api/status`);
            if (response.ok) {
              const newStatus = await response.json();
              setStatus(newStatus);
              
              // If processing is complete or errored, stop polling
              if (newStatus.stage === 'complete' || newStatus.stage === 'error') {
                if (newStatus.stage === 'complete') {
                  fetchPreviewData();
                  toast.success('Processing completed successfully', {
                    description: 'Your Obsidian bundle is ready for preview'
                  });
                } else if (newStatus.stage === 'error') {
                  toast.error('Processing failed', {
                    description: newStatus.message || 'There was an error processing your file'
                  });
                }
                
                if (statusPollingInterval) {
                  clearInterval(statusPollingInterval);
                  setStatusPollingInterval(null);
                }
              }
            }
          } catch (error) {
            console.error('Error polling status:', error);
          }
        }, 2000); // Poll every 2 seconds
        
        setStatusPollingInterval(interval);
        
        return () => {
          clearInterval(interval);
          setStatusPollingInterval(null);
        };
      }
    } else if (statusPollingInterval) {
      // Clean up polling if processing is complete or errored
      clearInterval(statusPollingInterval);
      setStatusPollingInterval(null);
    }
  }, [status.stage, statusPollingInterval]);
  
  // Function to fetch preview data when processing is complete
  const fetchPreviewData = async () => {
    try {
      // Fetch clusters
      const clustersResponse = await fetch(`${baseURL}/api/cluster`);
      if (!clustersResponse.ok) {
        throw new Error('Failed to fetch clusters');
      }
      const clustersData = await clustersResponse.json();
      
      // Fetch summaries
      const summariesResponse = await fetch(`${baseURL}/api/summarize`);
      if (!summariesResponse.ok) {
        throw new Error('Failed to fetch summaries');
      }
      const summariesData = await summariesResponse.json();
      
      // Extract entities from summaries (assuming entities are mentioned in summaries)
      const entitiesSet = new Set<string>();
      
      // Process summaries to create preview data
      const clusters: ClusterData[] = clustersData.map((cluster: any, index: number) => {
        // Find corresponding summary
        const summary = summariesData.summaries.find((s: any) => 
          s.filename.includes(cluster.id) || 
          s.filename.includes(cluster.title?.replace(/\s+/g, '_'))
        );
        
        // Extract entities from summary content
        const entityMatches = summary?.content.match(/\[\[(.*?)\]\]/g) || [];
        entityMatches.forEach((match: string) => {
          const entity = match.replace(/\[\[|\]\]/g, '');
          entitiesSet.add(entity);
        });
        
        return {
          id: cluster.id || String(index + 1),
          title: cluster.title || `Cluster ${index + 1}`,
          summary: cluster.summary || 'No summary available',
          tags: cluster.tags || [],
          content: summary ? summary.content.replace(/\n/g, '<br>') : '<p>Content not available</p>'
        };
      });
      
      // Convert Set to Array for entities
      const entities = Array.from(entitiesSet);
      
      // Create folder structure
      const folderStructure: FolderStructure = {
        name: 'Rant to Rock Export',
        type: 'folder',
        children: [
          {
            name: 'Topics',
            type: 'folder',
            children: clusters.map(cluster => ({
              name: `${cluster.title}.md`,
              type: 'file'
            }))
          },
          {
            name: 'Entities',
            type: 'folder',
            children: entities.slice(0, 10).map(entity => ({
              name: `${entity}.md`,
              type: 'file'
            }))
          },
          {
            name: 'Cross-References',
            type: 'folder',
            children: [
              { name: 'Topic Links.md', type: 'file' },
              { name: 'Entity References.md', type: 'file' }
            ]
          },
          {
            name: 'Original',
            type: 'folder',
            children: [
              { name: 'Full Transcript.md', type: 'file' },
              { name: 'Metadata.json', type: 'file' }
            ]
          },
          { name: 'README.md', type: 'file' }
        ]
      };
      
      // Set preview data
      setPreviewData({
        clusters,
        entities,
        folderStructure
      });
      
    } catch (error) {
      console.error('Error fetching preview data:', error);
      toast.error('Error loading preview', {
        description: 'Failed to load the processing results'
      });
    }
  };
  
  // Upload and process a file
  const processFile = useCallback(async (file: File, options: ProcessingOptions) => {
    // Reset state
    setStatus({ stage: 'idle', progress: 0 });
    setPreviewData(null);
    
    // Store file metadata
    setFileMetadata({
      name: file.name,
      size: file.size,
      type: file.type,
      lastModified: file.lastModified
    });
    
    try {
      // Create form data
      const formData = new FormData();
      formData.append('file', file);
      formData.append('options', JSON.stringify(options));
      
      // Upload file
      const response = await fetch(`${baseURL}/api/upload`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || 'Failed to upload file');
      }
      
      // Get initial status
      const statusResponse = await fetch(`${baseURL}/api/status`);
      if (statusResponse.ok) {
        const initialStatus = await statusResponse.json();
        setStatus(initialStatus);
      }
      
    } catch (error) {
      console.error('Processing error:', error);
      
      setStatus({ 
        stage: 'error', 
        progress: 0, 
        error: error instanceof Error ? error.message : 'Unknown error occurred',
        message: 'An error occurred during processing. Please try again.'
      });
      
      toast.error('Processing failed', {
        description: 'There was an error uploading your file'
      });
    }
  }, []);
  
  // Download the generated ZIP file
  const downloadZip = useCallback(async () => {
    try {
      toast.success('Download started', {
        description: 'Your Obsidian bundle is being downloaded'
      });
      
      // Create a link to download the file
      window.location.href = `${baseURL}/api/export/zip`;
      
    } catch (error) {
      console.error('Download error:', error);
      
      toast.error('Download failed', {
        description: 'There was an error downloading your file'
      });
    }
  }, []);
  
  // Reset the processing state
  const reset = useCallback(() => {
    setStatus({ stage: 'idle', progress: 0 });
    setFileMetadata(null);
    setPreviewData(null);
    
    if (statusPollingInterval) {
      clearInterval(statusPollingInterval);
      setStatusPollingInterval(null);
    }
  }, [statusPollingInterval]);
  
  return {
    status,
    fileMetadata,
    previewData,
    processFile,
    downloadZip,
    reset
  };
};
