
export interface FileMetadata {
  name: string;
  size: number;
  type: string;
  lastModified: number;
}

export type FileType = 'audio' | 'transcript';

export type ProcessingStage = 
  | 'idle' 
  | 'transcribing'
  | 'chunking'
  | 'clustering'
  | 'summarizing'
  | 'packaging'
  | 'complete'
  | 'error';

export interface ProcessingStatus {
  stage: ProcessingStage;
  progress: number; // 0-100
  message?: string;
  error?: string;
}

export interface ClusterData {
  id: string;
  title: string;
  summary: string;
  tags: string[];
  content: string; // Markdown content
}

export interface PreviewData {
  clusters: ClusterData[];
  entities: string[]; // Cross-reference entities
  folderStructure: FolderStructure;
}

export interface FolderStructure {
  name: string;
  type: 'folder' | 'file';
  children?: FolderStructure[];
}

export interface ProcessingOptions {
  fileType: FileType;
  advancedClustering: boolean;
  generateEntities: boolean;
  includeTimestamps: boolean;
}
