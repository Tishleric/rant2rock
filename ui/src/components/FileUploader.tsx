/**
 * FileUploader.tsx
 * 
 * This component provides a user interface for uploading audio recordings or text transcripts.
 * It includes drag-and-drop functionality, file type validation, and processing options.
 * The component communicates with the backend API through the parent component's callbacks.
 */

import React, { useCallback, useState } from 'react';
import { FileText, Upload, Music, X, Settings, Info } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { FileType, ProcessingOptions } from '@/types';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

interface FileUploaderProps {
  onFileSelect: (file: File, options: ProcessingOptions) => void;
  isProcessing: boolean;
}

const FileUploader: React.FC<FileUploaderProps> = ({ onFileSelect, isProcessing }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileType, setFileType] = useState<FileType>('audio');
  const [showOptions, setShowOptions] = useState(false);
  const [options, setOptions] = useState<ProcessingOptions>({
    fileType: 'audio',
    advancedClustering: true,
    generateEntities: true,
    includeTimestamps: true,
  });

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      // Validate file type
      const isAudio = file.type.startsWith('audio/');
      const isText = file.type === 'text/plain' || 
                     file.type === 'application/json' || 
                     file.name.endsWith('.txt') || 
                     file.name.endsWith('.json');
      
      if ((fileType === 'audio' && isAudio) || (fileType === 'transcript' && isText)) {
        setSelectedFile(file);
      } else {
        // Show error for invalid file type
        console.error('Invalid file type');
      }
    }
  }, [fileType]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  }, []);

  const handleRemoveFile = useCallback(() => {
    setSelectedFile(null);
  }, []);

  const handleSubmit = useCallback(() => {
    if (selectedFile) {
      onFileSelect(selectedFile, { ...options, fileType });
    }
  }, [selectedFile, onFileSelect, options, fileType]);

  const handleOptionChange = useCallback((key: keyof ProcessingOptions, value: boolean) => {
    setOptions(prev => ({ ...prev, [key]: value }));
  }, []);
  
  const handleFileTypeChange = useCallback((type: FileType) => {
    setFileType(type);
    setOptions(prev => ({ ...prev, fileType: type }));
    // Reset selected file when changing type
    setSelectedFile(null);
  }, []);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' bytes';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="w-full space-y-4 animate-fade-in">
      <Tabs defaultValue="audio" onValueChange={(value) => handleFileTypeChange(value as FileType)}>
        <TabsList className="grid grid-cols-2 w-full max-w-md mx-auto">
          <TabsTrigger value="audio" disabled={isProcessing}>
            <Music className="w-4 h-4 mr-2" />
            Audio Recording
          </TabsTrigger>
          <TabsTrigger value="transcript" disabled={isProcessing}>
            <FileText className="w-4 h-4 mr-2" />
            Text Transcript
          </TabsTrigger>
        </TabsList>
      </Tabs>
      
      <div 
        className={cn(
          "relative border-2 border-dashed rounded-lg p-8 flex flex-col items-center justify-center transition-all",
          dragActive ? "border-primary bg-primary/5" : "border-border",
          selectedFile ? "bg-secondary/30" : "bg-transparent",
          isProcessing ? "opacity-50 cursor-not-allowed" : "cursor-pointer"
        )}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {selectedFile ? (
          <div className="flex flex-col items-center">
            <div className="h-16 w-16 rounded-lg bg-obsidian/10 flex items-center justify-center mb-4">
              {fileType === 'audio' ? (
                <Music className="h-8 w-8 text-obsidian" />
              ) : (
                <FileText className="h-8 w-8 text-obsidian" />
              )}
            </div>
            <p className="text-lg font-medium mb-1">{selectedFile.name}</p>
            <p className="text-sm text-muted-foreground">{formatFileSize(selectedFile.size)}</p>
            
            {!isProcessing && (
              <Button 
                variant="ghost" 
                size="sm" 
                className="mt-4" 
                onClick={handleRemoveFile}
              >
                <X className="h-4 w-4 mr-1" /> Remove
              </Button>
            )}
          </div>
        ) : (
          <>
            <div className="h-20 w-20 rounded-full bg-secondary flex items-center justify-center mb-6 animate-pulse-soft">
              <Upload className="h-10 w-10 text-obsidian" />
            </div>
            <p className="text-lg font-medium text-center mb-2">
              Drag & drop your {fileType === 'audio' ? 'audio file' : 'transcript'} here
            </p>
            <p className="text-sm text-muted-foreground text-center mb-6">
              {fileType === 'audio' 
                ? "Supported formats: MP3, WAV, M4A, FLAC" 
                : "Supported formats: TXT, JSON"}
            </p>
            
            <label htmlFor="file-upload" className="cursor-pointer">
              <div className="bg-primary hover:bg-primary/90 text-primary-foreground px-4 py-2 rounded-md transition-colors">
                Browse files
              </div>
              <input
                id="file-upload"
                type="file"
                accept={fileType === 'audio' ? 'audio/*' : '.txt,.json,text/plain,application/json'}
                className="hidden"
                onChange={handleFileInput}
                disabled={isProcessing}
              />
            </label>
          </>
        )}
      </div>
      
      {selectedFile && (
        <div className="animate-fade-in">
          <div className="flex items-center justify-between mb-4">
            <div className="flex">
              <Button
                variant="ghost"
                size="sm"
                className="flex items-center"
                onClick={() => setShowOptions(!showOptions)}
              >
                <Settings className="h-4 w-4 mr-2" />
                {showOptions ? "Hide options" : "Show options"}
              </Button>
            </div>
            
            <Button 
              onClick={handleSubmit}
              disabled={!selectedFile || isProcessing}
              className="bg-obsidian hover:bg-obsidian-dark"
            >
              Start Processing
            </Button>
          </div>
          
          {showOptions && (
            <div className="bg-secondary/50 p-4 rounded-lg space-y-3 animate-scale-in">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Label htmlFor="advanced-clustering">Advanced clustering</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        Uses both semantic and temporal data for optimal topic separation
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <Switch
                  id="advanced-clustering"
                  checked={options.advancedClustering}
                  onCheckedChange={(checked) => handleOptionChange('advancedClustering', checked)}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Label htmlFor="generate-entities">Generate entity links</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        Creates cross-reference links between related concepts
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <Switch
                  id="generate-entities"
                  checked={options.generateEntities}
                  onCheckedChange={(checked) => handleOptionChange('generateEntities', checked)}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Label htmlFor="include-timestamps">Include timestamps</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        Adds time markers to the generated markdown files
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <Switch
                  id="include-timestamps"
                  checked={options.includeTimestamps}
                  onCheckedChange={(checked) => handleOptionChange('includeTimestamps', checked)}
                  disabled={fileType === 'transcript'}
                />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FileUploader;
