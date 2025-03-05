/**
 * Index.tsx
 * 
 * This is the main page component for the Rant to Rock application.
 * It orchestrates the file upload, processing, and preview workflow.
 * The component integrates with the backend API through the useFileProcessing hook.
 */

import React, { useState } from 'react';
import { useFileProcessing } from '@/hooks/useFileProcessing';
import FileUploader from '@/components/FileUploader';
import ProcessingStatus from '@/components/ProcessingStatus';
import PreviewSection from '@/components/PreviewSection';
import ErrorDisplay from '@/components/ErrorDisplay';
import { Toaster } from "@/components/ui/sonner";
import { ProcessingOptions } from '@/types';

const Index = () => {
  const { status, previewData, processFile, downloadZip, reset } = useFileProcessing();
  
  const handleFileSelect = (file: File, options: ProcessingOptions) => {
    processFile(file, options);
  };
  
  const renderContent = () => {
    if (status.stage === 'error') {
      return <ErrorDisplay message={status.message || 'Unknown error'} onReset={reset} />;
    }
    
    if (status.stage === 'idle') {
      return (
        <div className="max-w-3xl mx-auto">
          <FileUploader onFileSelect={handleFileSelect} isProcessing={false} />
        </div>
      );
    }
    
    if (status.stage === 'complete' && previewData) {
      return (
        <>
          <ProcessingStatus status={status} />
          <PreviewSection previewData={previewData} onDownload={downloadZip} />
        </>
      );
    }
    
    return (
      <div className="max-w-3xl mx-auto">
        <FileUploader onFileSelect={handleFileSelect} isProcessing={true} />
        <ProcessingStatus status={status} />
      </div>
    );
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-secondary/30">
      <Toaster position="top-center" />
      <header className="py-12 text-center">
        <div className="inline-flex items-center justify-center p-1 mb-4 rounded-full bg-obsidian/10">
          <span className="feature-chip">Obsidian Companion</span>
        </div>
        <h1 className="text-4xl font-bold tracking-tight mb-2">Rant to Rock</h1>
        <p className="text-lg text-muted-foreground max-w-xl mx-auto">
          Transform your audio recordings into beautifully organized Obsidian mind maps
        </p>
      </header>
      
      <main className="container px-4 pb-20">
        {renderContent()}
      </main>
      
      <footer className="py-6 border-t bg-white">
        <div className="container px-4 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-sm text-muted-foreground">
            Rant to Rock – Obsidian Companion App
          </p>
          <div className="flex gap-4">
            <a href="#" className="text-sm text-muted-foreground hover:text-obsidian">
              Help
            </a>
            <a href="#" className="text-sm text-muted-foreground hover:text-obsidian">
              Privacy
            </a>
            <a href="#" className="text-sm text-muted-foreground hover:text-obsidian">
              Terms
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
