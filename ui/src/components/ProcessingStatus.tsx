/**
 * ProcessingStatus.tsx
 * 
 * This component displays the current processing status of a file being processed.
 * It shows a progress bar and status indicators for each stage of processing.
 * The component visualizes the pipeline stages: transcribing, chunking, clustering, summarizing, and packaging.
 */

import React from 'react';
import { Check, Loader2, FileText, Scissors, Network, FileJson, Archive, AlertCircle } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import { ProcessingStage, ProcessingStatus as ProcessingStatusType } from '@/types';
import { cn } from '@/lib/utils';

interface ProcessingStatusProps {
  status: ProcessingStatusType;
}

const ProcessingStatus: React.FC<ProcessingStatusProps> = ({ status }) => {
  const stages: { stage: ProcessingStage; label: string; icon: React.ReactNode }[] = [
    { stage: 'transcribing', label: 'Transcribing', icon: <FileText className="h-5 w-5" /> },
    { stage: 'chunking', label: 'Chunking', icon: <Scissors className="h-5 w-5" /> },
    { stage: 'clustering', label: 'Clustering', icon: <Network className="h-5 w-5" /> },
    { stage: 'summarizing', label: 'Summarizing', icon: <FileJson className="h-5 w-5" /> },
    { stage: 'packaging', label: 'Packaging', icon: <Archive className="h-5 w-5" /> },
  ];

  const getStageStatus = (stageValue: ProcessingStage) => {
    if (status.stage === 'error') return 'stage-indicator-waiting';
    if (status.stage === stageValue) return 'stage-indicator-active';
    
    const stageIndex = stages.findIndex(s => s.stage === stageValue);
    const currentStageIndex = stages.findIndex(s => s.stage === status.stage);
    
    if (stageIndex < currentStageIndex || status.stage === 'complete') {
      return 'stage-indicator-completed';
    }
    
    return 'stage-indicator-waiting';
  };

  return (
    <div className="w-full space-y-6 py-6 animate-fade-in">
      <div className="relative">
        <Progress value={status.progress} className="h-2" />
        {status.stage !== 'idle' && (
          <div className="absolute -bottom-6 right-0 text-sm text-muted-foreground">
            {status.progress}%
          </div>
        )}
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mt-10">
        {stages.map(({ stage, label, icon }) => (
          <div 
            key={stage} 
            className={cn(
              "stage-indicator flex-col items-center justify-center text-center p-3 rounded-lg border transition-all",
              getStageStatus(stage),
              status.stage === stage ? "border-obsidian bg-obsidian/5" : "border-transparent"
            )}
          >
            <div className="flex items-center justify-center h-10 w-10 rounded-full bg-white border mb-2">
              {status.stage === stage ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : getStageStatus(stage).includes('completed') ? (
                <Check className="h-5 w-5" />
              ) : (
                icon
              )}
            </div>
            <div className="text-sm font-medium">{label}</div>
          </div>
        ))}
      </div>

      {status.message && (
        <div className="mt-6 text-center">
          <p className="text-sm font-medium">
            {status.stage === 'error' ? (
              <span className="flex items-center justify-center text-destructive">
                <AlertCircle className="h-4 w-4 mr-2" />
                {status.message}
              </span>
            ) : (
              status.message
            )}
          </p>
        </div>
      )}
    </div>
  );
};

export default ProcessingStatus;
