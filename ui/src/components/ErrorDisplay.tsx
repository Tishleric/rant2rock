
import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

interface ErrorDisplayProps {
  message: string;
  onReset: () => void;
}

const ErrorDisplay: React.FC<ErrorDisplayProps> = ({ message, onReset }) => {
  return (
    <div className="w-full py-8 animate-fade-in">
      <Alert variant="destructive" className="mb-6">
        <AlertCircle className="h-5 w-5" />
        <AlertTitle className="ml-2">Error Processing Your File</AlertTitle>
        <AlertDescription className="ml-7 mt-2">
          {message}
        </AlertDescription>
      </Alert>
      
      <div className="flex justify-center mt-6">
        <Button onClick={onReset} variant="outline" className="flex items-center gap-2">
          <RefreshCw className="h-4 w-4" />
          Try Again
        </Button>
      </div>
    </div>
  );
};

export default ErrorDisplay;
