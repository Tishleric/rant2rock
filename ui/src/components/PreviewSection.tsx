/**
 * PreviewSection.tsx
 * 
 * This component displays a preview of the processed data, showing clusters, entities, and folder structure.
 * It allows users to navigate between different clusters, view their content, and download the final ZIP package.
 * The component visualizes the results of the backend processing pipeline for user review.
 */

import React, { useState } from 'react';
import { FolderTree, MessageSquare, Tag, FileDown, ChevronRight, ChevronDown, FileText, Folder } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { ClusterData, FolderStructure, PreviewData } from '@/types';
import { cn } from '@/lib/utils';

interface PreviewSectionProps {
  previewData: PreviewData;
  onDownload: () => void;
}

const PreviewSection: React.FC<PreviewSectionProps> = ({ previewData, onDownload }) => {
  const [selectedCluster, setSelectedCluster] = useState<ClusterData | null>(
    previewData.clusters.length > 0 ? previewData.clusters[0] : null
  );
  const [expandedFolders, setExpandedFolders] = useState<Record<string, boolean>>({
    'root': true
  });

  const toggleFolder = (path: string) => {
    setExpandedFolders(prev => ({
      ...prev,
      [path]: !prev[path]
    }));
  };

  const renderFolderStructure = (item: FolderStructure, path = 'root', depth = 0) => {
    const currentPath = `${path}/${item.name}`;
    const isExpanded = expandedFolders[currentPath] ?? false;
    
    return (
      <div key={currentPath} className="ml-4">
        <div 
          className={cn(
            "flex items-center py-1 hover:bg-secondary/50 rounded px-2 -ml-2 cursor-pointer",
            item.type === 'folder' ? "font-medium" : "text-muted-foreground"
          )}
          onClick={() => item.type === 'folder' ? toggleFolder(currentPath) : null}
        >
          {item.type === 'folder' ? (
            <>
              {isExpanded ? 
                <ChevronDown className="h-4 w-4 mr-1 text-muted-foreground" /> : 
                <ChevronRight className="h-4 w-4 mr-1 text-muted-foreground" />}
              <Folder className="h-4 w-4 mr-2 text-obsidian" />
            </>
          ) : (
            <>
              <div className="w-4 mr-1" />
              <FileText className="h-4 w-4 mr-2 text-muted-foreground" />
            </>
          )}
          <span className="text-sm truncate">{item.name}</span>
        </div>
        
        {item.type === 'folder' && isExpanded && item.children && (
          <div>
            {item.children.map(child => renderFolderStructure(child, currentPath, depth + 1))}
          </div>
        )}
      </div>
    );
  };
  
  return (
    <div className="w-full py-6 animate-fade-in">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-semibold">Preview Results</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Review your mind map structure before downloading
          </p>
        </div>
        <Button 
          onClick={onDownload}
          className="bg-obsidian hover:bg-obsidian-dark flex gap-2"
        >
          <FileDown className="h-5 w-5" />
          Download ZIP
        </Button>
      </div>
      
      <Tabs defaultValue="clusters" className="w-full">
        <TabsList className="grid grid-cols-3 max-w-md mb-4">
          <TabsTrigger value="clusters">
            <MessageSquare className="h-4 w-4 mr-2" />
            Clusters
          </TabsTrigger>
          <TabsTrigger value="structure">
            <FolderTree className="h-4 w-4 mr-2" />
            Structure
          </TabsTrigger>
          <TabsTrigger value="entities">
            <Tag className="h-4 w-4 mr-2" />
            Entities
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="clusters" className="mt-0">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="md:col-span-1 border shadow-sm">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg">Topic Clusters</CardTitle>
                <CardDescription>
                  {previewData.clusters.length} clusters generated
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <ScrollArea className="h-[400px] px-6">
                  <div className="space-y-1 pb-6">
                    {previewData.clusters.map((cluster, index) => (
                      <div
                        key={cluster.id}
                        className={cn(
                          "flex items-center py-2 px-3 rounded-md cursor-pointer transition-colors",
                          selectedCluster?.id === cluster.id 
                            ? "bg-obsidian text-white" 
                            : "hover:bg-secondary"
                        )}
                        onClick={() => setSelectedCluster(cluster)}
                      >
                        <div className="mr-3 flex h-8 w-8 items-center justify-center rounded-full bg-secondary/80">
                          {index + 1}
                        </div>
                        <div className="space-y-1">
                          <p className={cn(
                            "text-sm font-medium leading-none",
                            selectedCluster?.id === cluster.id 
                              ? "text-white" 
                              : ""
                          )}>
                            {cluster.title}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
            
            <Card className="md:col-span-2 border shadow-sm">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg">
                  {selectedCluster?.title || "Select a cluster"}
                </CardTitle>
                <div className="flex flex-wrap gap-2 mt-2">
                  {selectedCluster?.tags.map((tag) => (
                    <Badge key={tag} variant="secondary" className="bg-obsidian/10 text-obsidian-dark hover:bg-obsidian/20">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </CardHeader>
              <Separator />
              <CardContent className="pt-4">
                {selectedCluster ? (
                  <ScrollArea className="h-[400px] pr-4">
                    <div 
                      className="markdown-preview" 
                      dangerouslySetInnerHTML={{ __html: selectedCluster.content }}
                    />
                  </ScrollArea>
                ) : (
                  <div className="flex items-center justify-center h-[400px] text-muted-foreground">
                    Select a cluster to view details
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="structure">
          <Card className="border shadow-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Folder Structure</CardTitle>
              <CardDescription>
                Preview the Obsidian vault structure
              </CardDescription>
            </CardHeader>
            <Separator />
            <CardContent className="pt-4">
              <ScrollArea className="h-[500px]">
                <div className="space-y-2">
                  {renderFolderStructure(previewData.folderStructure)}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="entities">
          <Card className="border shadow-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Cross-Referenced Entities</CardTitle>
              <CardDescription>
                {previewData.entities.length} entities extracted
              </CardDescription>
            </CardHeader>
            <Separator />
            <CardContent className="pt-4">
              <ScrollArea className="h-[500px]">
                <div className="flex flex-wrap gap-2 pb-6">
                  {previewData.entities.map((entity) => (
                    <Badge key={entity} className="bg-obsidian/10 text-obsidian-dark py-1 px-3">
                      {entity}
                    </Badge>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default PreviewSection;
