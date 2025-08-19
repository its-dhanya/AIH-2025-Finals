// (Only minimal edits to your original file â€" changed parts marked with // <-- CHANGED)
import React, { useState, useRef, useEffect, useCallback } from 'react';
import PDFViewer from '../components/pdf/PDFViewer';
import { useTheme } from '../context/ThemeContext';
import { Upload, FileText, X, Trash2, BookOpen, Search, ChevronLeft, ChevronRight, Maximize2, Minimize2, Folder, FolderPlus, ChevronDown, MoreVertical, ArrowLeft, ArrowRight } from 'lucide-react';
import {  Brain, Sparkles, TrendingUp, AlertTriangle, CheckCircle, RefreshCw, Mic} from 'lucide-react';
import ThemeToggle from '../components/ui/ThemeToggle';
import logo from '../assets/logo.png';
import PodcastPanel from '../components/PodcastPanel';

// --- Helper Function ---
const generateId = () => `id_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

// --- Custom scrollbar styles ---

const customScrollbarStyles = `
  .custom-scrollbar::-webkit-scrollbar { width: 6px; }
  .custom-scrollbar::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.1); border-radius: 3px; }
  .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.3); border-radius: 3px; }
  .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.5); }
  .smooth-hover { transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1); }
  .smooth-hover:hover { transform: translateY(-2px) scale(1.02); }
  .nav-item-hover { transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); }
  .nav-item-hover:hover { transform: translateX(8px); }
`;

// --- HELPER COMPONENTS ---
const ConnectionsPanel = ({
  connections = [],
  previousConnections = [],
  isLoading,
  theme,
  onClose,
  onConnectionClick,
  isShifted = false,
  isExpanded = true,
  onExpandToggle = () => {},
  width = 384,
  onStartResize = () => {},
  selectedText = '', 
  activeDoc = null 
}) => {
  const [expandedIds, setExpandedIds] = useState(new Set());
  
  // Insights state
  const [insights, setInsights] = useState(null);
  const [isGeneratingInsights, setIsGeneratingInsights] = useState(false);
  const [showInsights, setShowInsights] = useState(false);
  const [insightsError, setInsightsError] = useState(null);
  
  // View state - now includes 'podcast' as third option
  const [activeView, setActiveView] = useState('connections'); // 'connections', 'insights', or 'podcast'

  const toggleExpand = (id) => {
    setExpandedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const getIdForConn = (conn, index) => {
    return conn.id || conn.connection_id || `${(conn.document_name || 'doc')}_${index}`;
  };

  const makeSnippet = (text) => {
    if (!text) return '';
    const fullText = String(text).trim();
    const lines = fullText.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    if (lines.length >= 2) {
      return lines.slice(0, 2).join(' ');
    }
    const sentences = fullText.split(/(?<=\.)\s+/).map(s => s.trim()).filter(Boolean);
    if (sentences.length >= 2) {
      return sentences.slice(0, 2).join(' ');
    }
    return fullText.length > 220 ? fullText.slice(0, 220) + '...' : fullText;
  };

  // Generate insights function
  const generateInsights = async (connectionsData, selectedText) => {
    setIsGeneratingInsights(true);
    setInsightsError(null);
    
    try {
      const response = await fetch('http://localhost:3000/api/generate-insights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          selected_text: selectedText,
          connections: connectionsData,
          source_document: activeDoc?.fileName || activeDoc?.title || 'Current Document',
          context: {
            total_connections: connectionsData.length,
            documents: [...new Set(connectionsData.map(c => c.document_name))]
          }
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to generate insights: ${response.statusText}`);
      }

      const rawText = await response.text();
      try {
        const parsed = JSON.parse(rawText);
        setInsights(parsed.insights || parsed || { raw: rawText });
        setShowInsights(true);
        setActiveView('insights');
      } catch (e) {
        setInsights({ raw: rawText });
        setShowInsights(true);
        setActiveView('insights');
      }
    } catch (error) {
      console.error('Error generating insights:', error);
      setInsightsError(error.message);
      setShowInsights(true);
      setActiveView('insights');
    } finally {
      setIsGeneratingInsights(false);
    }
  };

  const panelRef = useRef(null);
  useEffect(() => {
    if (panelRef.current) panelRef.current.scrollTop = 0;
  }, [activeView, insights]);

  // Reset view and auto-refresh insights when new connections are loaded
  useEffect(() => {
    if (connections && connections.length > 0) {
      setActiveView('connections');
      
      if (showInsights) {
        generateInsights(connections, selectedText);
      }
    }
  }, [connections, selectedText]);

  const displayConnections = (connections && connections.length > 0) ? connections : (previousConnections || []);

  const handleResizeMouseDown = (e) => {
    e.preventDefault();
    onStartResize(e, 'connections');
  };

  // Updated ViewToggle with three options
  const ViewToggle = () => {
    if (!displayConnections.length) return null;
    
    return (
      <div className={`flex rounded-lg p-1 mb-4 ${
        theme === 'dark' ? 'bg-white/5 border border-white/10' : 'bg-gray-100 border border-gray-200'
      }`}>
        {/* Related Content Tab */}
        <button
          onClick={() => setActiveView('connections')}
          className={`flex-1 flex items-center justify-center gap-1.5 px-2 py-2 rounded-md text-xs font-semibold transition-all ${
            activeView === 'connections'
              ? theme === 'dark' 
                ? 'bg-cyan-500/20 text-cyan-300 shadow-sm' 
                : 'bg-white text-purple-700 shadow-sm'
              : theme === 'dark' 
                ? 'text-white/60 hover:text-white/80' 
                : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          <BookOpen size={12} />
          <span className="hidden sm:inline">Related</span>
          <span className={`px-1.5 py-0.5 rounded-full text-xs ${
            activeView === 'connections'
              ? theme === 'dark' ? 'bg-cyan-400/20 text-cyan-200' : 'bg-purple-100 text-purple-600'
              : theme === 'dark' ? 'bg-white/10 text-white/50' : 'bg-gray-200 text-gray-500'
          }`}>
            {displayConnections.length}
          </span>
        </button>
        
        {/* AI Insights Tab */}
        <button
          onClick={() => {
            if (showInsights) {
              setActiveView('insights');
            } else {
              generateInsights(displayConnections, selectedText);
            }
          }}
          disabled={isGeneratingInsights}
          className={`flex-1 flex items-center justify-center gap-1.5 px-2 py-2 rounded-md text-xs font-semibold transition-all ${
            activeView === 'insights'
              ? theme === 'dark' 
                ? 'bg-purple-500/20 text-purple-300 shadow-sm' 
                : 'bg-white text-purple-700 shadow-sm'
              : theme === 'dark' 
                ? 'text-white/60 hover:text-white/80' 
                : 'text-gray-600 hover:text-gray-900'
          } disabled:opacity-50`}
        >
          <Brain size={12} />
          <span className="hidden sm:inline">
            {isGeneratingInsights ? 'Loading...' : 'Insights'}
          </span>
          {showInsights && (
            <div className={`w-2 h-2 rounded-full ${
              theme === 'dark' ? 'bg-purple-400' : 'bg-purple-600'
            }`} />
          )}
        </button>

        {/* Podcast Tab */}
        <button
          onClick={() => setActiveView('podcast')}
          className={`flex-1 flex items-center justify-center gap-1.5 px-2 py-2 rounded-md text-xs font-semibold transition-all ${
            activeView === 'podcast'
              ? theme === 'dark' 
                ? 'bg-pink-500/20 text-pink-300 shadow-sm' 
                : 'bg-white text-pink-700 shadow-sm'
              : theme === 'dark' 
                ? 'text-white/60 hover:text-white/80' 
                : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          <Mic size={12} />
          <span className="hidden sm:inline">Podcast</span>
        </button>
      </div>
    );
  };

  const InsightCard = ({ insight, type, icon: Icon }) => (
    <div className={`p-4 rounded-lg border mb-3 ${
      theme === 'dark' 
        ? 'bg-gradient-to-br from-gray-800/50 to-gray-900/50 border-gray-700/50' 
        : 'bg-gradient-to-br from-white/80 to-gray-50/80 border-gray-300/50'
    } backdrop-blur-sm`}>
      <div className="flex items-start gap-3">
        <div className={`p-2 rounded-lg ${
          type === 'pattern' ? 'bg-blue-500/20 text-blue-400' :
          type === 'contradiction' ? 'bg-red-500/20 text-red-400' :
          type === 'example' ? 'bg-green-500/20 text-green-400' :
          'bg-purple-500/20 text-purple-400'
        }`}>
          <Icon size={16} />
        </div>
        <div className="flex-1">
          <h4 className={`font-semibold text-sm mb-1 ${
            theme === 'dark' ? 'text-white' : 'text-gray-900'
          }`}>
            {type === 'pattern' ? 'Pattern Identified' :
             type === 'contradiction' ? 'Contradictory Viewpoint' :
             type === 'example' ? 'Supporting Example' : 'Insight'}
          </h4>
          <p className={`text-sm leading-relaxed ${
            theme === 'dark' ? 'text-white/80' : 'text-gray-700'
          }`}>
            {typeof insight === 'string' ? insight : String(insight)}
          </p>
        </div>
      </div>
    </div>
  );

  const InsightsView = () => (
    <div className="space-y-4">
      {/* Header with refresh button */}
      <div className="flex justify-between items-center">
        <h3 className={`font-bold text-lg ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          AI Insights
        </h3>
        <button
          onClick={() => generateInsights(displayConnections, selectedText)}
          disabled={isGeneratingInsights}
          className={`p-1.5 rounded-lg transition-colors ${
            theme === 'dark' ? 'hover:bg-white/10 text-white/80' : 'hover:bg-gray-100 text-gray-700'
          } disabled:opacity-50`}
          title="Regenerate insights"
        >
          <RefreshCw size={16} className={isGeneratingInsights ? 'animate-spin' : ''} />
        </button>
      </div>

      {isGeneratingInsights ? (
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <div className="relative flex items-center justify-center mb-4">
            <div className="w-12 h-12 border-4 border-purple-400/20 rounded-full"></div>
            <div className="w-12 h-12 border-4 border-purple-500 border-t-transparent rounded-full animate-spin absolute"></div>
          </div>
          <p className={`font-semibold ${theme === 'dark' ? 'text-white/90' : 'text-gray-900'}`}>
            Generating Insights...
          </p>
          <p className={`text-sm ${theme === 'dark' ? 'text-white/70' : 'text-gray-700'}`}>
            Analyzing connections with AI
          </p>
        </div>
      ) : insightsError ? (
        <div className={`p-4 rounded-lg border-2 border-dashed ${
          theme === 'dark' ? 'border-red-500/30 bg-red-500/10' : 'border-red-400/30 bg-red-50'
        }`}>
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle size={16} className="text-red-500" />
            <span className={`font-semibold ${theme === 'dark' ? 'text-red-300' : 'text-red-700'}`}>
              Failed to generate insights
            </span>
          </div>
          <p className={`text-sm ${theme === 'dark' ? 'text-red-200' : 'text-red-600'}`}>
            {insightsError}
          </p>
        </div>
      ) : insights ? (
        <div>
          {insights.summary && (
            <div className={`p-4 rounded-lg mb-4 ${
              theme === 'dark' 
                ? 'bg-gradient-to-br from-purple-900/30 to-blue-900/30 border border-purple-500/30' 
                : 'bg-gradient-to-br from-purple-50 to-blue-50 border border-purple-200'
            }`}>
              <div className="flex items-center gap-2 mb-2">
                <Sparkles size={16} className={theme === 'dark' ? 'text-purple-400' : 'text-purple-600'} />
                <span className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                  Summary
                </span>
              </div>
              <p className={`text-sm leading-relaxed ${theme === 'dark' ? 'text-white/80' : 'text-gray-700'}`}>
                {typeof insights.summary === 'string' ? insights.summary : String(insights.summary)}
              </p>
            </div>
          )}

          {insights.patterns && Array.isArray(insights.patterns) && insights.patterns.map((pattern, index) => (
            <InsightCard
              key={`pattern-${index}`}
              insight={pattern}
              type="pattern"
              icon={TrendingUp}
            />
          ))}

          {insights.contradictions && Array.isArray(insights.contradictions) && insights.contradictions.map((contradiction, index) => (
            <InsightCard
              key={`contradiction-${index}`}
              insight={contradiction}
              type="contradiction"
              icon={AlertTriangle}
            />
          ))}

          {insights.examples && Array.isArray(insights.examples) && insights.examples.length > 0 && 
           insights.examples.filter(example => example && String(example).trim() !== '' && 
           !String(example).toLowerCase().includes('no explicit examples') &&
           !String(example).toLowerCase().includes('fallback used')).map((example, index) => (
            <InsightCard
              key={`example-${index}`}
              insight={example}
              type="example"
              icon={CheckCircle}
            />
          ))}

          {insights.additional_insights && Array.isArray(insights.additional_insights) && insights.additional_insights.map((insight, index) => (
            <InsightCard
              key={`additional-${index}`}
              insight={insight}
              type="additional"
              icon={Brain}
            />
          ))}

          {insights.raw && typeof insights.raw === 'string' && (
            <div className={`p-4 rounded-lg border ${
              theme === 'dark' 
                ? 'bg-gradient-to-br from-gray-800/50 to-gray-900/50 border-gray-700/50' 
                : 'bg-gradient-to-br from-white/80 to-gray-50/80 border-gray-300/50'
            } backdrop-blur-sm`}>
              <div className="flex items-center gap-2 mb-2">
                <Brain size={16} className={theme === 'dark' ? 'text-purple-400' : 'text-purple-600'} />
                <span className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                  Insights
                </span>
              </div>
              <p className={`text-sm leading-relaxed whitespace-pre-wrap ${theme === 'dark' ? 'text-white/80' : 'text-gray-700'}`}>
                {insights.raw}
              </p>
            </div>
          )}
        </div>
      ) : (
        <div className={`p-4 rounded-lg border-2 border-dashed text-center ${
          theme === 'dark' ? 'border-white/20 text-white/60' : 'border-gray-300 text-gray-500'
        }`}>
          <Brain size={24} className="mx-auto mb-2 opacity-50" />
          <p className="text-sm">
            No insights available. Click the refresh button to generate new insights.
          </p>
        </div>
      )}
    </div>
  );

  const ConnectionsView = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className={`font-bold text-lg ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          Related Content
        </h3>
        <span className={`text-xs px-2 py-1 rounded-full ${
          theme === 'dark' ? 'bg-cyan-500/20 text-cyan-300' : 'bg-purple-500/20 text-purple-700'
        }`}>
          {displayConnections.length} found
        </span>
      </div>

      <div className="space-y-3">
        {displayConnections.map((conn, index) => {
          const id = getIdForConn(conn, index);
          const fullText = conn.text || '';
          const snippet = makeSnippet(fullText);
          const isExpandedItem = expandedIds.has(id);
          const hasMore = fullText && fullText.length > snippet.length + 10;

          return (
            <div
              key={id}
              onClick={() => onConnectionClick(conn)}
              className={`p-3 rounded-lg border transition-all duration-200 cursor-pointer ${
                theme === 'dark' 
                  ? 'bg-white/5 border-white/10 hover:bg-pink-500/10 hover:border-pink-400/30' 
                  : 'bg-gray-50 border-gray-200 hover:bg-pink-100/50 hover:border-pink-400/30'
              }`}
            >
              <div className={`text-xs mb-1 truncate font-semibold ${
                theme === 'dark' ? 'text-cyan-300' : 'text-purple-600'
              }`}>
                {String((conn.document_name || conn.source || '')).replace(/\.pdf$/i, '')}
              </div>
              <div className={`font-semibold mb-2 text-sm ${theme === 'dark' ? 'text-white/90' : 'text-gray-900'}`}>
                {String(conn.section_title || conn.title || 'Section')}
              </div>
              <p className={`text-xs leading-relaxed ${
                theme === 'dark' ? 'text-white/70' : 'text-gray-700'
              }`}>
                {isExpandedItem ? fullText : snippet}
              </p>
              <div className="flex items-center justify-between mt-2">
                <div className={`text-xs ${theme === 'dark' ? 'text-white/50' : 'text-gray-500'}`}>
                  Page {conn.page_number}
                </div>
                {hasMore && (
                  <button
                    onClick={(e) => { e.stopPropagation(); toggleExpand(id); }}
                    className={`text-xs px-2 py-1 rounded transition-colors ${
                      theme === 'dark' ? 'text-cyan-300 hover:bg-cyan-500/20' : 'text-purple-600 hover:bg-purple-100'
                    }`}
                  >
                    {isExpandedItem ? 'Show less' : 'Read more'}
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );

  // New Podcast View
  const PodcastView = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className={`font-bold text-lg ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
          AI Podcast
        </h3>
      </div>
      
      <PodcastPanel
        insights={insights}
        selectedText={selectedText}
        connections={displayConnections}
        theme={theme}
        className=""
      />
    </div>
  );

  return (
    <div
      className={`fixed top-0 h-full ${isExpanded ? '' : ''} ${
        theme === 'dark' ? 'bg-black/90 border-white/20' : 'bg-white/95 border-gray-200/50'
      } backdrop-blur-xl border-l shadow-2xl z-50 transform transition-all duration-150 ease-in-out`}
      style={{
        width: isExpanded ? `${width}px` : '64px',
        right: 0,
      }}
    >
      {/* Header */}
      <div className={`p-4 border-b flex items-center justify-between ${
        theme === 'dark' ? 'border-white/10' : 'border-gray-200/30'
      }`}>
        <div className="flex items-center gap-3">
          <Search size={18} className={theme === 'dark' ? 'text-emerald-400' : 'text-purple-700'} />
          {isExpanded && (
            <div>
              <div className={`text-lg font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                Connections
              </div>
              <div className={`text-xs ${theme === 'dark' ? 'text-white/80' : 'text-gray-700'}`}>
                Related content + AI insights + Podcast
              </div>
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onExpandToggle}
            className={`p-2 rounded-lg transition-all duration-200 ${
              theme === 'dark' ? 'hover:bg-white/8 text-white/90' : 'hover:bg-gray-100 text-gray-900'
            }`}
            title={isExpanded ? 'Collapse' : 'Expand'}
          >
            {isExpanded ? <ChevronRight size={18} className="rotate-180" /> : <ChevronLeft size={18} />}
          </button>
          {isExpanded && (
            <button
              onClick={onClose}
              className={`p-2 rounded-lg transition-all duration-200 hover:scale-110 ${
                theme === 'dark' ? 'hover:bg-white/10 text-white/90' : 'hover:bg-gray-100 text-gray-900'
              }`}
            >
              <X size={18} />
            </button>
          )}
        </div>
      </div>

      {/* Body */}
      <div className="p-4 overflow-y-auto custom-scrollbar h-[calc(100vh-120px)]" ref={panelRef}>
        {isLoading ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="relative flex items-center justify-center">
              <div className="w-16 h-16 border-4 border-cyan-400/20 rounded-full"></div>
              <div className="w-16 h-16 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin absolute"></div>
            </div>
            <p className={`mt-4 text-lg font-semibold ${theme === 'dark' ? 'text-white/90' : 'text-gray-900'}`}>
              Finding Connections...
            </p>
            <p className={`text-sm ${theme === 'dark' ? 'text-white/70' : 'text-gray-700'}`}>
              Analyzing your library for relevant content.
            </p>
          </div>
        ) : displayConnections && displayConnections.length > 0 ? (
          <>
            {/* View Toggle */}
            <ViewToggle />
            
            {/* Content based on active view */}
            {activeView === 'connections' && <ConnectionsView />}
            {activeView === 'insights' && <InsightsView />}
            {activeView === 'podcast' && <PodcastView />}
          </>
        ) : (
          <div className="text-center py-8">
            <Search size={32} className={`mx-auto mb-4 ${theme === 'dark' ? 'text-white/30' : 'text-gray-400'}`} />
            <p className={`text-sm ${theme === 'dark' ? 'text-white/80' : 'text-gray-700'}`}>
              Select content to view connections, AI-generated insights, and create podcasts.
            </p>
          </div>
        )}
      </div>

      {/* Resize handle */}
      {isExpanded && (
        <div
          role="separator"
          onMouseDown={handleResizeMouseDown}
          onTouchStart={(e) => onStartResize(e, 'connections')}
          style={{
            position: 'absolute',
            left: '-8px',
            top: 0,
            bottom: 0,
            width: '16px',
            cursor: 'col-resize',
            zIndex: 60,
          }}
        />
      )}
    </div>
  );
};

const DocumentThumbnail = ({ filePromise, fileName, className = "" }) => {
  const [thumbnailUrl, setThumbnailUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const canvasRef = useRef(null);

  useEffect(() => {
    let isMounted = true;
    const generateThumbnail = async () => {
      try {
        if (!filePromise || !window.pdfjsLib) return;
        const arrayBuffer = await filePromise;
        const pdf = await window.pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        const page = await pdf.getPage(1);
        if (!isMounted || !canvasRef.current) return;
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        const viewport = page.getViewport({ scale: 0.3 });
        canvas.height = viewport.height;
        canvas.width = viewport.width;
        await page.render({ canvasContext: context, viewport }).promise;
        if (isMounted) {
          setThumbnailUrl(canvas.toDataURL());
          setLoading(false);
        }
      } catch (error) {
        console.error('Error generating thumbnail:', error);
        if (isMounted) setLoading(false);
      }
    };

    if (!window.pdfjsLib) {
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.11.338/pdf.min.js';
      script.onload = () => {
        if (window.pdfjsLib) {
          window.pdfjsLib.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.js';
        }
        generateThumbnail();
      };
      script.onerror = () => {
        console.error('Failed to load pdf.js');
        if (isMounted) setLoading(false);
      };
      document.head.appendChild(script);
    } else {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.js';
      generateThumbnail();
    }

    return () => { isMounted = false; };
  }, [filePromise]);

  return (
    <div className={`relative ${className}`}>
      <canvas ref={canvasRef} className="hidden" />
      {loading ? (
        <div className="w-12 h-16 bg-gradient-to-br from-gray-200 to-gray-300 dark:from-gray-700 dark:to-gray-800 rounded border animate-pulse flex items-center justify-center">
          <FileText size={16} className="text-gray-400" />
        </div>
      ) : thumbnailUrl ? (
        <img src={thumbnailUrl} alt={`${fileName} thumbnail`} className="w-12 h-16 object-cover rounded border border-white/20 shadow-sm" />
      ) : (
        <div className="w-12 h-16 bg-gradient-to-br from-gray-200 to-gray-300 dark:from-gray-700 dark:to-gray-800 rounded border flex items-center justify-center">
          <FileText size={16} className="text-gray-500" />
        </div>
      )}
    </div>
  );
};

const VantaBackground = ({ theme }) => {
  const vantaRef = useRef(null);
  const vantaEffectRef = useRef(null);

  useEffect(() => {
    const loadScript = (src) =>
      new Promise((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) return resolve();
        const script = document.createElement('script');
        script.src = src;
        script.async = true;
        script.onload = resolve;
        script.onerror = reject;
        document.body.appendChild(script);
      });

    let mounted = true;

    loadScript('https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js')
      .then(() => loadScript('https://cdnjs.cloudflare.com/ajax/libs/vanta/0.5.24/vanta.net.min.js'))
      .then(() => {
        if (!mounted || !window.VANTA || !vantaRef.current || vantaEffectRef.current) return;
        vantaEffectRef.current = window.VANTA.NET({
          el: vantaRef.current,
          mouseControls: true,
          touchControls: true,
          gyroControls: false,
          minHeight: 200.0,
          minWidth: 200.0,
          scale: 1.0,
          scaleMobile: 1.0,
          color: 0x6b21a8,
          backgroundColor: theme === 'dark' ? 0x0 : 0xfafafa,
          points: 12.0,
          maxDistance: 25.0,
          spacing: 18.0,
        });
      })
      .catch((err) => console.error('Failed to load Vanta/Three scripts', err));

    return () => {
      mounted = false;
      if (vantaEffectRef.current) {
        vantaEffectRef.current.destroy();
        vantaEffectRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (vantaEffectRef.current) {
      try {
        vantaEffectRef.current.setOptions({ backgroundColor: theme === 'dark' ? 0x0 : 0xfafafa });
      } catch (error) {
        console.error('Failed to update Vanta options:', error);
      }
    }
  }, [theme]);

  return <div ref={vantaRef} className="fixed inset-0 w-full h-full pointer-events-none z-0" />;
};

const MoveMenu = ({ isOpen, onClose, onMoveToCollection, onMakeIndividual, collections, currentCollectionId, theme }) => {
  const menuRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return;
    const handleClickOutside = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const availableCollections = collections.filter(col => col.id !== currentCollectionId);

  return (
    <div ref={menuRef} className={`absolute top-10 right-3 z-50 ${theme === 'dark' ? 'bg-gray-800 border-gray-600' : 'bg-white border-gray-200'} border rounded-lg shadow-xl py-1 min-w-[180px]`}>
      {currentCollectionId && (
        <button onClick={onMakeIndividual} className={`w-full text-left px-4 py-2 text-sm flex items-center gap-3 ${theme === 'dark' ? 'hover:bg-gray-700 text-white' : 'hover:bg-gray-100 text-gray-900'} transition-colors`}>
          <FileText size={14} /> Make Individual
        </button>
      )}
      {availableCollections.length > 0 && (
        <>
          <div className={`px-4 py-2 text-xs font-semibold ${theme === 'dark' ? 'text-gray-400 border-gray-600' : 'text-gray-500 border-gray-200'} ${currentCollectionId ? 'border-t' : ''}`}>
            Move to:
          </div>
          {availableCollections.map(collection => (
            <button key={collection.id} onClick={() => onMoveToCollection(collection.id)} className={`w-full text-left px-4 py-2 text-sm flex items-center gap-3 ${theme === 'dark' ? 'hover:bg-gray-700 text-white' : 'hover:bg-gray-100 text-gray-900'} transition-colors`}>
              <Folder size={14} /> {collection.name}
            </button>
          ))}
        </>
      )}
    </div>
  );
};

const DocumentItem = ({ doc, activeDoc, onSelect, onDelete, onMove, collections, currentCollectionId, theme }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <div
      className={`group relative p-4 rounded-xl border cursor-pointer transition-all duration-300 hover:shadow-lg ${
        activeDoc?.doc_id === doc.doc_id
          ? theme === 'dark'
            ? 'bg-gradient-to-br from-emerald-500/20 via-cyan-500/20 to-blue-500/20 border-emerald-400/60 shadow-xl shadow-emerald-500/10 ring-1 ring-emerald-400/30'
            : 'bg-gradient-to-br from-purple-500/20 via-pink-500/20 to-indigo-500/20 border-purple-400/60 shadow-xl shadow-purple-500/10 ring-1 ring-purple-400/30'
          : theme === 'dark'
            ? 'bg-white/5 border-white/10 hover:bg-gradient-to-br hover:from-orange-500/10 hover:to-purple-500/10 hover:border-orange-400/40 hover:shadow-orange-500/5'
            : 'bg-white/80 border-gray-200/50 hover:bg-gradient-to-br hover:from-purple-200/30 hover:to-indigo-200/30 hover:border-purple-400/50 hover:shadow-purple-500/10'
      } transform hover:scale-[1.02] hover:-translate-y-1`}
      onClick={() => onSelect(doc)}
    >
      <div className="flex items-center gap-4">
        <div className="shrink-0 relative">
          <DocumentThumbnail 
            filePromise={doc.filePromise} 
            fileName={doc.fileName} 
            className="transition-transform duration-300 group-hover:scale-110" 
          />
          <div className={`absolute -top-1 -right-1 w-3 h-3 rounded-full transition-all duration-200 ${
            activeDoc?.doc_id === doc.doc_id 
              ? theme === 'dark' ? 'bg-emerald-400 shadow-lg shadow-emerald-400/50' : 'bg-purple-500 shadow-lg shadow-purple-500/50'
              : 'opacity-0 group-hover:opacity-100'
          }`} />
        </div>
        
        <div className="flex-1 min-w-0">
          <h3 className={`font-bold text-base truncate mb-1 transition-colors ${
            theme === 'dark' ? 'text-white group-hover:text-emerald-300' : 'text-gray-900 group-hover:text-purple-700'
          }`}>
            {doc.title}
          </h3>
          <div className="flex items-center gap-2">
            <span className={`text-xs px-2 py-1 rounded-full font-medium ${
              theme === 'dark' 
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' 
                : 'bg-purple-500/20 text-purple-700 border border-purple-500/30'
            }`}>
              PDF Document
            </span>
            <span className={`text-xs ${theme === 'dark' ? 'text-white/60' : 'text-gray-500'}`}>
              • Recently added
            </span>
          </div>
        </div>
      </div>
      
      {/* Action buttons with better styling */}
      <div className="absolute top-3 right-3 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all duration-200">
        <button 
          onClick={(e) => { e.stopPropagation(); setIsMenuOpen(true); }} 
          className={`p-2 rounded-lg backdrop-blur-sm transition-all duration-200 hover:scale-110 ${
            theme === 'dark' 
              ? 'bg-white/10 hover:bg-white/20 text-white/80 hover:text-white border border-white/10' 
              : 'bg-black/10 hover:bg-black/20 text-gray-700 hover:text-gray-900 border border-black/10'
          }`} 
          title="Document actions"
        >
          <MoreVertical size={14} />
        </button>
        <button 
          onClick={(e) => { e.stopPropagation(); onDelete(doc.doc_id); }} 
          className={`p-2 rounded-lg backdrop-blur-sm transition-all duration-200 hover:scale-110 ${
            theme === 'dark' 
              ? 'bg-red-500/20 hover:bg-red-500/30 text-red-300 hover:text-red-200 border border-red-500/30' 
              : 'bg-red-500/10 hover:bg-red-500/20 text-red-600 hover:text-red-700 border border-red-500/20'
          }`} 
          title="Delete document"
        >
          <X size={14} />
        </button>
      </div>
      
      <MoveMenu 
        isOpen={isMenuOpen} 
        onClose={() => setIsMenuOpen(false)} 
        onMoveToCollection={(targetCollectionId) => onMove(doc.doc_id, currentCollectionId, targetCollectionId)} 
        onMakeIndividual={() => onMove(doc.doc_id, currentCollectionId, null)} 
        collections={collections} 
        currentCollectionId={currentCollectionId} 
        theme={theme} 
      />
    </div>
  );
};

const CollectionItem = ({ collection, activeDoc, onSelect, onDeleteDoc, onMoveDocument, onDeleteCollection, collections, theme }) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [isHovered, setIsHovered] = useState(false);
  
  return (
    <div 
      className={`rounded-xl border transition-all duration-300 overflow-hidden ${
        theme === 'dark' 
          ? 'bg-gradient-to-br from-white/5 to-white/10 border-white/10 hover:border-cyan-400/30 hover:shadow-lg hover:shadow-cyan-500/10' 
          : 'bg-gradient-to-br from-white/80 to-gray-50/80 border-gray-200/50 hover:border-purple-400/30 hover:shadow-lg hover:shadow-purple-500/10'
      }`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="p-4 border-b border-white/5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <div className={`p-2 rounded-lg transition-all duration-200 ${
              theme === 'dark' 
                ? 'bg-cyan-500/20 text-cyan-300' 
                : 'bg-purple-500/20 text-purple-600'
            } ${isHovered ? 'scale-110' : ''}`}>
              <Folder size={20} />
            </div>
            <div className="flex-1 min-w-0">
              <span className={`font-bold text-lg truncate block ${
                theme === 'dark' ? 'text-white' : 'text-gray-900'
              }`}>
                {collection.name}
              </span>
              <span className={`text-sm ${
                theme === 'dark' ? 'text-white/60' : 'text-gray-600'
              }`}>
                {collection.docs.length} document{collection.docs.length !== 1 ? 's' : ''}
              </span>
            </div>
            <div className={`px-3 py-1 rounded-full text-sm font-semibold ${
              theme === 'dark' 
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' 
                : 'bg-purple-500/20 text-purple-700 border border-purple-500/30'
            }`}>
              {collection.docs.length}
            </div>
          </div>
          
          <div className="flex items-center gap-1">
            <button 
              onClick={(e) => { e.stopPropagation(); onDeleteCollection(collection.id, collection.name); }} 
              className={`p-2 rounded-lg transition-all duration-200 hover:scale-110 ${
                theme === 'dark' 
                  ? 'hover:bg-red-500/20 text-white/60 hover:text-red-300' 
                  : 'hover:bg-red-500/10 text-gray-600 hover:text-red-600'
              }`}
            >
              <Trash2 size={16} />
            </button>
            <button 
              onClick={() => setIsExpanded(!isExpanded)} 
              className={`p-2 rounded-lg transition-all duration-300 ${
                theme === 'dark' 
                  ? 'hover:bg-white/10 text-white/80' 
                  : 'hover:bg-black/10 text-gray-700'
              }`} 
              style={{ transform: isExpanded ? 'rotate(0deg)' : 'rotate(-90deg)' }}
            >
              <ChevronDown size={18} />
            </button>
          </div>
        </div>
      </div>
      
      <div className={`transition-all duration-300 ease-in-out overflow-hidden ${
        isExpanded ? 'max-h-[2000px] opacity-100' : 'max-h-0 opacity-0'
      }`}>
        <div className="p-4 space-y-3">
          {collection.docs.length > 0 ? (
            collection.docs.map(doc => (
              <DocumentItem 
                key={doc.doc_id} 
                doc={doc} 
                activeDoc={activeDoc} 
                onSelect={onSelect} 
                onDelete={(docId) => onDeleteDoc(docId, collection.id)} 
                onMove={onMoveDocument} 
                collections={collections} 
                currentCollectionId={collection.id} 
                theme={theme} 
              />
            ))
          ) : (
            <div className={`text-center py-8 px-4 rounded-lg border-2 border-dashed ${
              theme === 'dark' 
                ? 'border-white/20 text-white/50 bg-white/5' 
                : 'border-gray-300 text-gray-500 bg-gray-50/50'
            }`}>
              <FileText size={32} className="mx-auto mb-3 opacity-50" />
              <p className="text-sm font-medium">This collection is empty</p>
              <p className="text-xs mt-1 opacity-80">Drag documents here or use the move menu</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
const UploadLoadingModal = ({ isOpen, progress, theme }) => {
  if (!isOpen) return null;
  
  const percentage = progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0;
  
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-md flex items-center justify-center z-[70] p-4">
      <div className={`relative rounded-2xl p-8 border ${
        theme === 'dark' ? 'bg-[#101010] border-white/20' : 'bg-white border-gray-200'
      } w-full max-w-md shadow-2xl`}>
        <div className="text-center">
          <div className="relative flex items-center justify-center mb-6">
            <div className="w-20 h-20 border-4 border-cyan-400/20 rounded-full"></div>
            <div className="w-20 h-20 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin absolute"></div>
            <Upload size={24} className="text-cyan-500 absolute" />
          </div>
          
          <h2 className={`text-2xl font-bold mb-2 ${
            theme === 'dark' ? 'text-white' : 'text-gray-900'
          }`}>
            Uploading Documents
          </h2>
          
          <p className={`mb-6 ${
            theme === 'dark' ? 'text-white/70' : 'text-gray-600'
          }`}>
            Processing {progress.current} of {progress.total} documents...
          </p>
          
          {/* Progress bar */}
          <div className={`w-full h-2 rounded-full mb-4 overflow-hidden ${
            theme === 'dark' ? 'bg-white/10' : 'bg-gray-200'
          }`}>
            <div 
              className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${percentage}%` }}
            />
          </div>
          
          <div className={`text-sm font-medium ${
            theme === 'dark' ? 'text-cyan-300' : 'text-cyan-600'
          }`}>
            {percentage}% Complete
          </div>
        </div>
      </div>
    </div>
  );
};
const DocumentList = ({ items, activeDoc, onSelect, onDeleteDoc, onMoveDocument, onCreateCollection, onDeleteCollection, isCollapsed, onToggle, theme }) => {
  const docCount = items.reduce((acc, item) => acc + (item.type === 'doc' ? 1 : (item.docs ? item.docs.length : 0)), 0);
  const collections = items.filter(item => item.type === 'collection');

  return (
    <div className={`transition-all duration-300 ease-in-out ${isCollapsed ? 'w-20' : 'w-80'} h-full bg-white/10 dark:bg-black/50 backdrop-blur-xl border-r border-white/20 flex flex-col z-10 shrink-0`}>
      {/* Enhanced Header */}
      <div className={`p-4 border-b border-white/10 ${
        theme === 'dark' ? 'bg-gradient-to-r from-emerald-500/10 to-cyan-500/10' : 'bg-gradient-to-r from-purple-500/10 to-pink-500/10'
      }`}>
        <div className="flex items-center justify-between">
          {!isCollapsed && (
            <div>
              <h2 className={`text-2xl font-bold mb-1 ${
                theme === 'dark' 
                  ? 'bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-500 bg-clip-text text-transparent' 
                  : 'bg-gradient-to-r from-purple-700 via-pink-600 to-indigo-700 bg-clip-text text-transparent'
              }`}>
                Library
              </h2>
              <div className="flex items-center gap-2">
                <span className={`text-sm ${theme === 'dark' ? 'text-white/80' : 'text-gray-800'}`}>
                  {docCount} document{docCount !== 1 ? 's' : ''}
                </span>
                {collections.length > 0 && (
                  <>
                    <span className={`text-sm ${theme === 'dark' ? 'text-white/50' : 'text-gray-500'}`}>•</span>
                    <span className={`text-sm ${theme === 'dark' ? 'text-white/80' : 'text-gray-800'}`}>
                      {collections.length} collection{collections.length !== 1 ? 's' : ''}
                    </span>
                  </>
                )}
              </div>
            </div>
          )}
          <button 
            onClick={onToggle} 
            className={`p-2 rounded-lg transition-all duration-200 ml-auto hover:scale-110 ${
              theme === 'dark' 
                ? 'hover:bg-white/10 text-white/90 hover:text-white' 
                : 'hover:bg-gray-200/50 text-gray-900 hover:text-black'
            }`}
          >
            {isCollapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
          </button>
        </div>
      </div>

      {/* Content */}
      {isCollapsed ? (
        <div className="flex-1 p-2 space-y-3 overflow-y-auto custom-scrollbar">
          {items.map(item =>
            item.type === 'collection' ? (
              <button 
                key={item.id} 
                onClick={onToggle} 
                title={`${item.name} (${item.docs.length} docs)`} 
                className={`w-16 h-16 flex items-center justify-center rounded-xl transition-all duration-200 relative group ${
                  theme === 'dark' ? 'bg-black/20 hover:bg-white/10' : 'bg-black/5 hover:bg-black/10'
                } hover:scale-110`}
              >
                <Folder size={28} className={theme === 'dark' ? 'text-cyan-300' : 'text-purple-600'} />
                {item.docs.length > 0 && (
                  <span className={`absolute -top-1 -right-1 w-5 h-5 text-xs font-bold rounded-full flex items-center justify-center shadow-lg ${
                    theme === 'dark' ? 'bg-cyan-500 text-white' : 'bg-purple-500 text-white'
                  }`}>
                    {item.docs.length}
                  </span>
                )}
              </button>
            ) : (
              <button 
                key={item.doc_id} 
                onClick={() => onSelect(item)} 
                title={item.title} 
                className={`relative group transition-all duration-300 rounded-lg overflow-hidden hover:scale-110 ${
                  activeDoc?.doc_id === item.doc_id 
                    ? theme === 'dark' ? 'ring-2 ring-emerald-400' : 'ring-2 ring-purple-500'
                    : ''
                }`}
              >
                <DocumentThumbnail filePromise={item.filePromise} fileName={item.fileName} />
              </button>
            )
          )}
        </div>
      ) : (
        <div className="flex-1 flex flex-col h-full overflow-hidden">
          {items.length > 0 ? (
            <>
              <div className="flex-1 p-4 overflow-y-auto custom-scrollbar space-y-4">
                {items.map(item =>
                  item.type === 'collection' ? (
                    <CollectionItem 
                      key={item.id} 
                      collection={item} 
                      activeDoc={activeDoc} 
                      onSelect={onSelect} 
                      onDeleteDoc={onDeleteDoc} 
                      onMoveDocument={onMoveDocument} 
                      onDeleteCollection={onDeleteCollection} 
                      collections={collections} 
                      theme={theme} 
                    />
                  ) : (
                    <DocumentItem 
                      key={item.doc_id} 
                      doc={item} 
                      activeDoc={activeDoc} 
                      onSelect={onSelect} 
                      onDelete={(docId) => onDeleteDoc(docId)} 
                      onMove={onMoveDocument} 
                      collections={collections} 
                      currentCollectionId={null} 
                      theme={theme} 
                    />
                  )
                )}
              </div>
              
              {/* Enhanced Create Collection Button */}
              <div className="p-4 border-t border-white/10 mt-auto">
                <button 
                  onClick={onCreateCollection} 
                  className={`w-full flex items-center justify-center gap-2 p-4 text-sm font-semibold rounded-xl transition-all duration-300 border-2 border-dashed hover:scale-105 ${
                    theme === 'dark'
                      ? 'bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 hover:text-cyan-200 border-cyan-500/30 hover:border-cyan-400/50 shadow-lg hover:shadow-cyan-500/20'
                      : 'bg-purple-500/10 hover:bg-purple-500/20 text-purple-600 hover:text-purple-700 border-purple-500/30 hover:border-purple-400/50 shadow-lg hover:shadow-purple-500/20'
                  }`}
                  title="Create a new collection"
                >
                  <FolderPlus size={16} />
                  Create Collection
                </button>
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-full p-8 text-center">
              <div className={`w-24 h-24 rounded-3xl flex items-center justify-center mb-8 border-2 border-dashed transition-all duration-300 ${
                theme === 'dark'
                  ? 'bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 border-emerald-400/30'
                  : 'bg-gradient-to-br from-purple-500/20 to-pink-500/20 border-purple-400/30'
              }`}>
                <FileText size={40} className={theme === 'dark' ? 'text-emerald-400' : 'text-purple-700'} />
              </div>
              <h2 className={`text-2xl font-bold mb-3 ${
                theme === 'dark' 
                  ? 'bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-500 bg-clip-text text-transparent' 
                  : 'bg-gradient-to-r from-purple-700 via-pink-600 to-indigo-700 bg-clip-text text-transparent'
              }`}>
                Your Digital Library
              </h2>
              <p className={`leading-relaxed text-sm max-w-xs mb-6 ${
                theme === 'dark' ? 'text-white/80' : 'text-gray-800'
              }`}>
                Upload PDFs to build your library and start discovering connections between your documents.
              </p>
              <div className={`text-xs px-3 py-2 rounded-full ${
                theme === 'dark' 
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' 
                  : 'bg-purple-500/20 text-purple-700 border border-purple-500/30'
              }`}>
                Drag & drop or click Upload
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
const NavigationPanel = ({
  outline,
  onSectionClick,
  isVisible,
  onToggle,
  theme,
  isExpanded = true,
  onExpandToggle = () => {},
  width = 384,
  offsetRight = 0,
  onStartResize = () => {}
}) => { // <-- CHANGED: accept width, offsetRight, and onStartResize for drag-resize
  if (!isVisible) return null;

  const handleResizeMouseDown = (e) => {
    e.preventDefault();
    onStartResize(e, 'nav');
  };

  return (
    <div
      className={`fixed top-0 h-full ${theme === 'dark' ? 'bg-black/90 border-white/20' : 'bg-white/95 border-gray-200/50'} backdrop-blur-xl border-l shadow-2xl z-50 transform transition-all duration-150 ease-in-out`}
      style={{
        width: isExpanded ? `${width}px` : '64px',
        right: offsetRight ? `${offsetRight}px` : '0px'
      }}
    >
      <div className={`p-4 border-b flex items-center justify-between ${theme === 'dark' ? 'border-white/10' : 'border-gray-200/30'}`}>
        <div className="flex items-center gap-3">
          <BookOpen size={18} className={theme === 'dark' ? 'text-cyan-400' : 'text-violet-700'} />
          {isExpanded && (
            <div>
              <div className="text-lg font-bold">Contents</div>
              <div className={`text-xs ${theme === 'dark' ? 'text-white/90' : 'text-gray-900'}`}>Navigate through sections</div>
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button onClick={onExpandToggle} className={`p-2 rounded-lg transition-all duration-200 ${theme === 'dark' ? 'hover:bg-white/8 text-white/90' : 'hover:bg-gray-100 text-gray-900'}`} title={isExpanded ? 'Collapse' : 'Expand'}>
            {isExpanded ? <ChevronRight size={18} className="rotate-180" /> : <ChevronLeft size={18} />}
          </button>
          {isExpanded && <button onClick={onToggle} className={`p-2 rounded-lg transition-all duration-200 hover:scale-110 ${theme === 'dark' ? 'hover:bg-white/10 text-white/90' : 'hover:bg-gray-100 text-gray-900'}`}><X size={18} /></button>}
        </div>
      </div>

      {!isExpanded ? (
        <div className="h-full flex flex-col items-center justify-start pt-4 px-1">
          <div className={`w-10 h-10 rounded-md flex items-center justify-center ${theme === 'dark' ? 'bg-black/30' : 'bg-white/60'} cursor-pointer`} onClick={onExpandToggle}>
            <BookOpen size={18} className={theme === 'dark' ? 'text-cyan-400' : 'text-violet-700'} />
          </div>
          <div className={`text-xs mt-3 ${theme === 'dark' ? 'text-white/70' : 'text-gray-700'}`}>Contents</div>
        </div>
      ) : (
        <div className="p-4 overflow-y-auto custom-scrollbar h-[calc(100vh-120px)]">
          <div className="space-y-2">
            {(outline || []).map((item, index) => (
              <div key={item.id || index} onClick={() => onSectionClick(item)} className={`group p-4 rounded-xl cursor-pointer nav-item-hover border ${theme === 'dark' ? 'hover:bg-gradient-to-r hover:from-cyan-500/20 hover:to-indigo-500/20 border-transparent hover:border-cyan-400/30' : 'hover:bg-gradient-to-r hover:from-purple-500/20 hover:to-indigo-500/20 border-transparent hover:border-purple-400/30'}`}>
                <div className="flex items-center gap-3">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm transition-transform duration-300 group-hover:scale-110 ${theme === 'dark' ? 'bg-gradient-to-br from-cyan-400 to-blue-500' : 'bg-gradient-to-br from-purple-500 to-pink-500'}`}>{index + 1}</div>
                  <div className="flex-1">
                    <span className={`font-medium block ${theme === 'dark' ? 'text-white/90' : 'text-gray-900'}`}>{item.title}</span>
                    {(item.page !== undefined) && (<span className={`text-xs mt-1 block ${theme === 'dark' ? 'text-white/80' : 'text-gray-800'}`}>Page {item.page + 1}</span>)}
                  </div>
                </div>
              </div>
            ))}
          </div>
          {(!outline || outline.length === 0) && (
            <div className="text-center py-8">
              <BookOpen size={32} className={`mx-auto mb-4 ${theme === 'dark' ? 'text-white/30' : 'text-gray-400'}`} />
              <p className={`text-sm ${theme === 'dark' ? 'text-white/80' : 'text-gray-700'}`}>No table of contents available.</p>
            </div>
          )}
        </div>
      )}

      {/* left-edge resize handle */}
      {isExpanded && (
        <div
          role="separator"
          onMouseDown={handleResizeMouseDown}
          onTouchStart={(e) => onStartResize(e, 'nav')}
          style={{
            position: 'absolute',
            left: '-8px',
            top: 0,
            bottom: 0,
            width: '16px',
            cursor: 'col-resize',
            zIndex: 60,
          }}
        />
      )}
    </div>
  );
};

const CollectionModal = ({ isOpen, files, collections, onClose, onConfirm, theme }) => {
  const [selectedOption, setSelectedOption] = useState('individual');
  const [newCollectionName, setNewCollectionName] = useState('');
  const [existingCollectionId, setExistingCollectionId] = useState(collections.length > 0 ? collections[0].id : '');

  useEffect(() => {
    if (collections.length > 0 && !existingCollectionId) {
      setExistingCollectionId(collections[0].id);
    }
  }, [collections, existingCollectionId]);

  if (!isOpen) return null;

  const handleConfirm = () => {
    onConfirm({ files, mode: selectedOption, collectionName: newCollectionName, collectionId: existingCollectionId });
    onClose();
    setNewCollectionName('');
    setSelectedOption('individual');
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-md flex items-center justify-center z-50 p-4">
      <div className={`relative rounded-2xl p-8 border ${theme === 'dark' ? 'bg-[#101010] border-white/20' : 'bg-gray-50 border-gray-200'} w-full max-w-lg shadow-2xl`}>
        <button onClick={onClose} className={`absolute top-4 right-4 p-2 rounded-full ${theme === 'dark' ? 'text-gray-400 hover:bg-white/10' : 'text-gray-500 hover:bg-gray-200'} transition`}><X size={20} /></button>
        <h2 className={`text-2xl font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-black'}`}>Organize Your Uploads</h2>
        <p className={`mb-6 ${theme === 'dark' ? 'text-white/70' : 'text-gray-600'}`}>You're uploading {files.length} document(s). Add them to your library.</p>
        <div className="space-y-4">
          <div onClick={() => setSelectedOption('individual')} className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${selectedOption === 'individual' ? 'border-cyan-500 bg-cyan-500/10' : `${theme === 'dark' ? 'border-white/20 hover:border-white/40' : 'border-gray-200 hover:border-gray-400'}`}`}>
            <label className="flex items-center gap-4 cursor-pointer">
              <input type="radio" name="collection-option" checked={selectedOption === 'individual'} onChange={() => setSelectedOption('individual')} className="form-radio text-cyan-500 bg-transparent focus:ring-cyan-500" />
              <span className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-black'}`}>Add as individual documents</span>
            </label>
          </div>
          <div onClick={() => setSelectedOption('new')} className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${selectedOption === 'new' ? 'border-cyan-500 bg-cyan-500/10' : `${theme === 'dark' ? 'border-white/20 hover:border-white/40' : 'border-gray-200 hover:border-gray-400'}`}`}>
            <label className="flex items-center gap-4 cursor-pointer mb-2">
              <input type="radio" name="collection-option" checked={selectedOption === 'new'} onChange={() => setSelectedOption('new')} className="form-radio text-cyan-500 bg-transparent focus:ring-cyan-500" />
              <span className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-black'}`}>Create a new collection</span>
            </label>
            {selectedOption === 'new' && (<input type="text" placeholder="Enter collection name..." value={newCollectionName} onChange={(e) => setNewCollectionName(e.target.value)} className={`mt-2 w-full pl-3 pr-3 py-2 ${theme === 'dark' ? 'bg-black/20 border-white/20 text-white placeholder-gray-400' : 'bg-white border-gray-300 text-black placeholder-gray-500'} border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500/50`} />)}
          </div>
          {collections.length > 0 && (
            <div onClick={() => setSelectedOption('existing')} className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${selectedOption === 'existing' ? 'border-cyan-500 bg-cyan-500/10' : `${theme === 'dark' ? 'border-white/20 hover:border-white/40' : 'border-gray-200 hover:border-gray-400'}`}`}>
              <label className="flex items-center gap-4 cursor-pointer mb-2">
                <input type="radio" name="collection-option" checked={selectedOption === 'existing'} onChange={() => setSelectedOption('existing')} className="form-radio text-cyan-500 bg-transparent focus:ring-cyan-500" />
                <span className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-black'}`}>Add to existing collection</span>
              </label>
              {selectedOption === 'existing' && (<select value={existingCollectionId} onChange={(e) => setExistingCollectionId(e.target.value)} className={`mt-2 w-full pl-3 pr-3 py-2 ${theme === 'dark' ? 'bg-black/20 border-white/20 text-white' : 'bg-white border-gray-300 text-black'} border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500/50`}>{collections.map(col => <option key={col.id} value={col.id}>{col.name}</option>)}</select>)}
            </div>
          )}
        </div>
        <div className="mt-8 flex justify-end gap-3">
          <button onClick={onClose} className={`px-5 py-2.5 rounded-lg font-semibold text-sm transition ${theme === 'dark' ? 'hover:bg-white/10 text-white/80' : 'hover:bg-gray-100 text-gray-700'}`}>Cancel</button>
          <button onClick={handleConfirm} className="px-6 py-2.5 rounded-lg font-semibold text-sm bg-gradient-to-r from-cyan-500 to-blue-500 text-white hover:from-cyan-600 hover:to-blue-600 transition-all shadow-lg hover:shadow-cyan-500/30">Confirm</button>
        </div>
      </div>
    </div>
  );
};

const ConfirmationModal = ({ isOpen, onClose, onConfirm, title, message, theme }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-md flex items-center justify-center z-[60] p-4">
      <div className={`relative rounded-2xl p-8 border ${theme === 'dark' ? 'bg-[#101010] border-white/20' : 'bg-white border-gray-200'} w-full max-w-md shadow-2xl`}>
        <h2 className={`text-xl font-bold mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>{title}</h2>
        <p className={`mb-8 ${theme === 'dark' ? 'text-white/80' : 'text-gray-600'}`}>{message}</p>
        <div className="flex justify-end gap-3">
          <button onClick={onClose} className={`px-5 py-2.5 rounded-lg font-semibold text-sm transition ${theme === 'dark' ? 'bg-white/10 hover:bg-white/20 text-white/90' : 'bg-gray-100 hover:bg-gray-200 text-gray-700'}`}>Cancel</button>
          <button onClick={() => { onConfirm(); onClose(); }} className="px-6 py-2.5 rounded-lg font-semibold text-sm bg-red-600 hover:bg-red-700 text-white transition-all shadow-lg shadow-red-500/20 hover:shadow-red-500/30">Delete</button>
        </div>
      </div>
    </div>
  );
};

const CreateCollectionModal = ({ isOpen, onClose, onConfirm, theme }) => {
  const [name, setName] = useState('');
  if (!isOpen) return null;
  const handleConfirm = () => {
    if (name.trim()) {
      onConfirm(name.trim());
      setName('');
      onClose();
    }
  };
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-md flex items-center justify-center z-[60] p-4">
      <div className={`relative rounded-2xl p-8 border ${theme === 'dark' ? 'bg-[#101010] border-white/20' : 'bg-white border-gray-200'} w-full max-w-md shadow-2xl`}>
        <h2 className={`text-xl font-bold mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>Create New Collection</h2>
        <p className={`mb-6 ${theme === 'dark' ? 'text-white/80' : 'text-gray-600'}`}>Please enter a name for your new collection.</p>
        <input type="text" placeholder="Collection name..." value={name} onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleConfirm()} className={`w-full pl-3 pr-3 py-2 ${theme === 'dark' ? 'bg-black/20 border-white/20 text-white placeholder-gray-400' : 'bg-white border-gray-300 text-black placeholder-gray-500'} border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500/50`} autoFocus />
        <div className="mt-8 flex justify-end gap-3">
          <button onClick={onClose} className={`px-5 py-2.5 rounded-lg font-semibold text-sm transition ${theme === 'dark' ? 'bg-white/10 hover:bg-white/20 text-white/90' : 'bg-gray-100 hover:bg-gray-200 text-gray-700'}`}>Cancel</button>
          <button onClick={handleConfirm} className="px-6 py-2.5 rounded-lg font-semibold text-sm bg-cyan-600 hover:bg-cyan-700 text-white transition-all shadow-lg shadow-cyan-500/20">Create</button>
        </div>
      </div>
    </div>
  );
};


// --- MAIN COMPONENT DEFINITION ---

const DocumentReaderView = () => {
  const [libraryItems, setLibraryItems] = useState([]);
  const [activeDoc, setActiveDoc] = useState(null);
  const [showNavPanel, setShowNavPanel] = useState(false);
  const [navPanelExpanded, setNavPanelExpanded] = useState(true); // <-- CHANGED
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [fullscreenMode, setFullscreenMode] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [collectionModal, setCollectionModal] = useState({ isOpen: false, files: [] });
  const [confirmationModal, setConfirmationModal] = useState({ isOpen: false, title: '', message: '', onConfirm: () => {} });
  const [createCollectionModalOpen, setCreateCollectionModalOpen] = useState(false);
  const { theme } = useTheme();

  const [connections, setConnections] = useState([]);
  const [previousConnections, setPreviousConnections] = useState([]); // keep last non-empty results
  const [isConnecting, setIsConnecting] = useState(false);
  const [showConnectionsPanel, setShowConnectionsPanel] = useState(false);
  const [connectionsPanelExpanded, setConnectionsPanelExpanded] = useState(true);

  // resizable widths (px)
  const [navWidth, setNavWidth] = useState(384); // default w-96
  const [connWidth, setConnWidth] = useState(384);
  const resizingRef = useRef({ dragging: false, panel: null }); // { dragging: bool, panel: 'nav'|'connections' }
  const minWidth = 200;
  const maxWidth = 800;

  // State for the PDF viewer
  const [numPages, setNumPages] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);

  // NEW: AI insights related states (added)
  const [selectedText, setSelectedText] = useState(''); // Track the selected text
  const [lastSelectedText, setLastSelectedText] = useState(''); // Keep last selected text for insights

  // insights state
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [insightsData, setInsightsData] = useState(null);
  const [insightsError, setInsightsError] = useState(null);

  // highlight target for PDFViewer
  const [highlightTarget, setHighlightTarget] = useState(null); // <-- CHANGED: { id, page } passed to PDFViewer

  // NEW: pending open nav state & fallback timer ref (kept for compatibility but not used to auto-open nav anymore)
  const [pendingOpenNavForDocId, setPendingOpenNavForDocId] = useState(null);
  const pendingOpenTimerRef = useRef(null);

  // NEW: reset-on-mount state â€" delay UI/data work until reset completes
  const [resetDone, setResetDone] = useState(false);

  // Helper: update outline for a doc in libraryItems and activeDoc
  const updateDocOutline = useCallback((docId, outline) => {
    setLibraryItems(prev => prev.map(item => {
      if (item.type === 'doc' && item.doc_id === docId) return { ...item, outline };
      if (item.type === 'collection') return { ...item, docs: item.docs.map(d => d.doc_id === docId ? { ...d, outline } : d) };
      return item;
    }));
    setActiveDoc(prev => (prev && prev.doc_id === docId) ? { ...prev, outline } : prev);
  }, []);

  // Reset backend on mount â€" wait then allow UI to continue
  useEffect(() => {
    let mounted = true;
    const resetSession = async () => {
      try {
        await fetch('http://localhost:3000/api/reset', { method: 'POST' });
        console.log('Session reset on page load');
      } catch (err) {
        console.error('Failed to reset session on load:', err);
      } finally {
        if (mounted) setResetDone(true);
      }
    };
    resetSession();
    return () => { mounted = false; };
  }, []);

  // global mouse/touch move handlers for resizing
  useEffect(() => {
    const onMove = (e) => {
      if (!resizingRef.current.dragging) return;
      const panel = resizingRef.current.panel;
      const clientX = (e.touches && e.touches[0]) ? e.touches[0].clientX : e.clientX;
      const winWidth = window.innerWidth;
      if (panel === 'connections') {
        const newConnWidth = Math.max(minWidth, Math.min(maxWidth, winWidth - clientX));
        setConnWidth(newConnWidth);
      } else if (panel === 'nav') {
        // nav is placed to the left of the connections panel (if connections open)
        const rightOffset = showConnectionsPanel && connectionsPanelExpanded ? connWidth : 0;
        const newNavWidth = Math.max(minWidth, Math.min(maxWidth, winWidth - rightOffset - clientX));
        setNavWidth(newNavWidth);
      }
    };

    const onUp = () => {
      if (resizingRef.current.dragging) {
        resizingRef.current.dragging = false;
        resizingRef.current.panel = null;
      }
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('touchmove', onMove, { passive: false });
    window.addEventListener('mouseup', onUp);
    window.addEventListener('touchend', onUp);
    window.addEventListener('touchcancel', onUp);

    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('touchmove', onMove);
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('touchend', onUp);
      window.removeEventListener('touchcancel', onUp);
    };
  }, [connWidth, showConnectionsPanel, connectionsPanelExpanded]);

  const startResize = (e, panel) => {
    e.preventDefault();
    resizingRef.current.dragging = true;
    resizingRef.current.panel = panel;
  };

  // Generate insights helper: only call when we have extracted section text/content
  // Fixed generateInsights function in ConnectionsPanel
const generateInsights = async (connectionsData, selectedText) => {
  setIsGeneratingInsights(true);
  setInsightsError(null);
  
  try {
    console.log('Generating insights with:', {
      selectedText: selectedText?.length,
      connections: connectionsData?.length,
      activeDoc: activeDoc?.fileName
    });

    // Ensure we have valid data
    if (!connectionsData || connectionsData.length === 0) {
      throw new Error('No connections data available');
    }

    if (!selectedText || selectedText.trim().length < 10) {
      throw new Error('Selected text is too short or missing');
    }

    // Format connections properly for the API
    const formattedConnections = connectionsData.map(conn => ({
      text: conn.text || '',
      document_name: conn.document_name || '',
      section_title: conn.section_title || null,
      page_number: conn.page_number || null,
      source: conn.source || null
    }));

    const requestPayload = {
      selected_text: selectedText.trim(),
      connections: formattedConnections,
      source_document: activeDoc?.fileName || activeDoc?.title || 'Current Document',
      context: {
        total_connections: formattedConnections.length,
        documents: [...new Set(formattedConnections.map(c => c.document_name).filter(Boolean))]
      }
    };

    console.log('Sending insights request:', requestPayload);

    const response = await fetch('http://localhost:3000/api/generate-insights', {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(requestPayload),
    });

    console.log('Insights response status:', response.status);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Insights API error:', errorText);
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    // Handle both JSON and text responses
    const contentType = response.headers.get('content-type') || '';
    let result;
    
    if (contentType.includes('application/json')) {
      result = await response.json();
    } else {
      const textResult = await response.text();
      console.log('Got text response:', textResult);
      try {
        result = JSON.parse(textResult);
      } catch (e) {
        // If it's not JSON, treat as raw text
        result = { raw: textResult };
      }
    }

    console.log('Insights result:', result);

    // Set the insights with proper fallback
    if (result && typeof result === 'object') {
      setInsights(result.insights || result);
    } else {
      setInsights({ raw: String(result) });
    }

  } catch (error) {
    console.error('Error generating insights:', error);
    setInsightsError(error.message || 'Failed to generate insights');
    
    // Set a basic fallback insight
    setInsights({
      summary: 'Unable to generate AI insights at this time.',
      patterns: [],
      contradictions: [],
      examples: [],
      additional_insights: [`Error: ${error.message}`]
    });
  } finally {
    setIsGeneratingInsights(false);
  }
};
  const generateInsightsTopLevel = useCallback(async ({ sectionText, docId, activeDoc, setInsightsData, setInsightsLoading, setInsightsError }) => {
    setInsightsError(null);
    setInsightsData(null);
  
    if (!sectionText || !sectionText.trim()) {
      console.warn('generateInsights: no section text provided.');
      return null;
    }
  
    setInsightsLoading(true);
    try {
      const res = await fetch('http://localhost:3000/api/generate-insights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ section_text: sectionText, doc_id: docId || activeDoc?.doc_id })
      });
  
      if (!res.ok) {
        const txt = await res.text().catch(() => '');
        const err = new Error(`Failed to generate insights: ${res.statusText} ${txt}`);
        throw err;
      }
  
      // <-- CHANGED: accept raw text or JSON (try parsing; fallback to raw)
      const rawText = await res.text();
      try {
        const parsed = JSON.parse(rawText);
        setInsightsData(parsed);
        return parsed;
      } catch (e) {
        setInsightsData({ raw: rawText });
        return { raw: rawText };
      }
    } catch (err) {
      console.error('Error generating insights:', err);
      setInsightsError(err.message || String(err));
      throw err;
    } finally {
      setInsightsLoading(false);
    }
  }, []); // eslint-disable-line
  

  const handleTextSelect = useCallback(async (selectedTextContent) => {
    if (!selectedTextContent || selectedTextContent.trim().length < 20) return;

    // Store selected text for insights display, but DO NOT auto-generate insights here.
    // Insights generation is intentionally restricted to explicit section clicks (see handleSectionClick).
    setSelectedText(selectedTextContent);
    setLastSelectedText(selectedTextContent);

    setIsConnecting(true);
    setShowConnectionsPanel(true); // auto-open connections panel when text is selected
    setConnections([]);
    try {
      const response = await fetch('http://localhost:3000/api/find-connections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          selected_text: selectedTextContent,
          current_doc_id: activeDoc?.doc_id,
          source_document_name: activeDoc?.fileName
        }),
      });
      if (!response.ok) throw new Error(`Connection lookup failed: ${response.statusText}`);
      const data = await response.json();
      const found = data.connections || [];
      setConnections(found);
      if (found.length > 0) {
        setPreviousConnections(found); // keep last non-empty results as fallback
      }
    } catch (error) {
      console.error("Error finding connections:", error);
    } finally {
      setIsConnecting(false);
    }
  }, [activeDoc]);

  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({ current: 0, total: 0 });

  const handleUploadAndOrganize = useCallback(async ({ files, mode, collectionName, collectionId }) => {
    setUploadLoading(true);
    setUploadProgress({ current: 0, total: files.length });
    
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    
    try {
      const response = await fetch('http://localhost:3000/api/upload-and-index', { 
        method: 'POST', 
        body: formData 
      });
      
      if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
      
      const newData = await response.json();
      
      const newDocsWithFiles = files.map((file, index) => {
        setUploadProgress(prev => ({ ...prev, current: index + 1 }));
        
        const filePromise = new Promise(resolve => {
          const reader = new FileReader();
          reader.onload = e => resolve(e.target.result);
          reader.readAsArrayBuffer(file);
        });
        
        return { 
          ...newData[index], 
          type: 'doc', 
          file, 
          fileName: file.name, 
          filePromise 
        };
      });
      
      setLibraryItems(prevItems => {
        let updatedItems = [...prevItems];
        
        if (mode === 'individual') {
          updatedItems.push(...newDocsWithFiles);
        } else if (mode === 'new') {
          updatedItems.push({
            id: generateId(),
            type: 'collection',
            name: collectionName,
            docs: newDocsWithFiles
          });
        } else if (mode === 'existing') {
          updatedItems = updatedItems.map(item => 
            item.id === collectionId 
              ? { ...item, docs: [...item.docs, ...newDocsWithFiles] }
              : item
          );
        }
        
        return updatedItems;
      });
      
      if (!activeDoc && newDocsWithFiles.length > 0) {
        setActiveDoc(newDocsWithFiles[0]);
      }
      
    } catch (error) {
      console.error("Error uploading files:", error);
      alert(`Upload failed: ${error.message}`);
    } finally {
      setUploadLoading(false);
      setUploadProgress({ current: 0, total: 0 });
    }
  }, [activeDoc]);
  
  const handleFileChange = (event) => {
    const newFiles = Array.from(event.target.files).filter(f => f.type === 'application/pdf');
    if (newFiles.length === 0) return;
    setCollectionModal({ isOpen: true, files: newFiles });
    event.target.value = null;
  };

  const handleDocumentSelect = useCallback((doc) => {
    setActiveDoc(doc);
    setCurrentPage(1);
    setNumPages(null);
  }, []);
  
  const handleDeleteDocument = useCallback((docIdToDelete, collectionId = null) => {
    setLibraryItems(prevItems => {
      if (collectionId) {
        return prevItems.map(item => 
          item.id === collectionId 
            ? { ...item, docs: item.docs.filter(doc => doc.doc_id !== docIdToDelete) }
            : item
        );
      } else {
        return prevItems.filter(item => item.type === 'collection' || item.doc_id !== docIdToDelete);
      }
    });
    if (activeDoc?.doc_id === docIdToDelete) setActiveDoc(null);
  }, [activeDoc]);

  const handleCreateCollection = useCallback((name) => {
    const newCollection = { id: generateId(), type: 'collection', name, docs: [] };
    setLibraryItems(prev => [...prev, newCollection]);
  }, []);
  
  const handleDeleteCollection = useCallback((collectionId, collectionName) => {
    setConfirmationModal({
      isOpen: true,
      title: `Delete Collection`,
      message: `Are you sure you want to delete "${collectionName}"? This will also remove all documents in the collection.`,
      onConfirm: () => {
        setLibraryItems(prevItems => {
          const collectionToDelete = prevItems.find(item => item.id === collectionId);
          const isDocActiveInCollection = collectionToDelete?.docs.some(doc => doc.doc_id === activeDoc?.doc_id);
          if (isDocActiveInCollection) setActiveDoc(null);
          return prevItems.filter(item => item.id !== collectionId);
        });
      }
    });
  }, [activeDoc]);
  
  const handleMoveDocument = useCallback((docId, sourceCollectionId, targetCollectionId) => {
    setLibraryItems(prevItems => {
      let docToMove = null;
      const itemsWithoutDoc = prevItems.map(item => {
        if (sourceCollectionId && item.id === sourceCollectionId) {
          docToMove = item.docs.find(doc => doc.doc_id === docId);
          return { ...item, docs: item.docs.filter(doc => doc.doc_id !== docId) };
        } else if (!sourceCollectionId && item.doc_id === docId) {
          docToMove = item;
          return null;
        }
        return item;
      }).filter(Boolean);
      
      if (!docToMove) return prevItems;
      
      if (targetCollectionId) {
        return itemsWithoutDoc.map(item => item.id === targetCollectionId ? { ...item, docs: [...item.docs, docToMove] } : item);
      } else {
        return [...itemsWithoutDoc, docToMove];
      }
    });
  }, []);

  const handleSectionClick = useCallback((section) => {
    if (section.page !== undefined) {
      const pageTo = section.page + 1;
      setCurrentPage(pageTo);
      // highlight the section on the PDF (send to PDFViewer)
      const highlight = { id: generateId(), page: pageTo };
      setHighlightTarget(highlight); // <-- CHANGED: set highlight target
      // auto-clear highlight after 5s
      setTimeout(() => {
        setHighlightTarget(null);
      }, 5000);

      // --- NEW: Only generate AI insights when the clicked section contains extracted text/content ---
      // Try to get text from the section item (outline should contain content/text when extracted)
      const sectionText = (section.text || section.content || section.excerpt || section.description || '').toString().trim();

      if (sectionText) {
        // generate insights only for extracted section text (prevents touching the /generate-insights endpoint for arbitrary selection)
        generateInsights({ sectionText, docId: activeDoc?.doc_id })
          .then(data => {
            // optionally you can display or store insightsData (we set state inside generateInsights)
            console.log('Insights generated for section:', data);
          })
          .catch(err => {
            // errors are stored in insightsError; log is fine
            console.warn('Insights generation failed for section click:', err?.message || err);
          });
      } else {
        // if no extracted content is present for the section, do not call the insights endpoint
        console.log('No extracted section text available â€" skipping insights generation.');
      }
    }
    // DO NOT close connections panel when a section is clicked (user requested).
  }, [activeDoc, generateInsights]);

  // Improved, robust connection click handling â€" DO NOT close connections panel anymore
  const handleConnectionClick = useCallback((connection) => {
    const normalize = (s) => {
      if (!s && s !== 0) return '';
      const str = String(s).replace(/^\uFEFF/, '').trim().toLowerCase();
      return str.replace(/\.[^.]+$/, '').trim();
    };

    const allDocs = libraryItems.flatMap(item => item?.type === 'collection' ? (item.docs || []) : [item]).filter(Boolean);
    const connNameNorm = normalize(connection.document_name);

    const docCandidates = (doc) => {
      const candidates = [
        doc.fileName,
        doc.title,
        doc.name,
        doc.file?.name,
        doc.document_name,
        doc.documentTitle,
      ];
      return candidates.map(normalize).filter(Boolean);
    };

    let targetDoc = allDocs.find(doc => docCandidates(doc).some(c => c === connNameNorm));

    if (!targetDoc) {
      targetDoc = allDocs.find(doc => {
        const candidates = docCandidates(doc);
        return candidates.some(c => c.includes(connNameNorm) || connNameNorm.includes(c));
      });
    }

    if (!targetDoc) {
      const connWithPdf = normalize(`${connection.document_name}.pdf`);
      targetDoc = allDocs.find(doc => docCandidates(doc).some(c => c === connWithPdf));
    }

    if (targetDoc) {
      setActiveDoc(targetDoc);
      // navigate to the page (connection.page_number likely 1-based)
      if (typeof connection.page_number === 'number') {
        setCurrentPage(connection.page_number);
        const highlight = { id: generateId(), page: connection.page_number };
        setHighlightTarget(highlight); // <-- CHANGED: highlight on connection click
        setTimeout(() => setHighlightTarget(null), 5000);
      }
      // IMPORTANT: do NOT close the connections panel when clicking a connection
      // setShowConnectionsPanel(false); <-- removed so panel remains open
    } else {
      console.warn("Could not find the document to navigate to:", connection.document_name);
    }
  }, [libraryItems]);

  // Effect: track pendingOpenNavForDocId â€" unchanged
  useEffect(() => {
    if (!pendingOpenNavForDocId) return;
    const checkDoc = () => {
      if (activeDoc?.doc_id === pendingOpenNavForDocId && activeDoc.outline && activeDoc.outline.length > 0) {
        setShowNavPanel(true);
        setPendingOpenNavForDocId(null);
        if (pendingOpenTimerRef.current) { clearTimeout(pendingOpenTimerRef.current); pendingOpenTimerRef.current = null; }
        return;
      }
      const allDocs = libraryItems.flatMap(item => item?.type === 'collection' ? (item.docs || []) : [item]).filter(Boolean);
      const found = allDocs.find(d => d.doc_id === pendingOpenNavForDocId && d.outline && d.outline.length > 0);
      if (found) {
        setActiveDoc(prev => (prev && prev.doc_id === found.doc_id) ? { ...prev, outline: found.outline } : prev);
        setShowNavPanel(true);
        setPendingOpenNavForDocId(null);
        if (pendingOpenTimerRef.current) { clearTimeout(pendingOpenTimerRef.current); pendingOpenTimerRef.current = null; }
      }
    };

    checkDoc();
  }, [pendingOpenNavForDocId, activeDoc, libraryItems]);

  // Clean up fallback timer on unmount
  useEffect(() => {
    return () => {
      if (pendingOpenTimerRef.current) {
        clearTimeout(pendingOpenTimerRef.current);
        pendingOpenTimerRef.current = null;
      }
    };
  }, []);

  // Optional: exposed to PDFViewer â€" when the viewer extracts an outline it can call this.
  const handleOutlineReadyFromViewer = useCallback((outline) => {
    if (!activeDoc) return;
    updateDocOutline(activeDoc.doc_id, outline);

    if (pendingOpenNavForDocId === activeDoc.doc_id) {
      setShowNavPanel(true);
      setPendingOpenNavForDocId(null);
      if (pendingOpenTimerRef.current) { clearTimeout(pendingOpenTimerRef.current); pendingOpenTimerRef.current = null; }
    }
  }, [activeDoc, pendingOpenNavForDocId, updateDocOutline]);

  const goToPrevPage = () => setCurrentPage(prev => Math.max(prev - 1, 1));
  const goToNextPage = () => setCurrentPage(prev => Math.min(prev + 1, numPages || 1));

  const filteredItems = libraryItems.filter(item => {
    if (searchQuery === '') return true;
    const lowerCaseQuery = searchQuery.toLowerCase();
    if (item.type === 'doc') {
      return (item.title?.toLowerCase().includes(lowerCaseQuery) || item.fileName?.toLowerCase().includes(lowerCaseQuery));
    }
    if (item.type === 'collection') {
      if (item.name.toLowerCase().includes(lowerCaseQuery)) return true;
      return item.docs.some(doc => doc.title?.toLowerCase().includes(lowerCaseQuery) || doc.fileName?.toLowerCase().includes(lowerCaseQuery));
    }
    return false;
  });

  // compute how much right offset the nav panel should have (if both open, nav is placed left of connections panel)
  const navOffsetRight = showConnectionsPanel && connectionsPanelExpanded ? connWidth : 0;

  // compute padding-right for main using dynamic widths (so content shifts smoothly)
  const mainStyle = {
    paddingRight: `${(showNavPanel ? (navPanelExpanded ? navWidth : 64) : 0) + (showConnectionsPanel ? (connectionsPanelExpanded ? connWidth : 64) : 0)}px`
  };

  // BLOCK UI until reset finishes
  if (!resetDone) {
    return (
      <div className="flex items-center justify-center h-screen w-screen bg-slate-100 dark:bg-black text-gray-900 dark:text-white">
        <div className="flex flex-col items-center gap-4">
          <div className="w-20 h-20 border-8 border-cyan-200/20 rounded-full relative">
            <div className="w-20 h-20 border-8 border-cyan-500 border-t-transparent rounded-full animate-spin absolute inset-0"></div>
          </div>
          <div className="text-center">
            <h2 className="text-xl font-semibold">Resetting sessionâ€¦</h2>
            <p className="text-sm text-gray-600 dark:text-gray-300 mt-2">Preparing a clean workspace â€" this happens automatically on page load.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-100 dark:bg-black text-gray-900 dark:text-white overflow-hidden">
      <style>{customScrollbarStyles}</style>
      <VantaBackground theme={theme} />
      
      {/* Modals */}
      <CollectionModal isOpen={collectionModal.isOpen} files={collectionModal.files} collections={libraryItems.filter(item => item.type === 'collection')} onClose={() => setCollectionModal({ isOpen: false, files: [] })} onConfirm={handleUploadAndOrganize} theme={theme} />
      <ConfirmationModal isOpen={confirmationModal.isOpen} onClose={() => setConfirmationModal(prev => ({ ...prev, isOpen: false }))} onConfirm={confirmationModal.onConfirm} title={confirmationModal.title} message={confirmationModal.message} theme={theme} />
      <CreateCollectionModal isOpen={createCollectionModalOpen} onClose={() => setCreateCollectionModalOpen(false)} onConfirm={handleCreateCollection} theme={theme} />

      <header className="fixed top-0 left-0 right-0 w-full bg-white/90 dark:bg-black/80 backdrop-blur-xl shadow-sm p-4 flex justify-between items-center z-30 border-b border-white/10">
        <div className="group cursor-pointer flex items-center space-x-3 truncate">
          <img src={logo} alt="weavedocs icon" className="h-8 w-auto" />
          <span className={`font-dancing text-3xl hidden sm:inline font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>weavedocs</span>
        </div>
        <div className="flex items-center gap-4 flex-1 justify-center max-w-md mx-4">
          <div className="relative flex-1">
            <Search size={16} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <input type="text" placeholder="Search library..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} className="w-full pl-10 pr-4 py-2 bg-white/10 dark:bg-black/20 border border-white/20 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500/50 text-gray-900 dark:text-white placeholder-gray-600 dark:placeholder-gray-400" />
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button onClick={() => setFullscreenMode(!fullscreenMode)} className="p-2 rounded-lg hover:bg-white/10 dark:hover:bg-black/20 text-gray-900 dark:text-white/90 hover:text-black dark:hover:text-white transition-all duration-200 hover:scale-110" title={fullscreenMode ? "Exit fullscreen" : "Enter fullscreen"}>
            {fullscreenMode ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
          </button>
          {activeDoc && (
            <>
              <button onClick={() => { setShowNavPanel(true); }} className="flex items-center gap-2 px-3 py-2 bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-medium rounded-lg hover:from-cyan-600 hover:to-blue-600 transition-all duration-200 shadow-lg hover:shadow-xl text-sm hover:scale-105">
                <BookOpen size={16} /><span className="hidden sm:inline">Contents</span>
              </button>

              {/* Connections button next to Contents */}
              <button onClick={() => setShowConnectionsPanel(prev => !prev)} className="flex items-center gap-2 px-3 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-medium rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all duration-200 shadow-lg hover:shadow-xl text-sm hover:scale-105" title="Connections">
                <Search size={16} /><span className="hidden sm:inline">Connections</span>
              </button>
            </>
          )}
          <label className="group relative">
            <div className={`flex items-center px-3 py-2 font-medium rounded-lg transition-all duration-200 cursor-pointer shadow-lg hover:shadow-xl text-sm text-white hover:scale-105 ${theme === 'dark' ? 'bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600' : 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700'}`}>
              <Upload size={16} className="mr-2" /><span className="hidden sm:inline">Upload</span>
            </div>
            <input type="file" className="absolute top-0 left-0 w-full h-full opacity-0 cursor-pointer" accept="application/pdf" multiple onChange={handleFileChange} />
          </label>
          <ThemeToggle />
        </div>
      </header>

      <div className="flex flex-1 pt-16 overflow-hidden">
        {!fullscreenMode && (
          <DocumentList 
            items={filteredItems} 
            activeDoc={activeDoc} 
            onSelect={handleDocumentSelect} 
            onDeleteDoc={handleDeleteDocument} 
            onMoveDocument={handleMoveDocument} 
            onCreateCollection={() => setCreateCollectionModalOpen(true)} 
            onDeleteCollection={handleDeleteCollection} 
            isCollapsed={sidebarCollapsed} 
            onToggle={() => setSidebarCollapsed(!sidebarCollapsed)} 
            theme={theme} 
          />
        )}

        <main className={`flex-1 bg-transparent overflow-hidden relative z-10 transition-all duration-300 ${fullscreenMode ? 'p-2' : 'p-6'}`} style={mainStyle}>
          <div className="w-full h-full rounded-2xl overflow-hidden shadow-2xl border border-white/20 relative z-20">
            {activeDoc ? (
              <div className="w-full h-full relative z-30 bg-white dark:bg-slate-900 rounded-2xl">
                <PDFViewer
                  key={activeDoc.doc_id}
                  filePromise={activeDoc.filePromise}
                  onTextSelect={handleTextSelect}
                  pageNumber={currentPage}
                  onDocumentLoad={setNumPages}
                  // Pass the outline-ready hook â€" PDFViewer may call this when outline is available
                  onOutlineReady={handleOutlineReadyFromViewer}
                  // <-- CHANGED: pass highlightTarget to PDFViewer so it can render highlight on the page
                  highlightTarget={highlightTarget}
                />
                {numPages && (
                  <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/50 backdrop-blur-md text-white rounded-full px-4 py-2 flex items-center gap-4 text-sm">
                    <button onClick={goToPrevPage} disabled={currentPage <= 1} className="disabled:opacity-50"><ArrowLeft size={16} /></button>
                    <span>Page {currentPage} of {numPages}</span>
                    <button onClick={goToNextPage} disabled={currentPage >= numPages} className="disabled:opacity-50"><ArrowRight size={16} /></button>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full bg-black/20 backdrop-blur-sm rounded-2xl relative z-20 border border-white/10">
                <div className="w-32 h-32 bg-gradient-to-br from-emerald-500/20 via-cyan-500/20 to-blue-500/20 rounded-3xl flex items-center justify-center mb-8 border border-emerald-400/30">
                  <FileText size={60} className={theme === 'dark' ? 'text-emerald-400' : 'text-purple-700'} />
                </div>
                <div className="text-center">
                  <h3 className={`text-3xl font-bold mb-4 ${theme === 'dark' ? 'bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-500 bg-clip-text text-transparent' : 'bg-gradient-to-r from-purple-800 via-pink-700 to-indigo-800 bg-clip-text text-transparent'}`}>Ready to Explore</h3>
                  <p className={`max-w-md leading-relaxed text-lg ${theme === 'dark' ? 'text-white/90' : 'text-gray-900'}`}>Select a document from your library to begin reading and analyzing.</p>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>

      {/* Panels */}
      {showNavPanel && (
        <NavigationPanel
          outline={activeDoc?.outline}
          onSectionClick={handleSectionClick}
          isVisible={showNavPanel}
          onToggle={() => setShowNavPanel(false)}
          theme={theme}
          isExpanded={navPanelExpanded}
          onExpandToggle={() => setNavPanelExpanded(prev => !prev)}
          width={navWidth}
          offsetRight={navOffsetRight}
          onStartResize={startResize} // <-- CHANGED: allow resizing
        />
      )}
      
      {showConnectionsPanel && (
         <ConnectionsPanel
           connections={connections}
           previousConnections={previousConnections}
           isLoading={isConnecting}
           theme={theme}
           onClose={() => setShowConnectionsPanel(false)}
           onConnectionClick={handleConnectionClick}
           isShifted={showNavPanel}
           isExpanded={connectionsPanelExpanded}
           onExpandToggle={() => setConnectionsPanelExpanded(prev => !prev)}
           width={connWidth}
           onStartResize={startResize} // <-- CHANGED: allow resizing
           selectedText={lastSelectedText} // <-- ADDED: Pass the selected text for insights
           activeDoc={activeDoc} // <-- ADDED: Pass the active document for context
         />
      )}
    </div>
  );
};


export default DocumentReaderView;