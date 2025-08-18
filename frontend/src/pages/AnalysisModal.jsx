import React, { useState } from 'react';
import { X, Brain, Loader2, FileText, BarChart3 } from 'lucide-react';

const AnalysisModal = ({ isOpen, onClose, document, collection, theme, onAnalyzeDocument, onAnalyzeCollection }) => {
  const [persona, setPersona] = useState('');
  const [query, setQuery] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!persona.trim() || !query.trim()) return;
    if (!document && !collection) return;

    setIsAnalyzing(true);
    setError(null);
    setAnalysisResult(null);

    try {
      let result;
      if (collection) {
        result = await onAnalyzeCollection(collection, persona.trim(), query.trim());
      } else {
        result = await onAnalyzeDocument(document.doc_id, persona.trim(), query.trim());
      }
      setAnalysisResult(result);
    } catch (err) {
      setError(err.message || 'Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleClose = () => {
    setAnalysisResult(null);
    setError(null);
    setPersona('');
    setQuery('');
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-md flex items-center justify-center z-[70] p-4">
      <div className={`relative rounded-2xl border w-full max-w-4xl max-h-[90vh] overflow-hidden shadow-2xl ${
        theme === 'dark' 
          ? 'bg-[#101010] border-white/20' 
          : 'bg-white border-gray-200'
      }`}>
        
        {/* Header */}
        <div className={`p-6 border-b ${theme === 'dark' ? 'border-white/10' : 'border-gray-200'}`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                theme === 'dark' 
                  ? 'bg-gradient-to-br from-purple-500 to-pink-500' 
                  : 'bg-gradient-to-br from-purple-600 to-pink-600'
              }`}>
                {collection ? <BarChart3 size={20} className="text-white" /> : <Brain size={20} className="text-white" />}
              </div>
              <div>
                <h2 className={`text-xl font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                  {collection ? 'Collection Analysis' : 'Document Analysis'}
                </h2>
                <p className={`text-sm ${theme === 'dark' ? 'text-white/70' : 'text-gray-600'}`}>
                  {collection 
                    ? `${collection.name} (${collection.docs.length} documents)`
                    : (document?.title || document?.fileName || 'Unknown Document')
                  }
                </p>
              </div>
            </div>
            <button
              onClick={handleClose}
              className={`p-2 rounded-lg transition-colors ${
                theme === 'dark'
                  ? 'hover:bg-white/10 text-white/80'
                  : 'hover:bg-gray-100 text-gray-700'
              }`}
            >
              <X size={20} />
            </button>
          </div>
        </div>

        <div className="flex flex-col lg:flex-row h-[calc(90vh-120px)]">
          {/* Analysis Configuration */}
          <div className={`lg:w-1/2 p-6 border-r ${theme === 'dark' ? 'border-white/10' : 'border-gray-200'} overflow-y-auto`}>
            <div className="space-y-6">
              
              {/* Persona Input */}
              <div>
                <h3 className={`text-lg font-semibold mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                  Analysis Persona
                </h3>
                <input
                  type="text"
                  placeholder="Enter the persona for analysis (e.g., 'lawyer', 'researcher', 'business analyst')..."
                  value={persona}
                  onChange={(e) => setPersona(e.target.value)}
                  className={`w-full px-4 py-3 rounded-lg border focus:outline-none focus:ring-2 focus:ring-purple-500/50 ${
                    theme === 'dark'
                      ? 'bg-black/20 border-white/20 text-white placeholder-gray-400'
                      : 'bg-white border-gray-300 text-black placeholder-gray-500'
                  }`}
                />
                <p className={`text-xs mt-2 ${theme === 'dark' ? 'text-white/60' : 'text-gray-500'}`}>
                  Examples: "academic researcher", "financial advisor", "legal expert", "marketing professional"
                </p>
              </div>

              {/* Query Input */}
              <div>
                <h3 className={`text-lg font-semibold mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                  Analysis Query
                </h3>
                <textarea
                  placeholder={collection 
                    ? "Enter your analysis request for the entire collection..."
                    : "Enter your specific question or analysis request..."
                  }
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  rows={4}
                  className={`w-full px-4 py-3 rounded-lg border resize-none focus:outline-none focus:ring-2 focus:ring-purple-500/50 ${
                    theme === 'dark'
                      ? 'bg-black/20 border-white/20 text-white placeholder-gray-400'
                      : 'bg-white border-gray-300 text-black placeholder-gray-500'
                  }`}
                />
                <p className={`text-xs mt-2 ${theme === 'dark' ? 'text-white/60' : 'text-gray-500'}`}>
                  {collection 
                    ? "Ask questions that span across all documents in this collection, look for patterns, or request comparative analysis."
                    : "Be specific about what you want to analyze or learn from this document."
                  }
                </p>
              </div>

              {/* Collection Info */}
              {collection && (
                <div className={`p-4 rounded-lg border ${
                  theme === 'dark' 
                    ? 'bg-cyan-500/10 border-cyan-500/30' 
                    : 'bg-blue-50 border-blue-200'
                }`}>
                  <div className="flex items-center gap-2 mb-2">
                    <BarChart3 size={16} className={theme === 'dark' ? 'text-cyan-400' : 'text-blue-600'} />
                    <h4 className={`font-semibold text-sm ${theme === 'dark' ? 'text-cyan-300' : 'text-blue-700'}`}>
                      Collection Analysis
                    </h4>
                  </div>
                  <p className={`text-xs ${theme === 'dark' ? 'text-cyan-200/80' : 'text-blue-600'}`}>
                    This will analyze all {collection.docs.length} documents in the "{collection.name}" collection together, 
                    looking for patterns, themes, and relationships across the entire set.
                  </p>
                </div>
              )}

              {/* Analyze Button */}
              <button
                onClick={handleAnalyze}
                disabled={isAnalyzing || (!document && !collection) || !persona.trim() || !query.trim()}
                className={`w-full flex items-center justify-center gap-2 py-3 px-4 rounded-lg font-semibold transition-all ${
                  isAnalyzing || (!document && !collection) || !persona.trim() || !query.trim()
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 hover:scale-105'
                } text-white shadow-lg`}
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 size={18} className="animate-spin" />
                    {collection ? 'Analyzing Collection...' : 'Analyzing...'}
                  </>
                ) : (
                  <>
                    {collection ? <BarChart3 size={18} /> : <Brain size={18} />}
                    {collection ? 'Analyze Collection' : 'Start Analysis'}
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Results Panel */}
          <div className="lg:w-1/2 p-6 overflow-y-auto">
            <h3 className={`text-lg font-semibold mb-4 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
              Analysis Results
            </h3>
            
            {error && (
              <div className={`p-4 rounded-lg border ${
                theme === 'dark'
                  ? 'bg-red-500/10 border-red-500/20 text-red-400'
                  : 'bg-red-50 border-red-200 text-red-700'
              }`}>
                <p className="font-medium">Analysis Error</p>
                <p className="text-sm mt-1">{error}</p>
              </div>
            )}

            {analysisResult && (
              <div className={`p-4 rounded-lg border space-y-4 ${
                theme === 'dark'
                  ? 'bg-white/5 border-white/10'
                  : 'bg-gray-50 border-gray-200'
              }`}>
                {collection && analysisResult.collection_summary && (
                  <div>
                    <h4 className={`font-semibold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                      Collection Overview
                    </h4>
                    <p className={`text-sm leading-relaxed ${theme === 'dark' ? 'text-white/90' : 'text-gray-700'}`}>
                      {analysisResult.collection_summary}
                    </p>
                  </div>
                )}

                {analysisResult.summary && (
                  <div>
                    <h4 className={`font-semibold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                      {collection ? 'Analysis Summary' : 'Summary'}
                    </h4>
                    <p className={`text-sm leading-relaxed ${theme === 'dark' ? 'text-white/90' : 'text-gray-700'}`}>
                      {analysisResult.summary}
                    </p>
                  </div>
                )}
                
                {analysisResult.keyPoints && (
                  <div>
                    <h4 className={`font-semibold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                      Key Findings
                    </h4>
                    <ul className={`text-sm space-y-1 ${theme === 'dark' ? 'text-white/90' : 'text-gray-700'}`}>
                      {analysisResult.keyPoints.map((point, index) => (
                        <li key={index} className="flex items-start gap-2">
                          <span className="text-purple-500 mt-1">•</span>
                          <span>{point}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {collection && analysisResult.document_insights && (
                  <div>
                    <h4 className={`font-semibold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                      Individual Document Insights
                    </h4>
                    <div className="space-y-2">
                      {analysisResult.document_insights.map((insight, index) => (
                        <div key={index} className={`p-3 rounded-lg ${theme === 'dark' ? 'bg-black/20' : 'bg-white'}`}>
                          <h5 className={`font-medium text-sm mb-1 ${theme === 'dark' ? 'text-cyan-300' : 'text-purple-700'}`}>
                            {insight.document_title}
                          </h5>
                          <p className={`text-xs ${theme === 'dark' ? 'text-white/80' : 'text-gray-600'}`}>
                            {insight.insight}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {collection && analysisResult.cross_document_patterns && (
                  <div>
                    <h4 className={`font-semibold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                      Cross-Document Patterns
                    </h4>
                    <ul className={`text-sm space-y-1 ${theme === 'dark' ? 'text-white/90' : 'text-gray-700'}`}>
                      {analysisResult.cross_document_patterns.map((pattern, index) => (
                        <li key={index} className="flex items-start gap-2">
                          <span className="text-cyan-500 mt-1">•</span>
                          <span>{pattern}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {analysisResult.recommendations && (
                  <div>
                    <h4 className={`font-semibold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                      Recommendations
                    </h4>
                    <p className={`text-sm leading-relaxed ${theme === 'dark' ? 'text-white/90' : 'text-gray-700'}`}>
                      {analysisResult.recommendations}
                    </p>
                  </div>
                )}

                {analysisResult.confidence && (
                  <div className="pt-2 border-t border-white/10">
                    <div className="flex items-center justify-between text-xs">
                      <span className={theme === 'dark' ? 'text-white/70' : 'text-gray-600'}>
                        Confidence Level
                      </span>
                      <span className={`font-medium ${
                        analysisResult.confidence >= 80 ? 'text-green-500' :
                        analysisResult.confidence >= 60 ? 'text-yellow-500' : 'text-red-500'
                      }`}>
                        {analysisResult.confidence}%
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}

            {!analysisResult && !error && !isAnalyzing && (
              <div className="flex flex-col items-center justify-center h-full text-center py-12">
                <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-4 ${
                  theme === 'dark'
                    ? 'bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-400/30'
                    : 'bg-gradient-to-br from-purple-100 to-pink-100 border border-purple-200'
                }`}>
                  {collection ? <BarChart3 size={28} className={theme === 'dark' ? 'text-purple-400' : 'text-purple-600'} /> : <FileText size={28} className={theme === 'dark' ? 'text-purple-400' : 'text-purple-600'} />}
                </div>
                <h4 className={`font-semibold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
                  Ready to Analyze
                </h4>
                <p className={`text-sm max-w-xs ${theme === 'dark' ? 'text-white/70' : 'text-gray-600'}`}>
                  {collection 
                    ? "Enter a persona and query to analyze this entire collection of documents together."
                    : "Select an analysis perspective and click \"Start Analysis\" to get insights about your document."
                  }
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisModal;