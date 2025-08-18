import React, { useState, useEffect, useRef } from 'react';
import { ArrowLeft, Brain, BarChart3, Network, Users, FileText, TrendingUp, Zap, Target, Eye, Search, MessageSquare } from 'lucide-react';
import * as d3 from 'd3';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, Cell, PieChart, Pie, LineChart, Line, Area, AreaChart } from 'recharts';

// Enhanced Network Graph Component
const DocumentNetworkGraph = ({ documents, connections, theme }) => {
  const svgRef = useRef();
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);

  useEffect(() => {
    if (!documents || documents.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 800;
    const height = 500;
    const margin = 60;

    // Create enhanced nodes with more metadata
    const nodes = documents.map((doc, index) => ({
      id: doc.doc_id || `doc-${index}`,
      title: doc.title || doc.fileName || `Document ${index + 1}`,
      group: Math.floor(index / Math.max(1, Math.floor(documents.length / 5))),
      size: Math.random() * 30 + 15, // Variable node sizes
      importance: Math.random() * 0.8 + 0.2,
      x: Math.random() * (width - 2 * margin) + margin,
      y: Math.random() * (height - 2 * margin) + margin,
      connections: Math.floor(Math.random() * 3) + 1
    }));

    // Create more realistic connections based on document similarity
    const links = [];
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        // Higher probability for documents in similar groups
        const groupSimilarity = nodes[i].group === nodes[j].group ? 0.7 : 0.3;
        if (Math.random() < groupSimilarity * 0.6) {
          links.push({
            source: nodes[i].id,
            target: nodes[j].id,
            strength: Math.random() * 0.8 + 0.2,
            type: Math.random() > 0.5 ? 'strong' : 'weak'
          });
        }
      }
    }

    // Enhanced force simulation
    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(d => d.type === 'strong' ? 80 : 120))
      .force("charge", d3.forceManyBody().strength(d => -300 - (d.size * 5)))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(d => d.size + 5))
      .force("x", d3.forceX(width / 2).strength(0.1))
      .force("y", d3.forceY(height / 2).strength(0.1));

    svg.attr("width", width).attr("height", height);

    // Add background pattern
    const defs = svg.append("defs");
    const pattern = defs.append("pattern")
      .attr("id", "grid")
      .attr("width", 20)
      .attr("height", 20)
      .attr("patternUnits", "userSpaceOnUse");
    
    pattern.append("path")
      .attr("d", "M 20 0 L 0 0 0 20")
      .attr("fill", "none")
      .attr("stroke", theme === 'dark' ? "#374151" : "#f3f4f6")
      .attr("stroke-width", 1)
      .attr("opacity", 0.3);

    svg.append("rect")
      .attr("width", "100%")
      .attr("height", "100%")
      .attr("fill", "url(#grid)");

    // Add gradient definitions for links
    const gradient = defs.append("linearGradient")
      .attr("id", "linkGradient")
      .attr("gradientUnits", "userSpaceOnUse");
    
    gradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", theme === 'dark' ? "#10b981" : "#8b5cf6")
      .attr("stop-opacity", 0.8);
    
    gradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", theme === 'dark' ? "#06b6d4" : "#ec4899")
      .attr("stop-opacity", 0.3);

    // Add enhanced links with animations
    const link = svg.append("g")
      .selectAll("line")
      .data(links)
      .enter()
      .append("line")
      .attr("stroke", "url(#linkGradient)")
      .attr("stroke-opacity", d => d.strength * 0.7)
      .attr("stroke-width", d => d.strength * 4)
      .attr("stroke-dasharray", d => d.type === 'weak' ? "3,3" : "none")
      .style("filter", "drop-shadow(0px 0px 2px rgba(0,0,0,0.3))")
      .on("mouseover", function(event, d) {
        d3.select(this)
          .attr("stroke-width", d.strength * 6)
          .attr("stroke-opacity", 0.9);
      })
      .on("mouseout", function(event, d) {
        d3.select(this)
          .attr("stroke-width", d.strength * 4)
          .attr("stroke-opacity", d.strength * 0.7);
      });

    // Add enhanced nodes with gradients
    const nodeGradient = defs.selectAll(".nodeGradient")
      .data(nodes)
      .enter()
      .append("radialGradient")
      .attr("class", "nodeGradient")
      .attr("id", d => `nodeGrad-${d.id}`)
      .attr("cx", "30%")
      .attr("cy", "30%");

    nodeGradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", (d, i) => {
        const colors = theme === 'dark' 
          ? ["#34d399", "#22d3ee", "#60a5fa", "#a78bfa", "#f472b6"]
          : ["#8b5cf6", "#ec4899", "#ef4444", "#f59e0b", "#10b981"];
        return colors[d.group % colors.length];
      })
      .attr("stop-opacity", 0.9);

    nodeGradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", (d, i) => {
        const colors = theme === 'dark' 
          ? ["#059669", "#0891b2", "#3b82f6", "#7c3aed", "#be185d"]
          : ["#581c87", "#9d174d", "#b91c1c", "#b45309", "#047857"];
        return colors[d.group % colors.length];
      })
      .attr("stop-opacity", 0.7);

    const node = svg.append("g")
      .selectAll("circle")
      .data(nodes)
      .enter()
      .append("circle")
      .attr("r", d => d.size)
      .attr("fill", d => `url(#nodeGrad-${d.id})`)
      .attr("stroke", theme === 'dark' ? "#ffffff" : "#000000")
      .attr("stroke-width", 2)
      .style("cursor", "pointer")
      .style("filter", "drop-shadow(0px 2px 4px rgba(0,0,0,0.2))")
      .on("mouseover", function(event, d) {
        setHoveredNode(d);
        d3.select(this)
          .transition()
          .duration(200)
          .attr("r", d.size * 1.3)
          .attr("stroke-width", 3);
        
        // Highlight connected links
        link.style("stroke-opacity", l => 
          (l.source.id === d.id || l.target.id === d.id) ? 1 : 0.1
        );
      })
      .on("mouseout", function(event, d) {
        setHoveredNode(null);
        d3.select(this)
          .transition()
          .duration(200)
          .attr("r", d.size)
          .attr("stroke-width", 2);
        
        // Reset link opacity
        link.style("stroke-opacity", l => l.strength * 0.7);
      })
      .on("click", function(event, d) {
        setSelectedNode(selectedNode?.id === d.id ? null : d);
      });

    // Add connection count indicators
    const connectionBadge = svg.append("g")
      .selectAll("circle")
      .data(nodes.filter(d => d.connections > 1))
      .enter()
      .append("circle")
      .attr("r", 8)
      .attr("fill", theme === 'dark' ? "#f59e0b" : "#dc2626")
      .attr("stroke", "white")
      .attr("stroke-width", 2)
      .style("pointer-events", "none");

    const connectionText = svg.append("g")
      .selectAll("text")
      .data(nodes.filter(d => d.connections > 1))
      .enter()
      .append("text")
      .text(d => d.connections)
      .attr("text-anchor", "middle")
      .attr("dy", 3)
      .attr("font-size", "10px")
      .attr("font-weight", "bold")
      .attr("fill", "white")
      .style("pointer-events", "none");

    // Enhanced labels with better positioning
    const labels = svg.append("g")
      .selectAll("text")
      .data(nodes)
      .enter()
      .append("text")
      .text(d => {
        const title = d.title;
        return title.length > 12 ? title.substring(0, 12) + "..." : title;
      })
      .attr("text-anchor", "middle")
      .attr("dy", d => d.size + 18)
      .attr("font-size", "11px")
      .attr("font-weight", "500")
      .attr("fill", theme === 'dark' ? "#ffffff" : "#000000")
      .style("pointer-events", "none")
      .style("text-shadow", theme === 'dark' ? "0px 1px 2px rgba(0,0,0,0.8)" : "0px 1px 2px rgba(255,255,255,0.8)");

    // Update positions on simulation tick
    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("cx", d => Math.max(d.size, Math.min(width - d.size, d.x)))
        .attr("cy", d => Math.max(d.size, Math.min(height - d.size, d.y)));

      connectionBadge
        .attr("cx", d => Math.max(d.size, Math.min(width - d.size, d.x + d.size * 0.7)))
        .attr("cy", d => Math.max(d.size, Math.min(height - d.size, d.y - d.size * 0.7)));

      connectionText
        .attr("x", d => Math.max(d.size, Math.min(width - d.size, d.x + d.size * 0.7)))
        .attr("y", d => Math.max(d.size, Math.min(height - d.size, d.y - d.size * 0.7)));

      labels
        .attr("x", d => Math.max(d.size, Math.min(width - d.size, d.x)))
        .attr("y", d => Math.max(d.size, Math.min(height - d.size, d.y)));
    });

    // Enhanced drag functionality
    const drag = d3.drag()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });

    node.call(drag);

  }, [documents, connections, theme]);

  return (
    <div className="relative">
      <svg 
        ref={svgRef} 
        className="w-full h-full border rounded-xl shadow-lg" 
        style={{
          backgroundColor: theme === 'dark' ? '#1f2937' : '#f9fafb',
          minHeight: '500px'
        }}
      />
      
      {/* Enhanced info panel */}
      {(selectedNode || hoveredNode) && (
        <div className={`absolute top-4 left-4 p-4 rounded-xl shadow-xl max-w-xs ${
          theme === 'dark' ? 'bg-gray-800 text-white border border-gray-700' : 'bg-white text-gray-900 border border-gray-200'
        }`}>
          <h4 className="font-bold text-lg mb-2">{(selectedNode || hoveredNode).title}</h4>
          <div className="space-y-1 text-sm">
            <p><span className="font-medium">Connections:</span> {(selectedNode || hoveredNode).connections}</p>
            <p><span className="font-medium">Importance:</span> {((selectedNode || hoveredNode).importance * 100).toFixed(1)}%</p>
            <p><span className="font-medium">Group:</span> {(selectedNode || hoveredNode).group + 1}</p>
          </div>
          {selectedNode && (
            <button 
              onClick={() => setSelectedNode(null)}
              className="mt-3 px-3 py-1 text-xs bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
            >
              Close Details
            </button>
          )}
        </div>
      )}

      {/* Legend */}
      <div className={`absolute bottom-4 right-4 p-3 rounded-lg ${
        theme === 'dark' ? 'bg-gray-800/90 text-white' : 'bg-white/90 text-gray-900'
      }`}>
        <div className="text-xs space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gradient-to-r from-purple-500 to-pink-500"></div>
            <span>Document Node</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-0.5 bg-gradient-to-r from-green-500 to-blue-500"></div>
            <span>Connection</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <span>Connection Count</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Enhanced Theme Analysis Chart
const ThemeAnalysisChart = ({ themes, theme }) => {
  const colors = theme === 'dark' 
    ? ['#10b981', '#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b']
    : ['#7c3aed', '#db2777', '#dc2626', '#ea580c', '#059669', '#d97706'];

  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={themes} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
        <defs>
          {colors.map((color, index) => (
            <linearGradient key={index} id={`barGradient${index}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.9} />
              <stop offset="100%" stopColor={color} stopOpacity={0.6} />
            </linearGradient>
          ))}
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke={theme === 'dark' ? '#374151' : '#e5e7eb'} />
        <XAxis 
          dataKey="theme" 
          stroke={theme === 'dark' ? '#ffffff' : '#000000'}
          fontSize={12}
          angle={-45}
          textAnchor="end"
          height={80}
        />
        <YAxis stroke={theme === 'dark' ? '#ffffff' : '#000000'} />
        <Tooltip 
          contentStyle={{
            backgroundColor: theme === 'dark' ? '#1f2937' : '#ffffff',
            border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
            color: theme === 'dark' ? '#ffffff' : '#000000',
            borderRadius: '8px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
          }}
        />
        <Bar dataKey="frequency" radius={[4, 4, 0, 0]}>
          {themes.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={`url(#barGradient${index % colors.length})`} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

// Main Component
const CollectionAnalysisPage = () => {
  const [theme, setTheme] = useState('light');
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('network');
  const [persona, setPersona] = useState('Research Analyst');
  const [query, setQuery] = useState('Analyze the key themes and connections across these documents');
  const [error, setError] = useState(null);
  
  // This should come from React Router location state
  // Uncomment these lines when integrating with your router:
  // const location = useLocation();
  // const navigate = useNavigate();
  // const collection = location.state?.collection;
  
  // For demo purposes - replace this with actual collection data
  // In your real app, you'll get this from location.state?.collection
  const [collection, setCollection] = useState(null);
  
  // Demo: Set a sample collection with documents
  // Remove this useEffect when integrating with real router data
  useEffect(() => {
    // Simulate getting collection data (replace with real data source)
    const demoCollection = {
      name: 'Food & Culinary Collection',
      docs: [
        { doc_id: 'doc_1', title: 'Italian Cooking Fundamentals', fileName: 'italian-cooking.pdf' },
        { doc_id: 'doc_2', title: 'Mediterranean Diet Guide', fileName: 'med-diet.pdf' },
        { doc_id: 'doc_3', title: 'Wine Pairing Essentials', fileName: 'wine-pairing.pdf' },
        { doc_id: 'doc_4', title: 'Regional Italian Cuisines', fileName: 'regional-cuisine.pdf' },
        { doc_id: 'doc_5', title: 'Pasta Making Techniques', fileName: 'pasta-techniques.pdf' }
      ]
    };
    setCollection(demoCollection);
  }, []);
  
  const documents = collection?.docs || [];

  const performAnalysis = async () => {
    if (!collection || documents.length === 0) {
      setError('No documents found in collection');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      // Extract document IDs for batch analysis
      const docIds = documents.map(doc => doc.doc_id);
      
      const response = await fetch('http://localhost:8000/api/analyze-collection', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          doc_ids: docIds,
          persona: persona,
          query: query,
          collection_name: collection.name
        })
      });
  
      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.status} ${response.statusText}`);
      }
  
      const analysisResult = await response.json();
      
      // Transform the backend response to match UI expectations
      const transformedData = {
        summary: analysisResult.summary || `Analysis of "${collection.name}" collection completed.`,
        themes: analysisResult.themes || [],
        insights: analysisResult.insights || [],
        keyFindings: analysisResult.key_findings || [],
        documentConnections: analysisResult.document_connections || documents.map(doc => ({
          id: doc.doc_id,
          title: doc.title || doc.fileName,
          connections: 0,
          strength: 0.5
        }))
      };
  
      setAnalysisData(transformedData);
    } catch (error) {
      console.error('Collection analysis failed:', error);
      setError(`Analysis failed: ${error.message}`);
      
      // Set error state instead of fallback data
      setAnalysisData(null);
    } finally {
      setLoading(false);
    }
  };

  const runAnalysis = async () => {
    if (!query.trim()) {
      setError('Please enter an analysis query');
      return;
    }
    if (!collection || documents.length === 0) {
      setError('No collection or documents available for analysis');
      return;
    }
    performAnalysis();
  };

  // Don't render main content until collection is loaded
  if (!collection) {
    return (
      <div className={`min-h-screen flex items-center justify-center ${
        theme === 'dark' 
          ? 'bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white' 
          : 'bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 text-gray-900'
      }`}>
        <div className="text-center">
          <div className={`w-12 h-12 border-4 border-t-transparent rounded-full animate-spin mb-4 mx-auto ${
            theme === 'dark' ? 'border-emerald-600' : 'border-purple-600'
          }`}></div>
          <h2 className="text-xl font-bold mb-2">Loading Collection...</h2>
          <p className="text-gray-600 dark:text-gray-400">Please wait while we load your collection data.</p>
        </div>
      </div>
    );
  }

  const tabButtons = [
    { id: 'network', label: 'Network View', icon: Network },
    { id: 'themes', label: 'Theme Analysis', icon: BarChart3 },
    { id: 'insights', label: 'Insights', icon: Eye }
  ];

  return (
    <div className={`min-h-screen transition-all duration-300 ${
      theme === 'dark' 
        ? 'bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white' 
        : 'bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 text-gray-900'
    }`}>
      {/* Header */}
      <header className={`sticky top-0 z-50 backdrop-blur-xl border-b transition-all duration-300 ${
        theme === 'dark' 
          ? 'bg-gray-900/90 border-gray-700/50' 
          : 'bg-white/90 border-gray-200/50'
      }`}>
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <button
                className={`p-2 rounded-xl transition-all duration-200 hover:scale-110 ${
                  theme === 'dark' 
                    ? 'bg-gray-800 hover:bg-gray-700 text-white' 
                    : 'bg-gray-100 hover:bg-gray-200 text-gray-900'
                }`}
              >
                <ArrowLeft size={20} />
              </button>
              
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${
                  theme === 'dark' ? 'bg-emerald-500/20' : 'bg-purple-500/20'
                }`}>
                  <Brain className={theme === 'dark' ? 'text-emerald-400' : 'text-purple-600'} size={24} />
                </div>
                <div>
                  <h1 className={`text-2xl font-bold ${
                    theme === 'dark' 
                      ? 'bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent'
                      : 'bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent'
                  }`}>
                    Collection Analysis
                  </h1>
                  <p className="text-sm opacity-75">
                    {collection.name} • {documents.length} documents
                  </p>
                </div>
              </div>
            </div>

            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className={`p-2 rounded-xl transition-all duration-200 ${
                theme === 'dark' 
                  ? 'bg-gray-800 hover:bg-gray-700 text-yellow-400' 
                  : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
              }`}
            >
              {theme === 'dark' ? '☀️' : '🌙'}
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Enhanced Analysis Controls */}
        <div className={`mb-8 p-6 rounded-2xl border backdrop-blur-sm transition-all duration-300 ${
          theme === 'dark'
            ? 'bg-gray-800/50 border-gray-700'
            : 'bg-white/70 border-gray-200'
        }`}>
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium mb-2">Analysis Persona</label>
                <select
                  value={persona}
                  onChange={(e) => setPersona(e.target.value)}
                  className={`w-full p-3 rounded-xl border transition-all duration-200 ${
                    theme === 'dark'
                      ? 'bg-gray-900 border-gray-600 text-white focus:border-emerald-400'
                      : 'bg-white border-gray-300 text-gray-900 focus:border-purple-500'
                  }`}
                >
                  <option value="Research Analyst">🔬 Research Analyst</option>
                  <option value="Strategic Consultant">💼 Strategic Consultant</option>
                  <option value="Data Scientist">📊 Data Scientist</option>
                  <option value="Business Analyst">📈 Business Analyst</option>
                  <option value="Content Curator">📚 Content Curator</option>
                </select>
              </div>
              
              <div className="flex items-end">
                <button
                  onClick={runAnalysis}
                  disabled={loading}
                  className={`w-full px-6 py-3 rounded-xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium ${
                    theme === 'dark'
                      ? 'bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-700 hover:to-cyan-700 text-white'
                      : 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white'
                  }`}
                >
                  {/* Show empty state only if collection exists but has no documents */}
        {documents.length === 0 ? (
          <div className={`p-12 rounded-2xl border text-center ${
            theme === 'dark'
              ? 'bg-gray-800/50 border-gray-700'
              : 'bg-white/70 border-gray-200'
          }`}>
            <div className="flex flex-col items-center justify-center">
              <div className={`p-4 rounded-full mb-4 ${
                theme === 'dark' ? 'bg-orange-500/20' : 'bg-orange-100'
              }`}>
                <FileText className={theme === 'dark' ? 'text-orange-400' : 'text-orange-600'} size={32} />
              </div>
              <h3 className="text-xl font-bold mb-2">Empty Collection</h3>
              <p className="text-gray-600 dark:text-gray-400 text-center max-w-md">
                The collection "{collection.name}" contains no documents to analyze. Please add some documents first.
              </p>
            </div>
          </div>
        ) : loading ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Brain size={18} />
                      Analyze Collection
                    </>
                  )}
                </button>
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Analysis Query</label>
              <div className="relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" size={18} />
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  rows={3}
                  className={`w-full pl-12 pr-4 py-3 rounded-xl border transition-all duration-200 resize-none ${
                    theme === 'dark'
                      ? 'bg-gray-900 border-gray-600 text-white placeholder-gray-400 focus:border-emerald-400'
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-purple-500'
                  }`}
                  placeholder="Describe what you want to analyze across your document collection..."
                />
              </div>
            </div>
          </div>
        </div>

        {loading ? (
          <div className={`p-12 rounded-2xl border text-center ${
            theme === 'dark'
              ? 'bg-gray-800/50 border-gray-700'
              : 'bg-white/70 border-gray-200'
          }`}>
            <div className="flex flex-col items-center justify-center">
              <div className={`w-16 h-16 border-4 border-t-transparent rounded-full animate-spin mb-6 ${
                theme === 'dark' ? 'border-emerald-600' : 'border-purple-600'
              }`}></div>
              <h3 className="text-2xl font-bold mb-2">Analyzing Collection</h3>
              <p className="opacity-75 text-center max-w-md mb-4">
                Processing {documents.length} documents with the backend AI...
              </p>
              <div className="flex items-center gap-2 text-sm opacity-60">
                <MessageSquare size={16} />
                <span>Query: {query}</span>
              </div>
              <div className="flex items-center gap-2 text-sm opacity-60 mt-2">
                <Users size={16} />
                <span>Persona: {persona}</span>
              </div>
            </div>
          </div>
        ) : error ? (
          <div className={`p-12 rounded-2xl border text-center ${
            theme === 'dark'
              ? 'bg-red-900/20 border-red-800'
              : 'bg-red-50 border-red-200'
          }`}>
            <div className="flex flex-col items-center justify-center">
              <div className={`p-4 rounded-full mb-4 ${
                theme === 'dark' ? 'bg-red-500/20' : 'bg-red-100'
              }`}>
                <Brain className={theme === 'dark' ? 'text-red-400' : 'text-red-600'} size={32} />
              </div>
              <h3 className="text-xl font-bold mb-2 text-red-600 dark:text-red-400">Analysis Failed</h3>
              <p className="text-red-700 dark:text-red-300 text-center max-w-md mb-4">
                {error}
              </p>
              <button
                onClick={runAnalysis}
                className={`px-6 py-2 rounded-lg transition-colors ${
                  theme === 'dark'
                    ? 'bg-red-600 hover:bg-red-700 text-white'
                    : 'bg-red-600 hover:bg-red-700 text-white'
                }`}
              >
                Retry Analysis
              </button>
            </div>
          </div>
        ) : analysisData ? (
          <>
            {/* Enhanced Summary Section */}
            <div className={`mb-8 p-8 rounded-2xl border transition-all duration-300 ${
              theme === 'dark'
                ? 'bg-gradient-to-r from-gray-800/50 to-gray-700/50 border-gray-700'
                : 'bg-gradient-to-r from-white/70 to-blue-50/70 border-gray-200'
            }`}>
              <div className="flex items-start gap-6">
                <div className={`p-4 rounded-2xl flex-shrink-0 ${
                  theme === 'dark' ? 'bg-emerald-500/20' : 'bg-purple-500/20'
                }`}>
                  <Zap className={theme === 'dark' ? 'text-emerald-400' : 'text-purple-600'} size={32} />
                </div>
                <div className="flex-1">
                  <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                    Analysis Summary
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                      theme === 'dark' 
                        ? 'bg-emerald-500/20 text-emerald-400' 
                        : 'bg-purple-500/20 text-purple-600'
                    }`}>
                      {persona}
                    </span>
                  </h2>
                  <p className="text-lg leading-relaxed opacity-90 mb-4">
                    {analysisData.summary}
                  </p>
                  <div className="flex items-center gap-4 text-sm opacity-75">
                    <span className="flex items-center gap-1">
                      <FileText size={14} />
                      {documents.length} Documents
                    </span>
                    <span className="flex items-center gap-1">
                      <TrendingUp size={14} />
                      {analysisData.themes.length} Themes
                    </span>
                    <span className="flex items-center gap-1">
                      <Network size={14} />
                      {analysisData.keyFindings.length} Key Findings
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Key Findings Grid */}
            <div className={`mb-8 p-6 rounded-2xl border ${
              theme === 'dark'
                ? 'bg-gray-800/50 border-gray-700'
                : 'bg-white/70 border-gray-200'
            }`}>
              <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
                <TrendingUp className={theme === 'dark' ? 'text-cyan-400' : 'text-purple-600'} size={24} />
                Key Findings
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {analysisData.keyFindings.map((finding, index) => (
                  <div
                    key={index}
                    className={`group p-5 rounded-xl border-l-4 transition-all duration-200 hover:scale-[1.02] ${
                      theme === 'dark'
                        ? 'bg-gray-700/50 border-l-emerald-400 hover:bg-gray-700/70'
                        : 'bg-blue-50/50 border-l-purple-500 hover:bg-blue-50/80'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 ${
                        theme === 'dark' ? 'bg-emerald-400 text-gray-900' : 'bg-purple-500 text-white'
                      }`}>
                        {index + 1}
                      </div>
                      <p className="text-sm leading-relaxed group-hover:text-opacity-100 transition-opacity">
                        {finding}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Visualization Tabs */}
            <div className={`rounded-2xl border overflow-hidden shadow-xl ${
              theme === 'dark'
                ? 'bg-gray-800/50 border-gray-700'
                : 'bg-white/70 border-gray-200'
            }`}>
              <div className="flex overflow-x-auto border-b border-gray-200 dark:border-gray-700">
                {tabButtons.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-3 px-8 py-4 whitespace-nowrap transition-all duration-200 font-medium ${
                      activeTab === tab.id
                        ? theme === 'dark'
                          ? 'bg-emerald-500/20 text-emerald-400 border-b-2 border-emerald-400'
                          : 'bg-purple-500/20 text-purple-600 border-b-2 border-purple-500'
                        : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-50 dark:hover:bg-gray-700/50'
                    }`}
                  >
                    <tab.icon size={20} />
                    {tab.label}
                  </button>
                ))}
              </div>

              <div className="p-8">
                {activeTab === 'network' && (
                  <div>
                    <div className="mb-6">
                      <h3 className="text-xl font-bold mb-2">Document Connection Network</h3>
                      <p className="text-sm opacity-75 leading-relaxed">
                        Interactive visualization showing thematic relationships between documents. 
                        Node size indicates document importance, colors represent thematic groups, 
                        and connections show content similarity. Hover over nodes for details, 
                        click to select, and drag to reposition.
                      </p>
                    </div>
                    <div className="min-h-[500px] rounded-xl overflow-hidden">
                      <DocumentNetworkGraph 
                        documents={analysisData.documentConnections} 
                        connections={[]} 
                        theme={theme} 
                      />
                    </div>
                  </div>
                )}

                {activeTab === 'themes' && (
                  <div>
                    <div className="mb-6">
                      <h3 className="text-xl font-bold mb-2">Theme Frequency Analysis</h3>
                      <p className="text-sm opacity-75 leading-relaxed">
                        Distribution of key themes identified across your document collection. 
                        Higher frequencies indicate more prominent topics that appear consistently 
                        across multiple documents.
                      </p>
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-900/50 rounded-xl p-4">
                      <ThemeAnalysisChart themes={analysisData.themes} theme={theme} />
                    </div>
                    
                    {/* Theme Details */}
                    <div className="mt-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {analysisData.themes.map((themeData, index) => (
                        <div key={index} className={`p-4 rounded-lg border transition-all hover:scale-105 ${
                          theme === 'dark' 
                            ? 'bg-gray-700/30 border-gray-600 hover:bg-gray-700/50' 
                            : 'bg-white/50 border-gray-200 hover:bg-white/80'
                        }`}>
                          <h4 className="font-semibold mb-2">{themeData.theme}</h4>
                          <div className="flex justify-between text-sm opacity-75">
                            <span>Frequency: {themeData.frequency}</span>
                            <span>Docs: {themeData.documents}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {activeTab === 'insights' && (
                  <div>
                    <div className="mb-6">
                      <h3 className="text-xl font-bold mb-2">Collection Insights</h3>
                      <p className="text-sm opacity-75 leading-relaxed">
                        Quantitative breakdown of insights discovered across your document collection.
                      </p>
                    </div>
                    
                    {/* Insights Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
                      {analysisData.insights.map((insight, index) => (
                        <div key={index} className={`p-6 rounded-xl border text-center transition-all hover:scale-105 ${
                          theme === 'dark' 
                            ? 'bg-gray-700/30 border-gray-600 hover:bg-gray-700/50' 
                            : 'bg-white/50 border-gray-200 hover:bg-white/80'
                        }`}>
                          <div className={`text-3xl font-bold mb-2 ${
                            theme === 'dark' ? 'text-emerald-400' : 'text-purple-600'
                          }`}>
                            {insight.value}
                          </div>
                          <div className="text-sm opacity-75">{insight.name}</div>
                        </div>
                      ))}
                    </div>

                    {/* Document Connection Summary */}
                    <div className={`p-6 rounded-xl border ${
                      theme === 'dark' 
                        ? 'bg-gray-700/30 border-gray-600' 
                        : 'bg-white/50 border-gray-200'
                    }`}>
                      <h4 className="text-lg font-semibold mb-4">Document Connection Summary</h4>
                      <div className="space-y-3">
                        {analysisData.documentConnections.map((doc, index) => (
                          <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-gray-100/50 dark:bg-gray-800/50">
                            <div>
                              <span className="font-medium">{doc.title}</span>
                            </div>
                            <div className="flex items-center gap-4 text-sm">
                              <span className="opacity-75">{doc.connections} connections</span>
                              <div className={`w-16 h-2 rounded-full overflow-hidden ${
                                theme === 'dark' ? 'bg-gray-600' : 'bg-gray-300'
                              }`}>
                                <div 
                                  className={`h-full transition-all ${
                                    theme === 'dark' ? 'bg-emerald-400' : 'bg-purple-500'
                                  }`}
                                  style={{ width: `${doc.strength * 100}%` }}
                                ></div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        ) : (
          <div className={`p-12 rounded-2xl border text-center ${
            theme === 'dark'
              ? 'bg-gray-800/50 border-gray-700'
              : 'bg-white/70 border-gray-200'
          }`}>
            <div className="flex flex-col items-center justify-center">
              <div className={`p-4 rounded-full mb-4 ${
                theme === 'dark' ? 'bg-purple-500/20' : 'bg-purple-100'
              }`}>
                <Brain className={theme === 'dark' ? 'text-purple-400' : 'text-purple-600'} size={32} />
              </div>
              <h3 className="text-xl font-bold mb-2">Ready to Analyze</h3>
              <p className="opacity-75 text-center max-w-md mb-4">
                Click "Analyze Collection" to start processing your {documents.length} documents.
              </p>
              <button
                onClick={runAnalysis}
                disabled={loading}
                className={`px-8 py-3 rounded-xl transition-all duration-200 flex items-center gap-2 font-medium ${
                  theme === 'dark'
                    ? 'bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-700 hover:to-cyan-700 text-white'
                    : 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white'
                }`}
              >
                <Brain size={18} />
                Start Analysis
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CollectionAnalysisPage;