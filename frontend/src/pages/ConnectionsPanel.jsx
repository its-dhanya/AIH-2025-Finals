import React from 'react';
import { Zap, Loader2, Link, X } from 'lucide-react';

const ConnectionsPanel = ({ connections, isLoading, theme, onClose }) => {
  return (
    <div className={`fixed right-0 top-0 h-full w-96 max-w-[90vw] ${theme === 'dark' ? 'bg-black/90 border-white/20' : 'bg-white/95 border-gray-200/50'} backdrop-blur-xl border-l shadow-2xl z-50 transform transition-transform duration-300 ease-in-out`}>
      <div className={`p-6 border-b ${theme === 'dark' ? 'border-white/10' : 'border-gray-200/30'}`}>
        <div className="flex items-center justify-between">
            <h3 className={`text-xl font-bold mb-1 flex items-center gap-2 ${theme === 'dark' ? 'bg-gradient-to-r from-emerald-400 to-cyan-500' : 'bg-gradient-to-r from-violet-700 to-fuchsia-700'} bg-clip-text text-transparent`}>
            <Zap size={20} className={theme === 'dark' ? 'text-emerald-400' : 'text-violet-700'} />
            Connections
            </h3>
            <button onClick={onClose} className={`p-2 rounded-lg transition-all duration-200 hover:scale-110 ${theme === 'dark' ? 'hover:bg-white/10 text-white/90' : 'hover:bg-gray-100 text-gray-900'}`}><X size={20} /></button>
        </div>
        <p className={`text-sm ${theme === 'dark' ? 'text-white/90' : 'text-gray-900'}`}>Real-time insights from your library</p>
      </div>

      <div className="p-4 overflow-y-auto custom-scrollbar h-[calc(100vh-120px)]">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Loader2 size={32} className={`mx-auto mb-4 animate-spin ${theme === 'dark' ? 'text-emerald-400' : 'text-purple-600'}`} />
            <p className={`text-sm ${theme === 'dark' ? 'text-white/80' : 'text-gray-700'}`}>Finding connections...</p>
          </div>
        ) : connections && connections.length > 0 ? (
          <div className="space-y-3">
            {connections.map((conn, index) => (
              <div key={index} className={`p-4 rounded-xl border transition-all cursor-pointer ${theme === 'dark' ? 'bg-gray-800/50 border-gray-700 hover:border-cyan-400/50' : 'bg-blue-50/70 border-gray-200 hover:border-purple-400/50'}`}>
                <p className={`text-sm leading-relaxed mb-3 ${theme === 'dark' ? 'text-white/90' : 'text-gray-800'}`}>
                  "{conn.snippet}"
                </p>
                <div className="text-xs flex items-center justify-between">
                  <span className={`font-semibold truncate ${theme === 'dark' ? 'text-cyan-300' : 'text-purple-700'}`}>
                    {conn.source_document}
                  </span>
                  <span className={`${theme === 'dark' ? 'text-white/60' : 'text-gray-500'}`}>
                    Page {conn.page_number}
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Link size={32} className={`mx-auto mb-4 ${theme === 'dark' ? 'text-white/30' : 'text-gray-400'}`} />
            <p className={`text-sm max-w-xs ${theme === 'dark' ? 'text-white/80' : 'text-gray-700'}`}>
              Select text in the document to instantly discover related information from your library.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ConnectionsPanel;