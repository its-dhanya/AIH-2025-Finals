import React, { useState, useEffect } from "react";
import { Mic, Download, RefreshCw, AlertTriangle, CheckCircle, Play, Users, User, Clock } from 'lucide-react';

/**
 * Enhanced PodcastPanel - Consistent with app theme and design
 * 
 * Props:
 *   - insights: object | null (structured insights if available)
 *   - selectedText: string
 *   - connections: array
 *   - theme: 'light' | 'dark'
 *   - className: optional string for wrapper styling
 */
export default function PodcastPanel({
  insights = null,
  selectedText = "",
  connections = [],
  theme = 'dark',
  className = ""
}) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [mode, setMode] = useState("dialogue");
  const [durationMinutes, setDurationMinutes] = useState(3);
  const [speakerA, setSpeakerA] = useState("Host");
  const [speakerB, setSpeakerB] = useState("Expert");
  const [podcastHealth, setPodcastHealth] = useState(null);
  const [isHealthLoading, setIsHealthLoading] = useState(true);

  // Cleanup audio URL on unmount
  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  // Check podcast health on mount
  useEffect(() => {
    checkPodcastHealth();
  }, []);

  const checkPodcastHealth = async () => {
    setIsHealthLoading(true);
    try {
      const response = await fetch("http://localhost:3000/api/podcast-health");
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }
      const health = await response.json();
      setPodcastHealth(health);
      
      if (!health.overall_healthy) {
        console.warn("Podcast service health issues:", health.errors);
      }
    } catch (err) {
      console.error("Podcast health check failed:", err);
      setPodcastHealth({ 
        overall_healthy: false, 
        errors: [`Health check failed: ${err.message}`],
        podcast_module_available: false
      });
    } finally {
      setIsHealthLoading(false);
    }
  };

  const handleGenerate = async () => {
    // Reset state
    setError(null);
    setSuccess(null);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    setIsGenerating(true);

    try {
      // Check health first
      if (podcastHealth && !podcastHealth.overall_healthy) {
        throw new Error("Podcast service is not available. Check the service health below.");
      }

      // Prepare payload
      const payload = {
        target_minutes: Math.max(1, Math.min(10, durationMinutes)),
        speaker_a: speakerA.trim() || "Host",
        speaker_b: speakerB.trim() || "Expert", 
        mode: mode,
        source_document: "Document Analysis"
      };

      // Add content - prefer insights over text
      if (insights && typeof insights === 'object') {
        payload.insights = insights;
        console.log("Using provided insights for podcast");
      } else if (selectedText || (connections && connections.length > 0)) {
        payload.selected_text = selectedText || "";
        payload.connections = connections || [];
        console.log(`Using selected text (${payload.selected_text.length} chars) and ${payload.connections.length} connections`);
      } else {
        throw new Error("No content available. Please provide insights, selected text, or document connections.");
      }

      console.log("Generating podcast with:", {
        mode: payload.mode,
        duration: payload.target_minutes,
        hasInsights: !!payload.insights,
        textLength: payload.selected_text?.length || 0,
        connectionsCount: payload.connections?.length || 0
      });

      const response = await fetch("http://localhost:3000/api/generate-podcast", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      // Handle different response types
      const contentType = response.headers.get("content-type") || "";
      
      if (!response.ok) {
        // Extract error message
        let errorMessage = `Request failed (${response.status})`;
        
        try {
          if (contentType.includes("application/json")) {
            const errorData = await response.json();
            errorMessage = errorData.detail || errorData.message || errorMessage;
          } else {
            const errorText = await response.text();
            errorMessage = errorText || errorMessage;
          }
        } catch (parseError) {
          console.error("Could not parse error response:", parseError);
        }
        
        throw new Error(errorMessage);
      }

      // Process successful response
      if (contentType.includes("audio")) {
        // Streaming audio response
        console.log("Received audio stream");
        const blob = await response.blob();
        
        if (blob.size === 0) {
          throw new Error("Received empty audio file");
        }
        
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
        setSuccess(`Podcast generated successfully! Duration: ~${durationMinutes} minutes`);
        
      } else if (contentType.includes("application/json")) {
        // JSON response with base64 audio
        console.log("Received JSON response");
        const data = await response.json();
        
        if (data.status === "success" && data.audio_base64) {
          try {
            // Decode base64 to blob
            const audioData = atob(data.audio_base64);
            const audioArray = new Uint8Array(audioData.length);
            for (let i = 0; i < audioData.length; i++) {
              audioArray[i] = audioData.charCodeAt(i);
            }
            
            const blob = new Blob([audioArray], { type: data.mime_type || "audio/mpeg" });
            const url = URL.createObjectURL(blob);
            setAudioUrl(url);
            setSuccess(`Podcast generated successfully! Duration: ~${durationMinutes} minutes`);
            
          } catch (decodeError) {
            throw new Error(`Failed to decode audio: ${decodeError.message}`);
          }
        } else {
          throw new Error(data.error || data.detail || "Unexpected response format");
        }
      } else {
        throw new Error(`Unexpected response type: ${contentType}`);
      }

    } catch (err) {
      console.error("Podcast generation error:", err);
      
      // User-friendly error messages
      let errorMsg = err.message || String(err);
      
      if (errorMsg.includes("503") || errorMsg.includes("service unavailable")) {
        errorMsg = "Podcast service is currently unavailable. Please check the service status below.";
      } else if (errorMsg.includes("TTS") || errorMsg.includes("text-to-speech")) {
        errorMsg = "Text-to-speech service is not configured. Please check with your administrator.";
      } else if (errorMsg.includes("pydub")) {
        errorMsg = "Audio processing library is not installed. Please check with your administrator.";
      } else if (errorMsg.includes("timeout")) {
        errorMsg = "Request timed out. Podcast generation can take time - please try again.";
      } else if (errorMsg.includes("400")) {
        errorMsg = "Invalid request data. Please check your input content.";
      } else if (errorMsg.includes("500")) {
        errorMsg = "Server error during generation. Please try again or contact support.";
      }
      
      setError(errorMsg);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = () => {
    if (!audioUrl) return;
    
    try {
      const link = document.createElement("a");
      link.href = audioUrl;
      link.download = `podcast_${new Date().toISOString().slice(0, 19).replace(/[:.]/g, "-")}.mp3`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err) {
      console.error("Download failed:", err);
      setError(`Download failed: ${err.message}`);
    }
  };

  const handleClear = () => {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    setError(null);
    setSuccess(null);
  };

  const canGenerate = () => {
    if (!podcastHealth || !podcastHealth.overall_healthy) return false;
    if (insights && typeof insights === 'object') return true;
    if (selectedText && selectedText.trim().length > 10) return true;
    if (connections && connections.length > 0) return true;
    return false;
  };

  const getContentStatus = () => {
    if (insights) {
      const summary = insights.summary || insights.metadata?.summary || "";
      return `AI insights available${summary ? ` (${summary.slice(0, 40)}...)` : ""}`;
    }
    if (selectedText && selectedText.length > 10) {
      return `Selected text (${selectedText.length} characters)`;
    }
    if (connections && connections.length > 0) {
      return `Document connections (${connections.length} found)`;
    }
    return "No content available";
  };

  return (
    <div className="space-y-4">
      {/* Service Health Status */}
      {isHealthLoading ? (
        <div className={`p-3 rounded-lg border ${
          theme === 'dark' 
            ? 'bg-gray-800/50 border-gray-700/50' 
            : 'bg-gray-50 border-gray-200'
        }`}>
          <div className="flex items-center gap-2">
            <div className={`animate-spin rounded-full h-4 w-4 border-b-2 ${
              theme === 'dark' ? 'border-cyan-400' : 'border-purple-600'
            }`}></div>
            <span className={`text-sm ${theme === 'dark' ? 'text-white/80' : 'text-gray-600'}`}>
              Checking service health...
            </span>
          </div>
        </div>
      ) : podcastHealth && !podcastHealth.overall_healthy ? (
        <div className={`p-3 rounded-lg border ${
          theme === 'dark' 
            ? 'bg-red-500/10 border-red-500/30' 
            : 'bg-red-50 border-red-200'
        }`}>
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle size={16} className="text-red-500" />
            <span className={`font-medium text-sm ${
              theme === 'dark' ? 'text-red-300' : 'text-red-800'
            }`}>
              Service Issues Detected
            </span>
          </div>
          {podcastHealth.errors && podcastHealth.errors.slice(0, 2).map((error, i) => (
            <div key={i} className={`text-sm ${
              theme === 'dark' ? 'text-red-200' : 'text-red-700'
            }`}>
              • {error}
            </div>
          ))}
        </div>
      ) : podcastHealth && podcastHealth.overall_healthy ? (
        <div className={`p-2 rounded-lg border ${
          theme === 'dark' 
            ? 'bg-green-500/10 border-green-500/30' 
            : 'bg-green-50 border-green-200'
        }`}>
          <div className="flex items-center gap-2">
            <CheckCircle size={16} className="text-green-500" />
            <div className={`text-sm ${theme === 'dark' ? 'text-green-300' : 'text-green-800'}`}>
              Service ready
            </div>
          </div>
        </div>
      ) : null}

      {/* Content Status */}
      <div className={`p-3 rounded-lg border ${
        theme === 'dark' 
          ? 'bg-gradient-to-br from-gray-800/50 to-gray-900/50 border-gray-700/50' 
          : 'bg-gradient-to-br from-white/80 to-gray-50/80 border-gray-300/50'
      } backdrop-blur-sm`}>
        <div className="flex items-center gap-2 mb-2">
          <Mic size={16} className={theme === 'dark' ? 'text-pink-400' : 'text-pink-600'} />
          <span className={`font-medium text-sm ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
            Content Source
          </span>
        </div>
        <p className={`text-sm ${
          theme === 'dark' ? 'text-white/70' : 'text-gray-700'
        }`}>
          {getContentStatus()}
        </p>
      </div>

      {/* Configuration */}
      <div className="space-y-3">
        {/* Mode and Duration */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className={`text-xs font-medium block mb-1 ${
              theme === 'dark' ? 'text-white/90' : 'text-gray-700'
            }`}>
              Format
            </label>
            <div className={`relative ${
              theme === 'dark' ? 'text-white' : 'text-gray-900'
            }`}>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                className={`w-full px-3 py-2 rounded-lg border text-sm focus:ring-2 focus:ring-pink-500/50 focus:border-transparent ${
                  theme === 'dark' 
                    ? 'bg-black/20 border-white/20 text-white' 
                    : 'bg-white border-gray-300 text-black'
                }`}
              >
                <option value="dialogue">Dialogue</option>
                <option value="overview">Overview</option>
              </select>
              <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none">
                {mode === 'dialogue' ? <Users size={14} /> : <User size={14} />}
              </div>
            </div>
          </div>

          <div>
            <label className={`text-xs font-medium block mb-1 ${
              theme === 'dark' ? 'text-white/90' : 'text-gray-700'
            }`}>
              Duration
            </label>
            <div className={`relative ${
              theme === 'dark' ? 'text-white' : 'text-gray-900'
            }`}>
              <input
                type="number"
                min={1}
                max={10}
                value={durationMinutes}
                onChange={(e) => setDurationMinutes(parseInt(e.target.value) || 3)}
                className={`w-full px-3 py-2 pr-8 rounded-lg border text-sm focus:ring-2 focus:ring-pink-500/50 focus:border-transparent ${
                  theme === 'dark' 
                    ? 'bg-black/20 border-white/20 text-white placeholder-gray-400' 
                    : 'bg-white border-gray-300 text-black placeholder-gray-500'
                }`}
              />
              <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none">
                <Clock size={14} />
              </div>
            </div>
          </div>
        </div>

        {/* Speaker Configuration (dialogue mode only) */}
        {mode === "dialogue" && (
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className={`text-xs font-medium block mb-1 ${
                theme === 'dark' ? 'text-white/90' : 'text-gray-700'
              }`}>
                Speaker A
              </label>
              <input 
                value={speakerA} 
                onChange={(e) => setSpeakerA(e.target.value)} 
                className={`w-full px-3 py-2 rounded-lg border text-sm focus:ring-2 focus:ring-pink-500/50 focus:border-transparent ${
                  theme === 'dark' 
                    ? 'bg-black/20 border-white/20 text-white placeholder-gray-400' 
                    : 'bg-white border-gray-300 text-black placeholder-gray-500'
                }`}
                placeholder="Host"
                maxLength={20}
              />
            </div>
            <div>
              <label className={`text-xs font-medium block mb-1 ${
                theme === 'dark' ? 'text-white/90' : 'text-gray-700'
              }`}>
                Speaker B
              </label>
              <input 
                value={speakerB} 
                onChange={(e) => setSpeakerB(e.target.value)} 
                className={`w-full px-3 py-2 rounded-lg border text-sm focus:ring-2 focus:ring-pink-500/50 focus:border-transparent ${
                  theme === 'dark' 
                    ? 'bg-black/20 border-white/20 text-white placeholder-gray-400' 
                    : 'bg-white border-gray-300 text-black placeholder-gray-500'
                }`}
                placeholder="Expert"
                maxLength={20}
              />
            </div>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex gap-2">
        <button
          className={`flex-1 px-4 py-3 rounded-lg font-semibold text-sm transition-all duration-200 flex items-center justify-center gap-2 ${
            canGenerate() && !isGenerating
              ? 'bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-700 hover:to-purple-700 text-white shadow-lg hover:shadow-xl hover:scale-105'
              : theme === 'dark'
                ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                : 'bg-gray-200 text-gray-500 cursor-not-allowed'
          }`}
          disabled={isGenerating || !canGenerate()}
          onClick={handleGenerate}
        >
          {isGenerating ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              Generating...
            </>
          ) : (
            <>
              <Mic size={16} />
              Generate
            </>
          )}
        </button>

        {audioUrl && (
          <button 
            className="px-4 py-3 rounded-lg font-semibold text-sm bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white transition-all duration-200 flex items-center gap-2 shadow-lg hover:shadow-xl hover:scale-105"
            onClick={handleDownload}
          >
            <Download size={16} />
            Download
          </button>
        )}

        <button
          className={`px-3 py-3 rounded-lg font-semibold text-sm transition-all duration-200 ${
            theme === 'dark'
              ? 'bg-white/10 hover:bg-white/20 text-white/90'
              : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
          }`}
          onClick={handleClear}
          disabled={isGenerating}
          title="Clear"
        >
          ×
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className={`p-4 rounded-lg border ${
          theme === 'dark' 
            ? 'bg-red-500/10 border-red-500/30' 
            : 'bg-red-50 border-red-200'
        }`}>
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle size={16} className="text-red-500" />
            <span className={`font-medium text-sm ${
              theme === 'dark' ? 'text-red-300' : 'text-red-800'
            }`}>
              Generation Failed
            </span>
          </div>
          <p className={`text-sm ${
            theme === 'dark' ? 'text-red-200' : 'text-red-700'
          }`}>
            {error}
          </p>
        </div>
      )}

      {/* Success Display */}
      {success && (
        <div className={`p-4 rounded-lg border ${
          theme === 'dark' 
            ? 'bg-green-500/10 border-green-500/30' 
            : 'bg-green-50 border-green-200'
        }`}>
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle size={16} className="text-green-500" />
            <span className={`font-medium text-sm ${
              theme === 'dark' ? 'text-green-300' : 'text-green-800'
            }`}>
              Success!
            </span>
          </div>
          <p className={`text-sm ${
            theme === 'dark' ? 'text-green-200' : 'text-green-700'
          }`}>
            {success}
          </p>
        </div>
      )}

      {/* Audio Player */}
      {audioUrl && (
        <div className={`p-4 rounded-lg border ${
          theme === 'dark' 
            ? 'bg-gradient-to-br from-pink-900/30 to-purple-900/30 border-pink-500/30' 
            : 'bg-gradient-to-br from-pink-50 to-purple-50 border-pink-200'
        }`}>
          <div className="flex items-center gap-2 mb-3">
            <Play size={16} className={theme === 'dark' ? 'text-pink-400' : 'text-pink-600'} />
            <span className={`font-medium text-sm ${theme === 'dark' ? 'text-white' : 'text-gray-900'}`}>
              Generated Podcast
            </span>
          </div>
          <audio 
            controls 
            src={audioUrl} 
            className="w-full rounded-lg"
            preload="metadata"
            onError={(e) => {
              console.error("Audio playback error:", e);
              setError("Audio playback failed. Try downloading the file instead.");
            }}
          >
            Your browser does not support the audio element.
          </audio>
        </div>
      )}

      {/* Help Text */}
      {!canGenerate() && (
        <div className={`p-4 rounded-lg border-l-4 ${
          theme === 'dark' 
            ? 'bg-blue-500/10 border-blue-400 text-blue-200' 
            : 'bg-blue-50 border-blue-400 text-blue-800'
        }`}>
          <div className={`font-medium text-sm mb-2 ${
            theme === 'dark' ? 'text-blue-300' : 'text-blue-800'
          }`}>
            💡 How to Generate a Podcast
          </div>
          <ul className={`list-disc list-inside space-y-1 text-sm ${
            theme === 'dark' ? 'text-blue-200' : 'text-blue-700'
          }`}>
            <li>Ensure the service is healthy (green status)</li>
            <li>Generate AI insights or select text in the document</li>
            <li>Choose your preferred format and duration</li>
            <li>Click Generate to create your audio podcast</li>
          </ul>
        </div>
      )}
    </div>
  );
}