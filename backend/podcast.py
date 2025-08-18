# backend/podcast.py - FIXED VERSION
import os
import tempfile
import uuid
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import shutil
import json
import gc

logger = logging.getLogger("podcast")
logging.basicConfig(level=logging.INFO)

# Try to import optional TTS backends with better error handling
HAS_COQUI = False
HAS_AZURE = False
HAS_PYTTSX3 = False
HAS_GTTS = False
HAS_PYDUB = False

try:
    from TTS.api import TTS
    HAS_COQUI = True
    logger.info("Coqui TTS available")
except Exception as e:
    logger.info("Coqui TTS not available: %s", e)

try:
    import azure.cognitiveservices.speech as speechsdk
    HAS_AZURE = True
    logger.info("Azure Speech SDK available")
except Exception as e:
    logger.info("Azure Speech SDK not available: %s", e)

try:
    import pyttsx3
    HAS_PYTTSX3 = True
    logger.info("pyttsx3 available")
except Exception as e:
    logger.info("pyttsx3 not available: %s", e)

try:
    from gtts import gTTS
    HAS_GTTS = True
    logger.info("gTTS available")
except Exception as e:
    logger.info("gTTS not available: %s", e)

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
    logger.info("pydub available")
except Exception as e:
    logger.error("pydub not available - required for audio processing: %s", e)

# Check if we have at least one working backend
def check_tts_availability():
    """Check if at least one TTS backend is available."""
    if not any([HAS_COQUI, HAS_AZURE, HAS_PYTTSX3, HAS_GTTS]):
        raise RuntimeError(
            "No TTS backend available. Install at least one of: "
            "pip install gTTS, pip install pyttsx3, "
            "pip install azure-cognitiveservices-speech, "
            "or pip install TTS"
        )
    if not HAS_PYDUB:
        raise RuntimeError(
            "pydub is required for audio processing. Install with: pip install pydub"
        )

# utility: write binary with error handling
def write_bytes(path: str, content: bytes):
    """Write bytes to file with proper error handling."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.error("Failed to write file %s: %s", path, e)
        raise

# ---------------------------
# Script generation - FIXED
# ---------------------------
def generate_podcast_script(insights: Dict[str, Any], mode: str = "dialogue") -> List[Dict[str, str]]:
    """
    Generate a podcast script from insights.
    FIXED: Better data validation and structure handling.
    """
    if not isinstance(insights, dict):
        raise ValueError("Insights must be a dictionary")
    
    # Safely extract and validate data
    summary = str(insights.get("summary", "")).strip()[:900] if insights.get("summary") else "Analysis completed"
    
    # Handle takeaways (can be list or string)
    takeaways_raw = insights.get("key_takeaways", [])
    if isinstance(takeaways_raw, str):
        takeaways = [takeaways_raw] if takeaways_raw.strip() else []
    elif isinstance(takeaways_raw, list):
        takeaways = [str(t).strip() for t in takeaways_raw if t and str(t).strip()][:4]
    else:
        takeaways = []
    
    # Handle did_you_know
    did_you_know_raw = insights.get("did_you_know", [])
    if isinstance(did_you_know_raw, str):
        did_you_know = [did_you_know_raw] if did_you_know_raw.strip() else []
    elif isinstance(did_you_know_raw, list):
        did_you_know = [str(d).strip() for d in did_you_know_raw if d and str(d).strip()][:3]
    else:
        did_you_know = []
    
    # Handle contradictions
    contradictions_raw = insights.get("contradictions", [])
    if isinstance(contradictions_raw, str):
        contradictions = [contradictions_raw] if contradictions_raw.strip() else []
    elif isinstance(contradictions_raw, list):
        contradictions = [str(c).strip() for c in contradictions_raw if c and str(c).strip()][:3]
    else:
        contradictions = []
    
    # Handle examples - FIXED to handle both formats
    examples = []
    examples_raw = insights.get("examples", []) or insights.get("example_objects", [])
    if examples_raw:
        for ex in examples_raw[:3]:
            if isinstance(ex, dict) and ex.get("text"):
                text = str(ex.get("text", "")).strip()
                if text:
                    examples.append({"text": text[:200]})  # Limit length for speech
            elif isinstance(ex, str) and ex.strip():
                examples.append({"text": ex.strip()[:200]})
    
    # Handle additional insights
    additional_raw = insights.get("additional_insights", [])
    if isinstance(additional_raw, str):
        additional = [additional_raw] if additional_raw.strip() else []
    elif isinstance(additional_raw, list):
        additional = [str(a).strip() for a in additional_raw if a and str(a).strip()][:2]
    else:
        additional = []

    # Build script
    segments = []
    
    # Intro
    host_intro = f"Welcome to today's podcast. {summary[:200]}" if summary else "Welcome to today's podcast."
    
    if mode == "overview":
        # Single speaker overview
        text_parts = [host_intro]
        
        if takeaways:
            text_parts.append("Key takeaways include: " + ". ".join(takeaways))
        
        if did_you_know:
            text_parts.append("Here are some interesting facts: " + ". ".join(did_you_know))
        
        if contradictions:
            text_parts.append("Important considerations: " + ". ".join(contradictions))
        
        if examples:
            ex_texts = [ex["text"][:100] for ex in examples]
            text_parts.append("For example: " + " Also, ".join(ex_texts))
        
        text_parts.append("That concludes our analysis. Thank you for listening.")
        
        full_text = " ".join(text_parts)
        segments.append({"speaker": "Host", "text": full_text})
        
    else:  # dialogue mode
        segments.append({"speaker": "Host", "text": host_intro})
        
        # Present takeaways
        for i, tk in enumerate(takeaways[:3], 1):
            segments.append({
                "speaker": "Host", 
                "text": f"Point {i}: {tk}. What do you think about this?"
            })
            
            guest_response = "That's an important observation."
            if i-1 < len(additional) and additional[i-1]:
                guest_response += f" {additional[i-1]}"
            else:
                guest_response += " This connects to broader themes we're seeing."
            
            segments.append({"speaker": "Guest", "text": guest_response})

        # Add interesting facts
        for fact in did_you_know[:2]:
            segments.append({"speaker": "Host", "text": f"Here's something interesting: {fact}"})
            segments.append({"speaker": "Guest", "text": "That's a valuable insight."})

        # Add contradictions
        for c in contradictions[:2]:
            segments.append({"speaker": "Host", "text": f"There's another perspective: {c}"})
            segments.append({"speaker": "Guest", "text": "Good point. It's important to consider multiple angles."})

        # Add examples
        for ex in examples[:2]:
            text = ex.get("text", "")[:150]  # Keep it concise for speech
            if text:
                segments.append({"speaker": "Host", "text": f"For instance: {text}"})
                segments.append({"speaker": "Guest", "text": "That example helps illustrate the concept clearly."})

        # Outro
        segments.append({"speaker": "Host", "text": "Thanks for the discussion. That's all for today."})
    
    # Validate segments
    valid_segments = []
    for seg in segments:
        if seg.get("text") and seg.get("speaker"):
            # Clean up text for TTS
            text = str(seg["text"]).strip()
            text = text.replace('"', "'")  # Avoid quote issues in TTS
            text = text.replace('\n', ' ')  # Remove line breaks
            if len(text) > 10:  # Minimum viable length
                valid_segments.append({"speaker": seg["speaker"], "text": text})
    
    if not valid_segments:
        # Fallback segment
        valid_segments = [{
            "speaker": "Host", 
            "text": "Welcome to our podcast. We've analyzed the content and found interesting insights. Thank you for listening."
        }]
    
    logger.info("Generated %d podcast segments", len(valid_segments))
    return valid_segments

# ---------------------------
# Synthesis functions - FIXED
# ---------------------------
def synthesize_with_coqui(text: str, model_name: Optional[str], out_path: str) -> str:
    """Synthesize using Coqui TTS with better error handling."""
    logger.info("Using Coqui TTS")
    try:
        model = model_name or "tts_models/en/ljspeech/tacotron2-DDC"
        tts = TTS(model_name=model, progress_bar=False, gpu=False)
        tts.tts_to_file(text=text, file_path=out_path)
        
        # Verify file was created
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("Coqui TTS produced no output")
        
        return out_path
    except Exception as e:
        logger.error("Coqui TTS failed: %s", e)
        raise

def synthesize_with_pyttsx3(text: str, out_path: str) -> str:
    """pyttsx3 synthesis with proper cleanup."""
    logger.info("Using pyttsx3 TTS")
    engine = None
    try:
        engine = pyttsx3.init()
        
        # Configure voice settings
        voices = engine.getProperty('voices')
        if voices and len(voices) > 0:
            engine.setProperty('voice', voices[0].id)
        
        engine.setProperty('rate', 160)    # Reasonable speaking rate
        engine.setProperty('volume', 0.9)  # Volume level
        
        # Generate audio
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        
        # Verify file creation
        if not os.path.exists(out_path):
            raise RuntimeError("pyttsx3 failed to create audio file")
        
        return out_path
        
    except Exception as e:
        logger.error("pyttsx3 TTS failed: %s", e)
        raise
    finally:
        if engine:
            try:
                engine.stop()
                del engine
            except:
                pass

def synthesize_with_gtts(text: str, lang: str, out_path: str) -> str:
    """gTTS synthesis with better error handling."""
    logger.info("Using gTTS")
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # gTTS needs MP3 extension
        temp_mp3 = out_path.replace('.wav', '.mp3') if out_path.endswith('.wav') else out_path
        tts.save(temp_mp3)
        
        # Verify file creation
        if not os.path.exists(temp_mp3) or os.path.getsize(temp_mp3) == 0:
            raise RuntimeError("gTTS produced no output")
        
        # Convert to WAV if needed
        if out_path.endswith('.wav') and temp_mp3.endswith('.mp3'):
            if not HAS_PYDUB:
                raise RuntimeError("pydub required for format conversion")
            
            audio = AudioSegment.from_mp3(temp_mp3)
            audio.export(out_path, format="wav")
            os.remove(temp_mp3)  # Clean up temp MP3
            return out_path
        
        return temp_mp3
        
    except Exception as e:
        logger.error("gTTS failed: %s", e)
        raise

def synthesize_with_azure(text: str, voice: str, out_path: str) -> str:
    """Azure TTS with improved error handling."""
    logger.info("Using Azure Speech SDK")
    
    key = os.environ.get("AZURE_TTS_KEY")
    region = os.environ.get("AZURE_TTS_REGION")
    
    if not key or not region:
        raise RuntimeError(
            "Azure TTS not configured. Set AZURE_TTS_KEY and AZURE_TTS_REGION environment variables."
        )

    try:
        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        speech_config.speech_synthesis_voice_name = voice or "en-US-JennyNeural"
        
        audio_config = speechsdk.audio.AudioOutputConfig(filename=out_path)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, 
            audio_config=audio_config
        )
        
        result = synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # Verify file creation
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                raise RuntimeError("Azure TTS produced no output file")
            return out_path
        else:
            error_details = result.error_details if hasattr(result, 'error_details') else "Unknown error"
            raise RuntimeError(f"Azure TTS failed: {result.reason} - {error_details}")
            
    except Exception as e:
        logger.error("Azure TTS failed: %s", e)
        raise

def synthesize_text_to_file(text: str, speaker: str = "default", lang: str = "en", out_path: str = None) -> str:
    """
    FIXED: Better TTS backend selection and error handling.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    # Clean text for TTS
    text = text.strip().replace('\n', ' ').replace('\r', ' ')
    if len(text) > 1000:  # Limit for TTS stability
        text = text[:1000] + "..."
    
    if not out_path:
        out_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.wav")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    provider = os.environ.get("TTS_PROVIDER", "").lower()
    logger.info(f"Synthesizing text (len={len(text)}) with provider: {provider or 'auto'}")

    # Track attempts for better error reporting
    attempts = []
    
    # Try Azure if configured
    if (provider == "azure" or not provider) and HAS_AZURE:
        key = os.environ.get("AZURE_TTS_KEY")
        region = os.environ.get("AZURE_TTS_REGION")
        if key and region:
            try:
                voice = os.environ.get("AZURE_TTS_VOICE", "en-US-JennyNeural")
                return synthesize_with_azure(text, voice, out_path)
            except Exception as e:
                attempts.append(f"Azure: {str(e)[:100]}")
                logger.warning("Azure TTS failed: %s", e)

    # Try gTTS (most reliable online option)
    if (provider in ["gtts", ""] or not attempts) and HAS_GTTS:
        try:
            return synthesize_with_gtts(text, lang, out_path)
        except Exception as e:
            attempts.append(f"gTTS: {str(e)[:100]}")
            logger.warning("gTTS failed: %s", e)

    # Try pyttsx3 (offline)
    if (provider in ["pyttsx3", "offline", ""] or not attempts) and HAS_PYTTSX3:
        try:
            return synthesize_with_pyttsx3(text, out_path)
        except Exception as e:
            attempts.append(f"pyttsx3: {str(e)[:100]}")
            logger.warning("pyttsx3 failed: %s", e)

    # Try Coqui as last resort (can be resource intensive)
    if (provider in ["coqui", "local"] or not attempts) and HAS_COQUI:
        try:
            model_name = os.environ.get("COQUI_TTS_MODEL")
            return synthesize_with_coqui(text, model_name, out_path)
        except Exception as e:
            attempts.append(f"Coqui: {str(e)[:100]}")
            logger.warning("Coqui TTS failed: %s", e)

    # If all failed, provide detailed error
    error_msg = f"All TTS backends failed. Attempts: {'; '.join(attempts)}"
    available_backends = [name for name, available in [
        ("gTTS", HAS_GTTS), ("pyttsx3", HAS_PYTTSX3), 
        ("Azure", HAS_AZURE), ("Coqui", HAS_COQUI)
    ] if available]
    
    if not available_backends:
        error_msg += " No TTS backends are installed."
    else:
        error_msg += f" Available backends: {', '.join(available_backends)}"
    
    raise RuntimeError(error_msg)

# ---------------------------
# Audio assembly - FIXED
# ---------------------------
def assemble_segments(segments: List[Dict[str, str]], gap_ms: int = 800) -> bytes:
    """
    FIXED: Better error handling and audio processing.
    """
    if not segments:
        raise ValueError("No segments to assemble")
    
    # Check dependencies
    if not HAS_PYDUB:
        raise RuntimeError("pydub is required for audio assembly")
    
    tmpdir = tempfile.mkdtemp(prefix="podcast_assembly_")
    audio_files = []
    
    try:
        logger.info(f"Assembling {len(segments)} segments with {gap_ms}ms gaps")
        
        # Synthesize each segment
        for i, segment in enumerate(segments):
            speaker = segment.get("speaker", "Host")
            text = segment.get("text", "").strip()
            
            if not text:
                logger.warning(f"Segment {i} has no text, skipping")
                continue
            
            logger.info(f"Synthesizing segment {i+1}/{len(segments)} ({speaker}): {text[:50]}...")
            
            segment_path = os.path.join(tmpdir, f"segment_{i:03d}.wav")
            
            try:
                synthesize_text_to_file(text, speaker=speaker, out_path=segment_path)
                
                # Verify the file exists and has content
                if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                    audio_files.append(segment_path)
                else:
                    logger.error(f"Segment {i} synthesis produced no valid audio file")
                    
            except Exception as e:
                logger.error(f"Failed to synthesize segment {i}: {e}")
                # Create a brief silence as fallback
                try:
                    silence = AudioSegment.silent(duration=1500)  # 1.5 seconds
                    silence.export(segment_path, format="wav")
                    audio_files.append(segment_path)
                except Exception as silence_error:
                    logger.error(f"Could not create silence fallback: {silence_error}")

        if not audio_files:
            raise RuntimeError("No audio segments were successfully created")

        # Combine all segments
        logger.info("Combining audio segments...")
        combined_audio = None
        gap = AudioSegment.silent(duration=gap_ms)
        
        for i, audio_file in enumerate(audio_files):
            try:
                # Load audio with format detection
                if audio_file.endswith('.mp3'):
                    segment_audio = AudioSegment.from_mp3(audio_file)
                else:
                    segment_audio = AudioSegment.from_wav(audio_file)
                
                # Normalize audio levels
                segment_audio = segment_audio.normalize()
                
                if combined_audio is None:
                    combined_audio = segment_audio
                else:
                    combined_audio = combined_audio + gap + segment_audio
                    
                logger.debug(f"Added segment {i+1}, total duration: {len(combined_audio)}ms")
                    
            except Exception as e:
                logger.error(f"Failed to process audio file {audio_file}: {e}")
                continue

        if combined_audio is None:
            logger.warning("No audio could be combined, creating fallback")
            combined_audio = AudioSegment.silent(duration=5000)  # 5 seconds of silence

        # Export final audio
        logger.info("Exporting final podcast...")
        final_path = os.path.join(tmpdir, "final_podcast.mp3")
        
        # Export with good quality settings
        combined_audio.export(
            final_path, 
            format="mp3", 
            bitrate="128k",
            parameters=["-ac", "1"]  # Mono for smaller file size
        )
        
        # Read final audio data
        if not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
            raise RuntimeError("Failed to export final podcast audio")
        
        with open(final_path, "rb") as f:
            audio_data = f.read()
        
        logger.info(f"Podcast assembly complete: {len(audio_data)} bytes, duration: ~{len(combined_audio)/1000:.1f}s")
        return audio_data

    except Exception as e:
        logger.error(f"Audio assembly failed: {e}")
        raise RuntimeError(f"Failed to assemble podcast audio: {str(e)}")
    finally:
        # Cleanup
        try:
            shutil.rmtree(tmpdir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory {tmpdir}: {e}")
        
        # Force garbage collection
        gc.collect()

# ---------------------------
# Main API function - FIXED
# ---------------------------
def create_podcast_from_insights(
    insights: Dict[str, Any], 
    filename: Optional[str] = None, 
    mode: str = "dialogue"
) -> str:
    """
    FIXED: Main function with comprehensive error handling.
    """
    if not insights:
        raise ValueError("Insights dictionary cannot be empty")

    # Validate mode
    if mode not in ["dialogue", "overview"]:
        logger.warning(f"Invalid mode '{mode}', defaulting to 'dialogue'")
        mode = "dialogue"

    # Check dependencies upfront
    check_tts_availability()

    logger.info(f"Creating podcast in {mode} mode from insights")
    
    try:
        # Generate script
        segments = generate_podcast_script(insights, mode=mode)
        
        if not segments:
            raise ValueError("No podcast segments were generated from insights")
        
        logger.info(f"Generated {len(segments)} script segments")
        
        # Assemble audio
        audio_data = assemble_segments(segments, gap_ms=800)
        
        if not audio_data or len(audio_data) < 1000:  # Sanity check
            raise RuntimeError("Generated audio data is too small or empty")
        
        # Write to output file
        if filename:
            out_path = filename
        else:
            out_path = os.path.join(tempfile.gettempdir(), f"podcast_{uuid.uuid4().hex}.mp3")
        
        write_bytes(out_path, audio_data)
        
        # Verify final file
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("Failed to write final podcast file")
        
        file_size = os.path.getsize(out_path)
        logger.info(f"Podcast created successfully: {out_path} ({file_size} bytes)")
        return out_path

    except Exception as e:
        logger.error(f"Podcast creation failed: {e}")
        raise RuntimeError(f"Failed to create podcast: {str(e)}")

# ---------------------------
# Test function - FIXED
# ---------------------------
def test_podcast_creation():
    """Test function with better error reporting."""
    test_insights = {
        "summary": "This is a test analysis of document insights covering key themes and findings.",
        "key_takeaways": [
            "First key insight about the main topic",
            "Second important finding from the analysis",
            "Third critical observation worth noting"
        ],
        "did_you_know": [
            "Interesting fact discovered in the research",
            "Surprising finding that emerged from the data"
        ],
        "contradictions": [
            "One area where sources present different viewpoints"
        ],
        "examples": [
            {"text": "Concrete example demonstrating the first principle"},
            {"text": "Another example showing practical application"}
        ],
        "additional_insights": [
            "Broader context for understanding these findings",
            "Implications for future research or application"
        ]
    }
    
    try:
        logger.info("Starting podcast test...")
        output_path = create_podcast_from_insights(test_insights, mode="dialogue")
        
        # Verify the result
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"✓ Test successful! Podcast created: {output_path} ({size} bytes)")
            return output_path
        else:
            print("✗ Test failed: Output file not found")
            return None
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        logger.exception("Test podcast creation failed")
        raise

if __name__ == "__main__":
    test_podcast_creation()