# backend/tts_unified.py - PROPERLY INTEGRATED VERSION
import os
import tempfile
import uuid
import logging
import subprocess
import requests
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import gc

logger = logging.getLogger("tts_unified")
logging.basicConfig(level=logging.INFO)

# Track which TTS backend produced the last successful audio
LAST_TTS_PROVIDER: Optional[str] = None

def get_last_tts_provider() -> Optional[str]:
    """Return the last TTS provider that produced audio (or None)."""
    return LAST_TTS_PROVIDER

# Try to import optional TTS backends with better error handling
HAS_COQUI = False
HAS_AZURE_SDK = False
HAS_PYTTSX3 = False
HAS_GTTS = False
HAS_PYDUB = False
HAS_GCP = False

try:
    from TTS.api import TTS
    HAS_COQUI = True
    logger.info("Coqui TTS available")
except Exception as e:
    logger.info("Coqui TTS not available: %s", e)

try:
    import azure.cognitiveservices.speech as speechsdk
    HAS_AZURE_SDK = True
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

try:
    from google.cloud import texttospeech
    HAS_GCP = True
    logger.info("Google Cloud TTS available")
except Exception as e:
    logger.info("Google Cloud TTS not available: %s", e)

# ---------------------------
# UNIFIED TTS INTERFACE
# ---------------------------

def generate_audio(text, output_file, provider=None, voice=None):
    """
    UNIFIED TTS INTERFACE - Main entry point for all TTS operations
    
    This function routes requests to appropriate TTS backends based on provider.
    Supports chunking for cloud providers and falls back gracefully.
    
    Args:
        text (str): Text to convert to speech
        output_file (str): Output file path
        provider (str, optional): TTS provider ("azure", "gcp", "gtts", "pyttsx3", "coqui", "local")
        voice (str, optional): Voice to use (provider-specific)
    
    Returns:
        str: Path to the generated audio file
    """
    global LAST_TTS_PROVIDER
    
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    # Clean text for TTS
    text = text.strip().replace('\n', ' ').replace('\r', ' ')
    
    # Determine provider priority
    provider = (provider or os.getenv("TTS_PROVIDER", "auto")).lower()
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle chunking for cloud providers
    max_chars_env = os.getenv("TTS_CLOUD_MAX_CHARS", "3000")
    max_chars = None
    try:
        max_chars = int(max_chars_env)
        if max_chars <= 0:
            max_chars = None
    except:
        max_chars = 3000

    # If cloud provider and text is long, use chunking
    if provider in ("azure", "gcp") and max_chars and len(text) > max_chars:
        return _generate_chunked_audio(text, output_file, provider, voice, max_chars)
    
    # Route to specific provider
    try:
        if provider == "azure" or (provider == "auto" and _is_azure_available()):
            return _generate_azure_openai_tts(text, output_file, voice)
        elif provider == "gcp" or (provider == "auto" and _is_gcp_available()):
            return _generate_gcp_tts(text, output_file, voice)
        elif provider == "gtts" or (provider == "auto" and HAS_GTTS):
            return _generate_gtts_tts(text, output_file, voice)
        elif provider == "azure_sdk" and HAS_AZURE_SDK:
            return _generate_azure_sdk_tts(text, output_file, voice)
        elif provider == "pyttsx3" or (provider == "auto" and HAS_PYTTSX3):
            return _generate_pyttsx3_tts(text, output_file, voice)
        elif provider == "coqui" and HAS_COQUI:
            return _generate_coqui_tts(text, output_file, voice)
        elif provider == "local":
            return _generate_local_tts(text, output_file, voice)
        elif provider == "auto":
            # Try providers in order of preference
            return _generate_auto_tts(text, output_file, voice)
        else:
            raise ValueError(f"Unsupported or unavailable TTS provider: {provider}")
            
    except Exception as e:
        logger.error(f"TTS generation failed with {provider}: {e}")
        if provider != "auto":
            # Try fallback if specific provider failed
            logger.info("Attempting fallback to auto provider selection")
            return _generate_auto_tts(text, output_file, voice)
        else:
            raise RuntimeError(f"All TTS providers failed: {str(e)}")

def _generate_auto_tts(text, output_file, voice):
    """Auto-select best available TTS provider."""
    global LAST_TTS_PROVIDER
    
    # Try providers in order of preference
    providers_to_try = [
        ("azure", _generate_azure_openai_tts, _is_azure_available),
        ("gcp", _generate_gcp_tts, _is_gcp_available),
        ("gtts", _generate_gtts_tts, lambda: HAS_GTTS),
        ("pyttsx3", _generate_pyttsx3_tts, lambda: HAS_PYTTSX3),
        ("coqui", _generate_coqui_tts, lambda: HAS_COQUI),
        ("local", _generate_local_tts, lambda: True),  # Always available as last resort
    ]
    
    attempts = []
    for provider_name, provider_func, availability_check in providers_to_try:
        if not availability_check():
            continue
            
        try:
            logger.info(f"Trying {provider_name} TTS...")
            result = provider_func(text, output_file, voice)
            LAST_TTS_PROVIDER = provider_name
            logger.info(f"Success with {provider_name} TTS")
            return result
        except Exception as e:
            attempts.append(f"{provider_name}: {str(e)[:100]}")
            logger.warning(f"{provider_name} TTS failed: {e}")
            continue
    
    raise RuntimeError(f"All TTS providers failed. Attempts: {'; '.join(attempts)}")

# ---------------------------
# PROVIDER AVAILABILITY CHECKS
# ---------------------------

def _is_azure_available():
    """Check if Azure OpenAI TTS is available."""
    return bool(os.getenv("AZURE_TTS_KEY") and os.getenv("AZURE_TTS_ENDPOINT"))

def _is_gcp_available():
    """Check if Google Cloud TTS is available."""
    return HAS_GCP and (bool(os.getenv("GOOGLE_API_KEY")) or bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")))

def _test_provider(provider):
    """Test if a specific provider is available."""
    try:
        if provider == "local":
            result = subprocess.run(['espeak-ng', '--version'], capture_output=True, timeout=5)
            return result.returncode == 0
        elif provider == "azure":
            return _is_azure_available()
        elif provider == "gcp":
            return _is_gcp_available()
        elif provider == "gtts":
            return HAS_GTTS
        elif provider == "pyttsx3":
            return HAS_PYTTSX3
        elif provider == "coqui":
            return HAS_COQUI
        return False
    except:
        return False

# ---------------------------
# CHUNKING SUPPORT
# ---------------------------

def _chunk_text_by_chars(text, max_chars):
    """Split text into chunks not exceeding max_chars, preferring whitespace boundaries."""
    import re

    if len(text) <= max_chars:
        return [text]

    tokens = re.findall(r"\S+\s*", text)
    chunks = []
    current = ""

    for token in tokens:
        if len(current) + len(token) <= max_chars:
            current += token
        else:
            if current:
                chunks.append(current.strip())
                current = ""
            # If token itself is longer than max_chars, split it
            if len(token) > max_chars:
                start = 0
                while start < len(token):
                    part = token[start:start + max_chars]
                    part = part.strip()
                    if part:
                        chunks.append(part)
                    start += max_chars
            else:
                current = token

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]

def _generate_chunked_audio(text, output_file, provider, voice, max_chars):
    """Generate audio from long text using chunking."""
    if not HAS_PYDUB:
        raise RuntimeError("Chunking requires pydub. Please install pydub.")
    
    chunks = _chunk_text_by_chars(text, max_chars)
    output_path = Path(output_file)
    
    temp_dir = Path(tempfile.gettempdir()) / f"tts_chunks_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    temp_files = []
    try:
        logger.info(f"Processing {len(chunks)} chunks for {provider} TTS")
        
        for i, chunk in enumerate(chunks):
            temp_file = str(temp_dir / f"chunk_{i:03d}.mp3")
            
            # Generate chunk audio using non-chunked version
            if provider == "azure":
                _generate_azure_openai_tts(chunk, temp_file, voice)
            elif provider == "gcp":
                _generate_gcp_tts(chunk, temp_file, voice)
            else:
                raise ValueError("Chunked synthesis only supported for azure/gcp providers")
                
            temp_files.append(temp_file)

        # Combine chunks
        combined_audio = None
        for temp_file in temp_files:
            segment = AudioSegment.from_file(temp_file)
            combined_audio = segment if combined_audio is None else (combined_audio + segment)

        # Export final audio
        suffix = output_path.suffix.lower().lstrip(".") or "mp3"
        combined_audio.export(str(output_path), format=suffix)
        
        logger.info(f"Chunked {provider} TTS audio saved: {output_file} ({len(chunks)} chunks)")
        return str(output_path)
        
    finally:
        # Cleanup
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

# ---------------------------
# TTS PROVIDER IMPLEMENTATIONS
# ---------------------------

def _generate_azure_openai_tts(text, output_file, voice=None):
    """Generate audio using Azure OpenAI TTS API."""
    global LAST_TTS_PROVIDER
    
    api_key = os.getenv("AZURE_TTS_KEY")
    endpoint = os.getenv("AZURE_TTS_ENDPOINT")
    deployment = os.getenv("AZURE_TTS_DEPLOYMENT", "tts")
    voice = voice or os.getenv("AZURE_TTS_VOICE", "alloy")
    api_version = os.getenv("AZURE_TTS_API_VERSION", "2025-03-01-preview")
    
    if not api_key or not endpoint:
        raise ValueError("AZURE_TTS_KEY and AZURE_TTS_ENDPOINT must be set")
    
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": deployment,
        "input": text,
        "voice": voice,
    }
    
    try:
        url = f"{endpoint}/openai/deployments/{deployment}/audio/speech?api-version={api_version}"
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        LAST_TTS_PROVIDER = "azure"
        logger.info(f"Azure OpenAI TTS audio saved: {output_file}")
        return output_file
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Azure OpenAI TTS failed: {e}")

def _generate_gcp_tts(text, output_file, voice=None):
    """Generate audio using Google Cloud Text-to-Speech."""
    global LAST_TTS_PROVIDER
    
    api_key = os.getenv("GOOGLE_API_KEY")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    gcp_voice = voice or os.getenv("GCP_TTS_VOICE", "en-US-Neural2-F")
    language = os.getenv("GCP_TTS_LANGUAGE", "en-US")
    
    if not api_key and not credentials_path:
        raise ValueError("Either GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS must be set")
    
    try:
        if api_key:
            # Use REST API with API key
            url = "https://texttospeech.googleapis.com/v1/text:synthesize"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": {"text": text},
                "voice": {"languageCode": language, "name": gcp_voice},
                "audioConfig": {"audioEncoding": "MP3"}
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            import base64
            audio_content = base64.b64decode(response.json()["audioContent"])
            
        else:
            # Use service account credentials
            if not HAS_GCP:
                raise RuntimeError("google-cloud-texttospeech not installed")
                
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            client = texttospeech.TextToSpeechClient()
            
            input_text = texttospeech.SynthesisInput(text=text)
            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language, name=gcp_voice
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            
            response = client.synthesize_speech(
                input=input_text, voice=voice_params, audio_config=audio_config
            )
            audio_content = response.audio_content

        with open(output_file, "wb") as f:
            f.write(audio_content)
        
        LAST_TTS_PROVIDER = "gcp"
        logger.info(f"Google Cloud TTS audio saved: {output_file}")
        return output_file
        
    except Exception as e:
        raise RuntimeError(f"Google Cloud TTS failed: {e}")

def _generate_gtts_tts(text, output_file, voice=None):
    """Generate audio using gTTS."""
    global LAST_TTS_PROVIDER
    
    if not HAS_GTTS:
        raise RuntimeError("gTTS not available")
    
    lang = voice or "en"
    
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # gTTS outputs MP3
        if output_file.endswith('.wav') and HAS_PYDUB:
            temp_mp3 = output_file.replace('.wav', '_temp.mp3')
            tts.save(temp_mp3)
            audio = AudioSegment.from_mp3(temp_mp3)
            audio.export(output_file, format="wav")
            os.remove(temp_mp3)
        else:
            tts.save(output_file)
        
        LAST_TTS_PROVIDER = "gtts"
        logger.info(f"gTTS audio saved: {output_file}")
        return output_file
        
    except Exception as e:
        raise RuntimeError(f"gTTS failed: {e}")

def _generate_azure_sdk_tts(text, output_file, voice=None):
    """Generate audio using Azure Speech SDK."""
    global LAST_TTS_PROVIDER
    
    if not HAS_AZURE_SDK:
        raise RuntimeError("Azure Speech SDK not available")
    
    key = os.environ.get("AZURE_TTS_KEY")
    region = os.environ.get("AZURE_TTS_REGION")
    
    if not key or not region:
        raise RuntimeError("AZURE_TTS_KEY and AZURE_TTS_REGION must be set")

    try:
        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        speech_config.speech_synthesis_voice_name = voice or "en-US-JennyNeural"
        
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=audio_config
        )
        
        result = synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            LAST_TTS_PROVIDER = "azure_sdk"
            return output_file
        else:
            error_details = getattr(result, 'error_details', 'Unknown error')
            raise RuntimeError(f"Azure SDK TTS failed: {result.reason} - {error_details}")
            
    except Exception as e:
        raise RuntimeError(f"Azure SDK TTS failed: {e}")

def _generate_pyttsx3_tts(text, output_file, voice=None):
    """Generate audio using pyttsx3."""
    global LAST_TTS_PROVIDER
    
    if not HAS_PYTTSX3:
        raise RuntimeError("pyttsx3 not available")
    
    engine = None
    try:
        engine = pyttsx3.init()
        
        # Configure voice
        voices = engine.getProperty('voices')
        if voices and voice:
            # Try to find requested voice
            for v in voices:
                if voice.lower() in v.id.lower() or voice.lower() in v.name.lower():
                    engine.setProperty('voice', v.id)
                    break
        elif voices:
            engine.setProperty('voice', voices[0].id)
        
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 0.9)
        
        engine.save_to_file(text, output_file)
        engine.runAndWait()
        
        if not os.path.exists(output_file):
            raise RuntimeError("pyttsx3 failed to create audio file")
        
        LAST_TTS_PROVIDER = "pyttsx3"
        logger.info(f"pyttsx3 audio saved: {output_file}")
        return output_file
        
    except Exception as e:
        raise RuntimeError(f"pyttsx3 failed: {e}")
    finally:
        if engine:
            try:
                engine.stop()
                del engine
            except:
                pass

def _generate_coqui_tts(text, output_file, voice=None):
    """Generate audio using Coqui TTS."""
    global LAST_TTS_PROVIDER
    
    if not HAS_COQUI:
        raise RuntimeError("Coqui TTS not available")
    
    try:
        model_name = voice or os.environ.get("COQUI_TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")
        tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
        tts.tts_to_file(text=text, file_path=output_file)
        
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            raise RuntimeError("Coqui TTS produced no output")
        
        LAST_TTS_PROVIDER = "coqui"
        logger.info(f"Coqui TTS audio saved: {output_file}")
        return output_file
        
    except Exception as e:
        raise RuntimeError(f"Coqui TTS failed: {e}")

def _generate_local_tts(text, output_file, voice=None):
    """Generate audio using local espeak-ng."""
    global LAST_TTS_PROVIDER
    
    espeak_voice = voice or os.getenv("ESPEAK_VOICE", "en")
    espeak_speed = os.getenv("ESPEAK_SPEED", "150")
    
    temp_wav_file = output_file.replace('.mp3', '.wav') if output_file.endswith('.mp3') else output_file
    
    try:
        cmd = [
            'espeak-ng',
            '-v', espeak_voice,
            '-s', str(espeak_speed),
            '-w', temp_wav_file,
            text
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            raise RuntimeError(f"espeak-ng failed: {result.stderr}")
        
        if not os.path.exists(temp_wav_file):
            raise RuntimeError("espeak-ng did not create output file")
        
        # Convert to MP3 if needed
        if output_file.endswith('.mp3') and temp_wav_file != output_file:
            if not HAS_PYDUB:
                raise RuntimeError("pydub required for MP3 conversion")
            
            audio = AudioSegment.from_wav(temp_wav_file)
            audio.export(output_file, format="mp3")
            os.remove(temp_wav_file)
        
        LAST_TTS_PROVIDER = "local"
        logger.info(f"Local TTS audio saved: {output_file}")
        return output_file
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("espeak-ng synthesis timed out")
    except FileNotFoundError:
        raise RuntimeError("espeak-ng not installed. Install: sudo apt-get install espeak-ng")
    except Exception as e:
        raise RuntimeError(f"Local TTS failed: {e}")

# ---------------------------
# PODCAST GENERATION (from original file)
# ---------------------------

def generate_podcast_script(insights: Dict[str, Any], mode: str = "dialogue") -> List[Dict[str, str]]:
    """Generate a podcast script from insights."""
    if not isinstance(insights, dict):
        raise ValueError("Insights must be a dictionary")
    
    # Safely extract and validate data (same as original implementation)
    summary = str(insights.get("summary", "")).strip()[:900] if insights.get("summary") else "Analysis completed"
    
    # Handle takeaways
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
    
    # Handle examples
    examples = []
    examples_raw = insights.get("examples", []) or insights.get("example_objects", [])
    if examples_raw:
        for ex in examples_raw[:3]:
            if isinstance(ex, dict) and ex.get("text"):
                text = str(ex.get("text", "")).strip()
                if text:
                    examples.append({"text": text[:200]})
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

    # Build script (same logic as original)
    segments = []
    
    host_intro = f"Welcome to today's podcast. {summary[:200]}" if summary else "Welcome to today's podcast."
    
    if mode == "overview":
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

        for fact in did_you_know[:2]:
            segments.append({"speaker": "Host", "text": f"Here's something interesting: {fact}"})
            segments.append({"speaker": "Guest", "text": "That's a valuable insight."})

        for c in contradictions[:2]:
            segments.append({"speaker": "Host", "text": f"There's another perspective: {c}"})
            segments.append({"speaker": "Guest", "text": "Good point. It's important to consider multiple angles."})

        for ex in examples[:2]:
            text = ex.get("text", "")[:150]
            if text:
                segments.append({"speaker": "Host", "text": f"For instance: {text}"})
                segments.append({"speaker": "Guest", "text": "That example helps illustrate the concept clearly."})

        segments.append({"speaker": "Host", "text": "Thanks for the discussion. That's all for today."})
    
    # Validate segments
    valid_segments = []
    for seg in segments:
        if seg.get("text") and seg.get("speaker"):
            text = str(seg["text"]).strip().replace('"', "'").replace('\n', ' ')
            if len(text) > 10:
                valid_segments.append({"speaker": seg["speaker"], "text": text})
    
    if not valid_segments:
        valid_segments = [{
            "speaker": "Host", 
            "text": "Welcome to our podcast. We've analyzed the content and found interesting insights. Thank you for listening."
        }]
    
    logger.info("Generated %d podcast segments", len(valid_segments))
    return valid_segments

def create_podcast_from_insights(
    insights: Dict[str, Any], 
    filename: Optional[str] = None, 
    mode: str = "dialogue"
) -> str:
    """
    Create a complete podcast from insights using the unified TTS system.
    
    Args:
        insights: Dictionary containing analysis insights
        filename: Optional output filename
        mode: "dialogue" or "overview"
    
    Returns:
        str: Path to the generated podcast file
    """
    if not insights:
        raise ValueError("Insights dictionary cannot be empty")

    if mode not in ["dialogue", "overview"]:
        logger.warning(f"Invalid mode '{mode}', defaulting to 'dialogue'")
        mode = "dialogue"

    logger.info(f"Creating podcast in {mode} mode from insights")
    
    try:
        # Generate script
        segments = generate_podcast_script(insights, mode=mode)
        
        if not segments:
            raise ValueError("No podcast segments were generated from insights")
        
        logger.info(f"Generated {len(segments)} script segments")
        
        # Generate audio for each segment and combine
        if filename:
            out_path = filename
        else:
            out_path = os.path.join(tempfile.gettempdir(), f"podcast_{uuid.uuid4().hex}.mp3")
        
        # Create output directory
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        
        if not HAS_PYDUB:
            raise RuntimeError("pydub is required for podcast assembly")
        
        # Create temporary directory for segments
        tmpdir = tempfile.mkdtemp(prefix="podcast_assembly_")
        audio_files = []
        
        try:
            # Generate each segment
            for i, segment in enumerate(segments):
                speaker = segment.get("speaker", "Host")
                text = segment.get("text", "").strip()
                
                if not text:
                    logger.warning(f"Segment {i} has no text, skipping")
                    continue
                
                logger.info(f"Synthesizing segment {i+1}/{len(segments)} ({speaker}): {text[:50]}...")
                
                segment_path = os.path.join(tmpdir, f"segment_{i:03d}.mp3")
                
                try:
                    # Use different voices for different speakers if supported
                    voice = None
                    provider = os.getenv("TTS_PROVIDER", "auto").lower()
                    
                    if provider == "azure":
                        voice = "alloy" if speaker == "Host" else "echo"
                    elif provider == "gcp":
                        voice = "en-US-Neural2-F" if speaker == "Host" else "en-US-Neural2-A"
                    
                    # Generate audio using unified interface
                    generate_audio(text, segment_path, voice=voice)
                    
                    # Verify the file exists and has content
                    if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                        audio_files.append(segment_path)
                    else:
                        logger.error(f"Segment {i} synthesis produced no valid audio file")
                        
                except Exception as e:
                    logger.error(f"Failed to synthesize segment {i}: {e}")
                    # Create brief silence as fallback
                    try:
                        silence = AudioSegment.silent(duration=1500)
                        silence.export(segment_path, format="mp3")
                        audio_files.append(segment_path)
                    except Exception as silence_error:
                        logger.error(f"Could not create silence fallback: {silence_error}")

            if not audio_files:
                raise RuntimeError("No audio segments were successfully created")

            # Combine all segments
            logger.info("Combining audio segments...")
            combined_audio = None
            gap = AudioSegment.silent(duration=800)  # 800ms gap between segments
            
            for i, audio_file in enumerate(audio_files):
                try:
                    segment_audio = AudioSegment.from_file(audio_file)
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
                combined_audio = AudioSegment.silent(duration=5000)

            # Export final audio
            logger.info("Exporting final podcast...")
            combined_audio.export(
                out_path, 
                format="mp3", 
                bitrate="128k",
                parameters=["-ac", "1"]  # Mono for smaller file size
            )
            
            # Verify final file
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                raise RuntimeError("Failed to export final podcast audio")
            
            file_size = os.path.getsize(out_path)
            duration_seconds = len(combined_audio) / 1000
            logger.info(f"Podcast created successfully: {out_path} ({file_size} bytes, {duration_seconds:.1f}s)")
            return out_path

        finally:
            # Cleanup temporary directory
            try:
                shutil.rmtree(tmpdir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory {tmpdir}: {e}")
            
            gc.collect()

    except Exception as e:
        logger.error(f"Podcast creation failed: {e}")
        raise RuntimeError(f"Failed to create podcast: {str(e)}")

# ---------------------------
# UTILITY AND TEST FUNCTIONS
# ---------------------------

def list_available_providers():
    """List all available TTS providers and their status."""
    providers = {
        "azure": "Azure OpenAI TTS (requires AZURE_TTS_KEY and AZURE_TTS_ENDPOINT)",
        "gcp": "Google Cloud Text-to-Speech (requires GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS)",
        "gtts": "Google Text-to-Speech (requires internet connection)",
        "azure_sdk": "Azure Speech SDK (requires AZURE_TTS_KEY and AZURE_TTS_REGION)",
        "pyttsx3": "Local pyttsx3 TTS (offline)",
        "coqui": "Coqui TTS (local ML models)",
        "local": "espeak-ng (system TTS)"
    }
    
    print("Available TTS Providers:")
    print("=" * 50)
    for provider, description in providers.items():
        status = "✅ Available" if _test_provider(provider) else "❌ Not available"
        print(f"  {provider:12}: {description}")
        print(f"  {' ' * 12}  Status: {status}")
        print()

def test_tts_providers():
    """Test all available TTS providers."""
    test_text = "Hello, this is a test of text to speech functionality."
    test_file = "test_output"
    
    providers = ["azure", "gcp", "gtts", "pyttsx3", "coqui", "local"]
    
    print("Testing TTS Providers:")
    print("=" * 50)
    
    for provider in providers:
        if not _test_provider(provider):
            print(f"❌ {provider.upper()}: Not available (skipping test)")
            continue
            
        try:
            print(f"🔄 Testing {provider.upper()}...")
            output_file = f"{test_file}_{provider}.mp3"
            result = generate_audio(test_text, output_file, provider=provider)
            
            # Verify file size
            if os.path.exists(result):
                size = os.path.getsize(result)
                print(f"✅ {provider.upper()}: Success ({size} bytes) - {result}")
            else:
                print(f"❌ {provider.upper()}: File not created")
                
        except Exception as e:
            print(f"❌ {provider.upper()}: Failed - {str(e)[:100]}")
        print()

def test_podcast_creation():
    """Test complete podcast creation with sample data."""
    test_insights = {
        "summary": "This is a comprehensive test analysis covering multiple aspects of document insights and key findings from our research.",
        "key_takeaways": [
            "First major insight about artificial intelligence and its applications in modern technology",
            "Second important finding regarding data processing and machine learning algorithms",
            "Third critical observation about user experience and interface design principles"
        ],
        "did_you_know": [
            "Fascinating fact about neural networks that most people don't realize",
            "Surprising statistical finding that emerged from our comprehensive data analysis"
        ],
        "contradictions": [
            "One area where different research sources present conflicting viewpoints on methodology"
        ],
        "examples": [
            {"text": "Concrete example demonstrating machine learning applications in real-world scenarios"},
            {"text": "Another practical example showing successful implementation of AI systems"}
        ],
        "additional_insights": [
            "Broader technological context that helps understand these developments",
            "Important implications for future research and practical applications"
        ]
    }
    
    try:
        print("Testing Podcast Creation:")
        print("=" * 50)
        print("🔄 Generating podcast script...")
        
        # Test both modes
        for mode in ["dialogue", "overview"]:
            print(f"\n🔄 Testing {mode} mode...")
            
            output_path = f"test_podcast_{mode}.mp3"
            result = create_podcast_from_insights(test_insights, filename=output_path, mode=mode)
            
            if os.path.exists(result):
                size = os.path.getsize(result)
                print(f"✅ {mode.title()} podcast created: {result} ({size} bytes)")
                print(f"   TTS backend used: {get_last_tts_provider()}")
            else:
                print(f"❌ {mode.title()} podcast creation failed: File not found")
        
        print("\n✅ Podcast creation test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Podcast creation test failed: {e}")
        logger.exception("Podcast creation test failed")
        return False

def check_dependencies():
    """Check all required and optional dependencies."""
    print("Checking Dependencies:")
    print("=" * 50)
    
    dependencies = {
        "pydub": HAS_PYDUB,
        "gTTS": HAS_GTTS,
        "pyttsx3": HAS_PYTTSX3,
        "azure-cognitiveservices-speech": HAS_AZURE_SDK,
        "google-cloud-texttospeech": HAS_GCP,
        "TTS (Coqui)": HAS_COQUI,
        "requests": True,  # Should always be available
    }
    
    for dep, available in dependencies.items():
        status = "✅ Installed" if available else "❌ Not installed"
        print(f"  {dep:30}: {status}")
    
    print("\nSystem Commands:")
    try:
        result = subprocess.run(['espeak-ng', '--version'], capture_output=True, timeout=5)
        espeak_status = "✅ Available" if result.returncode == 0 else "❌ Not available"
    except:
        espeak_status = "❌ Not available"
    
    print(f"  {'espeak-ng':30}: {espeak_status}")
    
    # Environment variables
    print("\nEnvironment Variables:")
    env_vars = [
        "TTS_PROVIDER", "AZURE_TTS_KEY", "AZURE_TTS_ENDPOINT", 
        "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
        "TTS_CLOUD_MAX_CHARS"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        status = "✅ Set" if value else "❌ Not set"
        if value and "KEY" in var:
            # Mask sensitive values
            display_value = value[:8] + "..." if len(value) > 8 else "***"
        else:
            display_value = value or "None"
        print(f"  {var:30}: {status} ({display_value})")

# ---------------------------
# MAIN ENTRY POINT
# ---------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "check":
            check_dependencies()
        elif command == "list":
            list_available_providers()
        elif command == "test":
            test_tts_providers()
        elif command == "podcast":
            test_podcast_creation()
        elif command == "all":
            check_dependencies()
            print("\n")
            list_available_providers()
            print("\n")
            test_tts_providers()
            print("\n")
            test_podcast_creation()
        else:
            print("Usage: python tts_unified.py [check|list|test|podcast|all]")
            print("  check   - Check dependencies and environment")
            print("  list    - List available TTS providers")
            print("  test    - Test all TTS providers")
            print("  podcast - Test podcast creation")
            print("  all     - Run all tests")
    else:
        # Default: run basic test
        print("Running basic TTS test...")
        try:
            result = generate_audio("Hello, this is a test of the unified TTS system.", "test_basic.mp3")
            print(f"✅ Basic test successful: {result}")
            print(f"   Provider used: {get_last_tts_provider()}")
        except Exception as e:
            print(f"❌ Basic test failed: {e}")
            print("\nRun with 'check' argument to diagnose issues:")
            print("  python tts_unified.py check")