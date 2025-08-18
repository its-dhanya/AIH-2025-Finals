# backend/api.py
import sys
import os
import uuid
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import logging
import json
from datetime import datetime
import time
import re
import aiohttp
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, Field
import gc
from threading import Lock
import random
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleRequest
from fastapi.responses import StreamingResponse, JSONResponse
# Add these imports after your existing imports
from langchain_google_genai import ChatGoogleGenerativeAI
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleRequest
import io
# Optional jsonschema import (best-effort)
try:
    from jsonschema import validate, ValidationError
except ImportError:
    ValidationError = Exception
    def validate(instance, schema):
        pass

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)
try:
    import podcast as podcast_module
    PODCAST_AVAILABLE = True
    logger.info("Podcast module loaded successfully")
except ImportError as e:
    PODCAST_AVAILABLE = False
    podcast_module = None
    logger.warning("Podcast module not available: %s", e)

# --- Path setup to find the 'libs' directory (project root assumed to be parent of backend) ---
backend_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(backend_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# --- Project-specific imports (adjust paths if necessary) ---
# Ensure these exist in your codebase
from libs.lib_a.main import process_pdf_for_api
from libs.lib_b.main import DocumentAnalysisPipeline

# -------------------- Gemini / Generative Language configuration --------------------
# NOTE: do NOT commit real API keys. Provide via environment variables.

# -------------------- Gemini / LangChain configuration --------------------
# -------------------- Gemini / Generative Language configuration --------------------
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
GEMINI_API_BASE = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com").rstrip("/")

PREFERRED_GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash-8b-001",
    "gemini-2.0-flash",
    "gemini-2.5-flash"
]

# Retry/backoff/timeouts
MAX_RETRIES_PER_MODEL = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
INITIAL_BACKOFF = float(os.getenv("GEMINI_INITIAL_BACKOFF", "1.0"))
BACKOFF_FACTOR = float(os.getenv("GEMINI_BACKOFF_FACTOR", "2.0"))
MAX_BACKOFF = float(os.getenv("GEMINI_MAX_BACKOFF", "10.0"))
MAX_OVERALL_TIMEOUT = int(os.getenv("GEMINI_OVERALL_TIMEOUT", "30"))
SEMAPHORE_LIMIT = int(os.getenv("GEMINI_CONCURRENCY", "1"))

# -------------------- JSON Schema for validation --------------------
INSIGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_takeaways": {"type": "array", "items": {"type": "string"}},
        "did_you_know": {"type": "array", "items": {"type": "string"}},
        "contradictions": {"type": "array", "items": {"type": "string"}},
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "document_name": {"type": ["string", "null"]},
                    "page_number": {"type": ["integer", "null"]}
                },
                "required": ["text"]
            }
        },
        "additional_insights": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["summary", "key_takeaways", "examples"]
}

# -------------------- Pydantic models --------------------
class FindConnectionsRequest(BaseModel):
    selected_text: str
    current_doc_id: Optional[str] = None
    source_document_name: Optional[str] = None
    connection_type: Optional[str] = "semantic"
    max_results: Optional[int] = 10
    include_preview: Optional[bool] = True
    min_connection_strength: Optional[str] = "weak"

class AnalyzeRequest(BaseModel):
    doc_id: str
    persona: str
    query: str

class AnalyzeCollectionRequest(BaseModel):
    doc_ids: List[str]
    collection_name: str
    persona: str
    query: str

class ConnectionData(BaseModel):
    text: str
    document_name: str
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    source: Optional[str] = None

class ExampleData(BaseModel):
    text: str
    document_name: Optional[str] = None
    page_number: Optional[int] = None

class InsightRequest(BaseModel):
    selected_text: str
    connections: List[ConnectionData]
    source_document: str
    context: Dict[str, Any] = {}
class PodcastRequest(BaseModel):
    insights: Optional[Dict[str, Any]] = None
    selected_text: Optional[str] = ""
    connections: Optional[List[ConnectionData]] = []
    source_document: Optional[str] = "Document Analysis"  # Fixed: provide default
    target_minutes: Optional[int] = Field(default=3, ge=1, le=10)
    speaker_a: Optional[str] = "Host"
    speaker_b: Optional[str] = "Guest"
    mode: Optional[str] = Field(default="dialogue", pattern="^(dialogue|overview)$")
# -------------------- Global instances --------------------
logger.info("Initializing Document Analysis Pipeline...")
pipeline = DocumentAnalysisPipeline()
try:
    pipeline.model_manager.load_models()
    logger.info("Pipeline initialized and models loaded.")
except Exception as e:
    logger.warning("Pipeline model loading at startup failed: %s", e)

db: Dict[str, Any] = {}
router = APIRouter()
reset_lock = Lock()

# -------------------- JSON extraction/repair utilities --------------------
def extract_first_json_object(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    start = t.find('{')
    if start == -1:
        # try array
        start = t.find('[')
        if start == -1:
            return ""
        bracket = 0
        end_idx = None
        for i in range(start, len(t)):
            if t[i] == '[':
                bracket += 1
            elif t[i] == ']':
                bracket -= 1
                if bracket == 0:
                    end_idx = i + 1
                    break
        return t[start:end_idx] if end_idx else t[start:]
    brace = 0
    end_idx = None
    for i in range(start, len(t)):
        if t[i] == '{':
            brace += 1
        elif t[i] == '}':
            brace -= 1
            if brace == 0:
                end_idx = i + 1
                break
    return t[start:end_idx] if end_idx else t[start:]

def repair_json_text(json_text: str) -> str:
    if not json_text:
        return ""
    s = json_text.strip()
    # remove code fences and pre tags
    s = re.sub(r"```(?:json)?", "", s, flags=re.I)
    s = re.sub(r"</?pre.*?>", "", s, flags=re.I)
    # remove single-line comments
    s = re.sub(r"//.*?$", "", s, flags=re.M)
    s = re.sub(r"#.*?$", "", s, flags=re.M)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    # convert Python-like booleans/None
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    # convert single quotes to double where reasonable
    def single_to_double(m):
        inner = m.group(1)
        if '"' in inner:
            return m.group(0)
        return '"' + inner.replace('"', '\\"') + '"'
    s = re.sub(r"'([^']*?)'", single_to_double, s)
    # remove trailing commas
    s = re.sub(r",\s*([\}\]])", r"\1", s)
    # try to quote unquoted keys (simple heuristic)
    s = re.sub(r'(?P<pre>[\{\s,])(?P<key>[A-Za-z_][A-Za-z0-9_\-]*?)\s*:', lambda m: f'{m.group("pre")}"{m.group("key")}":', s)
    # fix bracket/brace mismatches
    open_braces = s.count('{'); close_braces = s.count('}')
    if open_braces > close_braces:
        s += '}' * (open_braces - close_braces)
    open_brackets = s.count('['); close_brackets = s.count(']')
    if open_brackets > close_brackets:
        s += ']' * (open_brackets - close_brackets)
    return s.strip()

def extract_json_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*({[\s\S]*?})\s*```", text, re.I)
    if fence:
        return repair_json_text(fence.group(1))
    obj = extract_first_json_object(text)
    if obj:
        return repair_json_text(obj)
    return None

def parse_and_validate_json(json_text: str) -> Tuple[Optional[dict], str]:
    """Try parse + validate JSON (multiple strategies)."""
    if not json_text:
        return None, "empty"
    # direct parse
    try:
        data = json.loads(json_text)
        if isinstance(data, dict):
            data = ensure_required_fields(data)
            is_valid, err = validate_json_against_schema(data)
            if is_valid:
                return data, ""
            else:
                return data, f"validation_failed: {err}"
    except Exception:
        pass
    # extract & repair
    try:
        extracted = extract_json_from_text(json_text)
        if extracted:
            repaired = repair_json_text(extracted)
            data = json.loads(repaired)
            if isinstance(data, dict):
                data = ensure_required_fields(data)
                is_valid, err = validate_json_against_schema(data)
                if is_valid:
                    return data, ""
                else:
                    return data, f"validation_failed_after_repair: {err}"
    except Exception:
        pass
    # construct minimal object by regex
    try:
        summary_match = re.search(r'"summary"\s*:\s*"([^"]{5,})"', json_text, re.I)
        takeaways_match = re.search(r'"key_takeaways"\s*:\s*\[([^\]]*)\]', json_text, re.I | re.S)
        constructed = {
            "summary": summary_match.group(1) if summary_match else "Analysis completed",
            "key_takeaways": [],
            "did_you_know": [],
            "contradictions": [],
            "examples": [],
            "additional_insights": []
        }
        if takeaways_match:
            items = re.findall(r'"([^"]+)"', takeaways_match.group(1))
            if items:
                constructed["key_takeaways"] = items[:5]
        constructed = ensure_required_fields(constructed)
        return constructed, "constructed_partial"
    except Exception:
        return None, "failed_all"

def validate_json_against_schema(data: dict) -> Tuple[bool, str]:
    try:
        validate(instance=data, schema=INSIGHT_SCHEMA)
        return True, ""
    except ValidationError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def ensure_required_fields(data: dict) -> dict:
    """Ensure insight dict contains required keys and normalized types."""
    if not isinstance(data, dict):
        data = {}
    defaults = {
        "summary": "",
        "key_takeaways": [],
        "did_you_know": [],
        "contradictions": [],
        "examples": [],
        "additional_insights": []
    }
    for k, v in defaults.items():
        if k not in data or data[k] is None:
            data[k] = v
        else:
            if isinstance(v, list) and not isinstance(data[k], list):
                if isinstance(data[k], str):
                    data[k] = [data[k]]
                elif hasattr(data[k], "__iter__"):
                    data[k] = list(data[k])
                else:
                    data[k] = v
    # normalize examples
    cleaned_examples = []
    for ex in data.get("examples", []) or []:
        if isinstance(ex, dict) and ex.get("text"):
            try:
                page_num = int(ex.get("page_number")) if ex.get("page_number") is not None and str(ex.get("page_number")).isdigit() else None
            except Exception:
                page_num = None
            cleaned_examples.append({
                "text": str(ex.get("text")).strip(),
                "document_name": str(ex.get("document_name")).strip() if ex.get("document_name") else None,
                "page_number": page_num
            })
        elif isinstance(ex, str) and ex.strip():
            cleaned_examples.append({"text": ex.strip(), "document_name": None, "page_number": None})
    data["examples"] = cleaned_examples or []
    # fallbacks
    if not data["summary"]:
        data["summary"] = "Analysis completed"
    if not data["key_takeaways"]:
        data["key_takeaways"] = ["Key information extracted from content"]
    if not data["examples"]:
        data["examples"] = [{"text": "No explicit examples were provided by the model; fallback used.", "document_name": None, "page_number": None}]
    return data

def get_gemini_access_token():
    """Get access token for Gemini API using service account or API key."""
    if GEMINI_API_KEY:
        return GEMINI_API_KEY, "api_key"
    
    if GEMINI_CREDENTIALS_PATH and os.path.exists(GEMINI_CREDENTIALS_PATH):
        try:
            credentials = service_account.Credentials.from_service_account_file(
                GEMINI_CREDENTIALS_PATH,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            credentials.refresh(GoogleRequest())
            return credentials.token, "service_account"
        except Exception as e:
            logger.error(f"Failed to get service account token: {e}")
            raise ValueError(f"Failed to authenticate with service account: {e}")
    
    raise ValueError("Either GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS must be set")
def get_gemini_llm():
    """Initialize Gemini LLM with proper authentication."""
    api_key = GEMINI_API_KEY
    credentials_path = GEMINI_CREDENTIALS_PATH
    
    if not api_key and not credentials_path:
        raise ValueError("Either GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS must be set.")
    
    # Use API key if available, otherwise use service account credentials
    if api_key:
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.7
        )
    else:
        # For service account credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.7
        )

async def generate_json_with_langchain_gemini(prompt: str) -> Tuple[Optional[dict], dict]:
    """Generate JSON using LangChain Gemini interface."""
    try:
        llm = get_gemini_llm()
        
        # Format as LangChain messages
        messages = [
            {"role": "system", "content": "You are a JSON-only assistant. Return only valid JSON with no additional text."},
            {"role": "user", "content": prompt}
        ]
        
        # Get response
        response = llm.invoke(messages)
        response_text = response.content
        
        # Parse the JSON response
        parsed, parse_err = parse_and_validate_json(response_text)
        
        metadata = {
            "provider": "langchain_gemini",
            "model": GEMINI_MODEL,
            "timestamp": datetime.now().isoformat(),
            "parse_error": parse_err if parse_err else None
        }
        
        if parsed:
            return parsed, metadata
        else:
            logger.warning("LangChain Gemini returned unparseable JSON")
            return None, metadata
            
    except Exception as e:
        logger.exception("LangChain Gemini call failed: %s", e)
        return None, {
            "provider": "langchain_gemini", 
            "error": str(e)[:200],
            "timestamp": datetime.now().isoformat()
        }
async def generate_podcast_script_with_langchain(insights: dict, mode: str = "dialogue") -> List[dict]:
    """Generate podcast script using LangChain Gemini."""
    try:
        llm = get_gemini_llm()
        prompt = create_podcast_script_prompt(insights, mode)
        
        messages = [
            {"role": "system", "content": "You are a podcast script generator. Return only valid JSON arrays."},
            {"role": "user", "content": prompt}
        ]
        
        response = llm.invoke(messages)
        response_text = response.content
        
        # Parse JSON array
        try:
            script_data = json.loads(response_text.strip())
            if isinstance(script_data, list):
                return script_data
            else:
                logger.warning("Script response not a list, wrapping")
                return [{"speaker": "Host", "text": str(script_data)}]
        except json.JSONDecodeError:
            logger.warning("Failed to parse script JSON, using fallback")
            return [
                {"speaker": "Host", "text": f"Welcome to today's analysis of {insights.get('summary', 'our content')[:100]}"},
                {"speaker": "Guest", "text": "The key takeaway is: " + str(insights.get('key_takeaways', ['interesting insights'])[0])}
            ]
            
    except Exception as e:
        logger.exception("LangChain script generation failed: %s", e)
        # Return basic fallback
        return [
            {"speaker": "Host", "text": f"Today we're discussing {insights.get('summary', 'important content')[:150]}"},
            {"speaker": "Guest", "text": "The main points include: " + ", ".join(insights.get('key_takeaways', ['key insights'])[:2])}
        ]
def create_podcast_script_prompt(insights: dict, mode: str = "dialogue", max_segments: int = 12) -> str:
    """
    Build a prompt that asks Gemini to produce a JSON array of segments for a podcast.
    Each segment: {"speaker":"Host"|"Guest", "text":"..."}.
    """
    summary = insights.get("summary", "")[:1200].replace('"', "'")
    takeaways = insights.get("key_takeaways", [])[:6]
    did_you_know = insights.get("did_you_know", [])[:4]
    contradictions = insights.get("contradictions", [])[:4]
    examples = insights.get("examples", [])[:4]
    additional = insights.get("additional_insights", [])[:3]

    # format context section
    ctx_lines = []
    if summary:
        ctx_lines.append(f"SUMMARY: {summary}")
    if takeaways:
        ctx_lines.append("TAKEAWAYS: " + " | ".join(takeaways))
    if did_you_know:
        ctx_lines.append("DID_YOU_KNOW: " + " | ".join(did_you_know))
    if contradictions:
        ctx_lines.append("CONTRADICTIONS: " + " | ".join(contradictions))
    if examples:
        exs = []
        for e in examples[:3]:
            if isinstance(e, dict):
                exs.append(e.get("text","")[:160].replace('"', "'"))
            else:
                exs.append(str(e)[:160].replace('"', "'"))
        ctx_lines.append("EXAMPLES: " + " | ".join(exs))
    if additional:
        ctx_lines.append("ADDITIONAL: " + " | ".join(additional))

    prompt = (
        "You are a podcast script writer. Produce exactly one JSON array (no explanation, no markdown, no extra text) "
        "containing between 3 and {max_segments} ordered segment objects for a short 2-5 minute podcast. "
        "Each segment must be an object with keys: speaker (either 'Host' or 'Guest' or 'Narrator'), and text (the spoken content). "
        "Keep each text snippet short (10-40 seconds spoken length). "
        "Use conversational tone for 'dialogue' mode or succinct monologue tone for 'overview' mode.\n\n"
        "MODE: {mode}\n\n"
        "CONTEXT:\n{context}\n\n"
        "Return ONLY the JSON array. Example:\n"
        '[{{"speaker":"Host","text":"..."}},{{"speaker":"Guest","text":"..."}},...]\n'
    ).format(max_segments=max_segments, mode=mode, context="\n".join(ctx_lines))
    return prompt

# -------------------- Model ranking helper --------------------
class ModelRanker:
    def __init__(self):
        self.success_rates: Dict[str, int] = {}
        self.attempt_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, Dict[str, int]] = {}

    def record_attempt(self, model: str, success: bool, error_type: Optional[str] = None):
        self.attempt_counts.setdefault(model, 0)
        self.success_rates.setdefault(model, 0)
        self.error_counts.setdefault(model, {})
        self.attempt_counts[model] += 1
        if success:
            self.success_rates[model] += 1
        elif error_type:
            self.error_counts[model].setdefault(error_type, 0)
            self.error_counts[model][error_type] += 1

    def get_success_rate(self, model: str) -> float:
        attempts = self.attempt_counts.get(model, 0)
        if attempts == 0:
            return 0.5
        return self.success_rates.get(model, 0) / attempts

    def get_model_stats(self, model: str) -> Dict[str, Any]:
        return {
            "attempts": self.attempt_counts.get(model, 0),
            "successes": self.success_rates.get(model, 0),
            "success_rate": self.get_success_rate(model),
            "errors": self.error_counts.get(model, {})
        }

model_ranker = ModelRanker()

# -------------------- Gemini integration (multi-model) --------------------
semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)

async def gemini_list_models(session: aiohttp.ClientSession) -> List[str]:
    try:
        url = f"{GEMINI_API_BASE}/v1/models?key={GEMINI_API_KEY}"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                txt = await resp.text()
                logger.warning("List models failed %s: %s", resp.status, txt[:400])
                return PREFERRED_GEMINI_MODELS
            data = await resp.json()
            models = []
            for m in data.get("models", []):
                name = m.get("name") or m.get("id") or m.get("model")
                if name:
                    models.append(name.split("/")[-1])
            return models or PREFERRED_GEMINI_MODELS
    except Exception as e:
        logger.warning("List models exception: %s", e)
        return PREFERRED_GEMINI_MODELS

def pick_candidate_models(available: List[str]) -> List[str]:
    chosen = []
    for p in PREFERRED_GEMINI_MODELS:
        for a in available:
            if p in a or a in p or a == p:
                if a not in chosen:
                    chosen.append(a)
    for a in available:
        if a not in chosen:
            chosen.append(a)
        if len(chosen) >= 6:
            break
    return chosen

async def generate_json_with_gemini_multi_model(
    prompt: str,
    session: aiohttp.ClientSession,
    overall_timeout: int = MAX_OVERALL_TIMEOUT
) -> Tuple[Optional[dict], dict]:
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not configured")
        return None, {"provider": "gemini", "error": "no_api_key"}

    start_time = time.time()
    all_tried = []
    try:
        available = await gemini_list_models(session)
        candidates = pick_candidate_models(available)
    except Exception as e:
        logger.warning("Could not list models, using preference list: %s", e)
        candidates = PREFERRED_GEMINI_MODELS[:3]

    if not candidates:
        candidates = PREFERRED_GEMINI_MODELS[:3]

    # Try a handful of models
    for model_idx, model in enumerate(candidates[:5]):
        if time.time() - start_time > overall_timeout:
            break
        logger.info("Attempting model %d/%d: %s", model_idx + 1, min(len(candidates), 5), model)
        backoff = INITIAL_BACKOFF
        max_attempts = MAX_RETRIES_PER_MODEL if model_idx == 0 else 2

        for attempt in range(1, max_attempts + 1):
            if time.time() - start_time > overall_timeout:
                break
            try:
                access_token, auth_type = get_gemini_access_token()
            except Exception as e:
                return None, {"provider": "gemini", "error": "authentication_failed", "details": str(e)}

            # Build URL + headers depending on auth type
            if auth_type == "api_key":
                model_url = f"{GEMINI_API_BASE}/v1/models/{model}:generateContent?key={access_token}"
                headers = {"Content-Type": "application/json", "x-goog-api-key": access_token}
            else:  # service_account / bearer
                model_url = f"{GEMINI_API_BASE}/v1/models/{model}:generateContent"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}

            payload = {"contents": [{"parts": [{"text": prompt}], "role": "user"}]}
            timeout = aiohttp.ClientTimeout(total=18)

            try:
                async with semaphore:
                    async with session.post(model_url, json=payload, headers=headers, timeout=timeout) as resp:
                        status = resp.status
                        text_body = await resp.text()
                        meta = {
                            "model": model,
                            "status": status,
                            "attempt": attempt,
                            "timestamp": datetime.now().isoformat()
                        }
                        all_tried.append(meta)

                        if status == 429:
                            ra = resp.headers.get("Retry-After")
                            try:
                                wait = int(ra) if ra and ra.isdigit() else min(MAX_BACKOFF, backoff * BACKOFF_FACTOR)
                            except Exception:
                                wait = min(MAX_BACKOFF, backoff * BACKOFF_FACTOR)
                            model_ranker.record_attempt(model, False, "rate_limit")
                            logger.warning("Rate limited on %s, waiting %ds (attempt %d)", model, wait, attempt)
                            if attempt < max_attempts:
                                await asyncio.sleep(wait + random.uniform(0, 1))
                                backoff = min(MAX_BACKOFF, backoff * BACKOFF_FACTOR)
                                continue
                            else:
                                break

                        if status >= 500:
                            model_ranker.record_attempt(model, False, f"server_error_{status}")
                            logger.warning(
                                "Server error %s on %s (attempt %d): %s", status, model, attempt, text_body[:200]
                            )
                            if attempt < max_attempts:
                                await asyncio.sleep(min(MAX_BACKOFF, backoff * BACKOFF_FACTOR))
                                backoff = min(MAX_BACKOFF, backoff * BACKOFF_FACTOR)
                                continue
                            else:
                                break

                        if 400 <= status < 500:
                            model_ranker.record_attempt(model, False, f"client_error_{status}")
                            logger.warning("Client error %s on %s: %s", status, model, text_body[:300])
                            break

                        # Parse candidate text
                        raw_texts = []
                        try:
                            data = json.loads(text_body) if text_body.strip() else {}
                        except Exception:
                            data = {}

                        if isinstance(data, dict):
                            for cand in data.get("candidates", []) or []:
                                content = cand.get("content") or {}
                                parts = content.get("parts") if isinstance(content, dict) else None
                                if parts and isinstance(parts, list):
                                    for p in parts:
                                        if isinstance(p, dict) and "text" in p:
                                            raw_texts.append(p["text"])
                                        elif isinstance(p, str):
                                            raw_texts.append(p)
                        if not raw_texts:
                            raw_texts = [text_body]

                        # Try parsing each candidate
                        for raw in raw_texts:
                            candidate = extract_json_from_text(raw) or extract_first_json_object(raw)
                            if not candidate:
                                continue
                            repaired = repair_json_text(candidate)
                            parsed, parse_err = parse_and_validate_json(repaired)
                            if parsed:
                                model_ranker.record_attempt(model, True)
                                return parsed, {
                                    "provider": "gemini",
                                    "model": model,
                                    "attempts": all_tried,
                                    "parse_error": parse_err
                                }

                        # If we reach here: got 200 but couldn't extract JSON
                        logger.warning("Got 200 from %s but couldn't extract valid JSON", model)
                        model_ranker.record_attempt(model, False, "no_valid_json")
                        break

            except asyncio.TimeoutError:
                model_ranker.record_attempt(model, False, "timeout")
                logger.warning("Timeout for %s (attempt %d)", model, attempt)
                if attempt < max_attempts:
                    await asyncio.sleep(min(MAX_BACKOFF, backoff))
                    backoff = min(MAX_BACKOFF, backoff * BACKOFF_FACTOR)
                    continue
                else:
                    break
            except Exception as e:
                model_ranker.record_attempt(model, False, "exception")
                logger.exception(
                    "Exception calling Gemini %s attempt %d: %s", model, attempt, str(e)[:200]
                )
                if attempt < max_attempts:
                    await asyncio.sleep(min(MAX_BACKOFF, backoff))
                    backoff = min(MAX_BACKOFF, backoff * BACKOFF_FACTOR)
                    continue
                else:
                    break

    logger.error("All Gemini attempts failed; tried metadata: %s", all_tried)
    return None, {"provider": "gemini", "error": "all_models_failed", "tried": all_tried}


async def generate_json_with_gemini(prompt: str, session: aiohttp.ClientSession, overall_timeout: int = MAX_OVERALL_TIMEOUT) -> Tuple[Optional[dict], dict]:
    return await generate_json_with_gemini_multi_model(prompt, session, overall_timeout)

# -------------------- Prompt builder (uses multiple connections) --------------------
def create_insights_prompt(request: InsightRequest, max_connections: int = 8) -> str:
    selected_preview = (request.selected_text or "").strip().replace("\n", " ")[:1000]
    conn_texts = []
    for i, c in enumerate((request.connections or [])[:max_connections], start=1):
        raw = (getattr(c, "text", "") or "").strip().replace("\n", " ")
        if not raw:
            continue
        first_sent = re.split(r'[.!?]\s+', raw)[0][:220]
        doc = getattr(c, "document_name", "") or ""
        page = getattr(c, "page_number", None)
        conn_texts.append(f"CONN_{i}: {first_sent} (doc: {doc}, page: {page})")
    connections_block = "\n".join(conn_texts) if conn_texts else "(none)"

    prompt = (
        "You are a JSON-only document analysis assistant. RETURN EXACTLY ONE valid JSON object with the keys:\n"
        "summary (string), key_takeaways (array of short strings), did_you_know (array of strings), "
        "contradictions (array of strings), examples (array of objects {text, document_name, page_number}), additional_insights (array of strings).\n\n"
        f"SELECTED_PREVIEW: \"{selected_preview}\"\n\n"
        "RELATED_CONNECTIONS:\n"
        f"{connections_block}\n\n"
        "INSTRUCTIONS:\n"
        "- Provide 2-6 key_takeaways (short phrases)\n"
        "- Provide 0-4 did_you_know facts (concise)\n"
        "- Provide 0-4 contradictions or counterpoints\n"
        "- Provide 1-6 examples drawn from the RELATED_CONNECTIONS (each as an object with text, document_name, page_number)\n"
        "- additional_insights: optional cross-document or broader implications\n"
        "Return only the JSON object and nothing else."
    )
    return prompt

# -------------------- Deterministic fallback (multi-connection) --------------------
def generate_deterministic_fallback(request: InsightRequest) -> Tuple[dict, dict]:
    selected = (request.selected_text or "").strip()
    connections = request.connections or []

    summary_parts = []
    if selected:
        sents = [s.strip() for s in re.split(r'[.!?]+', selected) if s.strip()]
        if sents:
            summary_parts.append(sents[0][:300])
    for c in connections[:3]:
        t = getattr(c, "text", "") or ""
        s = re.split(r'[.!?]+', t)[0].strip()
        if s:
            summary_parts.append(s[:200])
    summary = " ".join(summary_parts) or "Analysis completed."

    takeaways = []
    entities = []
    for c in connections[:8]:
        txt = getattr(c, "text", "") or ""
        caps = re.findall(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b', txt)
        if caps:
            for ent in caps[:2]:
                if ent not in entities:
                    entities.append(ent)
        lead = re.split(r'[.!?]+', txt)[0].strip()
        if lead and len(lead) > 20 and lead not in takeaways:
            takeaways.append(lead if len(lead) <= 120 else lead[:117] + "...")
    if entities:
        takeaways.insert(0, f"Entities: {', '.join(entities[:3])}")
    if not takeaways:
        takeaways = ["Content reviewed; no clear takeaways automatically extracted."]

    did_you_know = []
    nums = re.findall(r'\b\d+(?:\.\d+)?%?\b', selected)
    years = re.findall(r'\b(?:19|20)\d{2}\b', selected)
    if nums:
        did_you_know.append(f"Contains {len(nums)} numeric references.")
    if years:
        did_you_know.append(f"Mentions years: {', '.join(sorted(set(years))[:3])}")
    if not did_you_know and entities:
        did_you_know.append(f"Mentions topics/entities: {', '.join(entities[:3])}")

    contradictions = []
    combined = (selected + " " + " ".join([getattr(c, "text", "") for c in connections[:6]])).lower()
    pairs = [('increase','decrease'), ('rise','fall'), ('positive','negative'), ('support','oppose'), ('benefit','harm')]
    for a,b in pairs:
        if a in combined and b in combined:
            contradictions.append(f"References both '{a}' and '{b}' across sources")
    if any(k in combined for k in ['however','but','although','despite']):
        contradictions.append("Contains qualifying language implying nuanced viewpoints")

    examples = []
    seen = set()
    for c in connections[:8]:
        txt = getattr(c, "text", "") or ""
        lead = re.split(r'[.!?]+', txt)[0].strip()
        if lead and lead.lower() not in seen and len(lead) > 15:
            seen.add(lead.lower())
            examples.append({
                "text": lead[:400],
                "document_name": getattr(c, "document_name", None),
                "page_number": getattr(c, "page_number", None)
            })
        if len(examples) >= 6:
            break
    if not examples and selected:
        examples.append({"text": selected[:400], "document_name": request.source_document, "page_number": None})

    additional_insights = []
    unique_docs = set([getattr(c, "document_name", "") for c in connections if getattr(c, "document_name", "")])
    if unique_docs:
        additional_insights.append(f"Cross-references found across {len(unique_docs)} documents")
    if len(connections) > 5:
        additional_insights.append("High connectivity indicates centrality of topic in collection")

    data = {
        "summary": summary,
        "key_takeaways": takeaways[:6],
        "did_you_know": did_you_know[:4],
        "contradictions": contradictions[:4],
        "examples": examples[:6],
        "additional_insights": additional_insights[:6]
    }
    data = ensure_required_fields(data)
    meta = {
        "provider": "deterministic_fallback",
        "timestamp": datetime.now().isoformat(),
        "connections_count": len(connections)
    }
    return data, meta

# -------------------- Safe conversion for frontend --------------------
def convert_insights_to_dict_safe(data: dict, metadata: dict) -> dict:
    def safe_str(o, max_len=500):
        if o is None:
            return ""
        if isinstance(o, str):
            return o.strip()[:max_len]
        if isinstance(o, (dict, list)):
            try:
                return json.dumps(o, ensure_ascii=False)[:max_len]
            except:
                return str(o)[:max_len]
        return str(o)[:max_len]

    raw_examples = data.get("examples", []) or []
    example_objects = []
    example_texts = []
    for ex in raw_examples[:8]:
        try:
            if isinstance(ex, dict):
                text = safe_str(ex.get("text", ""), 500)
                doc = safe_str(ex.get("document_name", ""), 120) or None
                page = ex.get("page_number")
                try:
                    page = int(page) if page is not None else None
                except:
                    page = None
                if text and len(text) > 3 and text.lower() not in ["null", "none", "[object object]"]:
                    example_objects.append({"text": text, "document_name": doc, "page_number": page})
                    example_texts.append(text)
            else:
                txt = safe_str(ex, 500)
                if txt:
                    example_objects.append({"text": txt, "document_name": None, "page_number": None})
                    example_texts.append(txt)
        except Exception as e:
            logger.warning("Error processing example: %s", e)
            continue

    def dedupe_and_clean(lst, max_items=8):
        cleaned = []
        seen = set()
        for it in (lst or [])[:max_items]:
            s = safe_str(it, 400).strip()
            if s and s.lower() not in ["null", "none", "[object object]"] and s not in seen:
                cleaned.append(s); seen.add(s)
        return cleaned

    result = {
        "summary": safe_str(data.get("summary", "Analysis completed"), 1200),
        "key_takeaways": dedupe_and_clean(data.get("key_takeaways", []), 8),
        "did_you_know": dedupe_and_clean(data.get("did_you_know", []), 6),
        "contradictions": dedupe_and_clean(data.get("contradictions", []), 6),
        "examples": example_texts,
        "example_objects": example_objects,
        "example_texts": example_texts,
        "additional_insights": dedupe_and_clean(data.get("additional_insights", []), 6),
        "metadata": metadata or {}
    }

    # ensure minimums
    if not result["key_takeaways"]:
        result["key_takeaways"] = ["Key information extracted"]
    if not result["examples"]:
        result["examples"] = ["No explicit examples; fallback used"]
        result["example_texts"] = result["examples"]
        result["example_objects"] = [{"text": result["examples"][0], "document_name": None, "page_number": None}]

    # validate final structure and attach warning to metadata if invalid
    is_valid, err = validate_json_against_schema({
        "summary": result["summary"],
        "key_takeaways": result["key_takeaways"],
        "did_you_know": result["did_you_know"],
        "contradictions": result["contradictions"],
        "examples": result["example_objects"],
        "additional_insights": result["additional_insights"]
    })
    if not is_valid:
        logger.warning("Final insights failed schema validation: %s", err)
        result["metadata"]["validation_warning"] = err

    logger.info("Final result - Examples: %d, Example_texts: %d, Summary length: %d",
                len(result["examples"]), len(result["example_texts"]), len(result["summary"]))
    return result

# -------------------- Main orchestration for insights --------------------
def get_gemini_llm():
    """Initialize Gemini LLM with proper authentication."""
    api_key = GEMINI_API_KEY
    credentials_path = GEMINI_CREDENTIALS_PATH
    
    if not api_key and not credentials_path:
        raise ValueError("Either GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS must be set.")
    
    # Use API key if available, otherwise use service account credentials
    if api_key:
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.7
        )
    else:
        # For service account credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.7
        )

async def generate_json_with_langchain_gemini(prompt: str) -> Tuple[Optional[dict], dict]:
    """Generate JSON using LangChain Gemini interface."""
    try:
        llm = get_gemini_llm()
        
        # Format as LangChain messages
        messages = [
            {"role": "system", "content": "You are a JSON-only assistant. Return only valid JSON with no additional text."},
            {"role": "user", "content": prompt}
        ]
        
        # Get response
        response = llm.invoke(messages)
        response_text = response.content
        
        # Parse the JSON response
        parsed, parse_err = parse_and_validate_json(response_text)
        
        metadata = {
            "provider": "langchain_gemini",
            "model": GEMINI_MODEL,
            "timestamp": datetime.now().isoformat(),
            "parse_error": parse_err if parse_err else None
        }
        
        if parsed:
            return parsed, metadata
        else:
            logger.warning("LangChain Gemini returned unparseable JSON")
            return None, metadata
            
    except Exception as e:
        logger.exception("LangChain Gemini call failed: %s", e)
        return None, {
            "provider": "langchain_gemini", 
            "error": str(e)[:200],
            "timestamp": datetime.now().isoformat()
        }
# -------------------- API endpoints --------------------
@router.post("/generate-insights")
async def generate_insights(request: InsightRequest):
    logger.info("Generating insights for selected_text(len=%d) and %d connections", len(request.selected_text or ""), len(request.connections or []))
    try:
        if not request.connections:
            raise HTTPException(status_code=400, detail="No connections provided")
        if not request.selected_text or len(request.selected_text.strip()) < 3:
            logger.warning("Selected text short/empty - continuing using connections")
        result = await process_insights_with_validation(request)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("generate_insights error: %s", e)
        try:
            data, meta = generate_deterministic_fallback(request)
            return convert_insights_to_dict_safe(data, meta)
        except Exception as e2:
            logger.exception("Fallback failed: %s", e2)
            return {
                "summary": "Error",
                "key_takeaways": ["Analysis failed"],
                "did_you_know": [],
                "contradictions": [],
                "examples": [],
                "example_texts": [],
                "additional_insights": [],
                "metadata": {"error": str(e)[:200]}
            }

# --- other endpoints (reset, upload, find-connections, analyze, analyze-collection, etc.) ---
@router.post("/reset")
async def reset_session():
    global db, pipeline
    with reset_lock:
        db.clear()
        try:
            if pipeline:
                if hasattr(pipeline, "clear_index"):
                    pipeline.clear_index()
                elif hasattr(pipeline, "reset_index"):
                    pipeline.reset_index()
                else:
                    if hasattr(pipeline, "vector_store"):
                        try:
                            if hasattr(pipeline.vector_store, "reset"):
                                pipeline.vector_store.reset()
                            elif hasattr(pipeline.vector_store, "clear"):
                                pipeline.vector_store.clear()
                            else:
                                pipeline.vector_store = None
                        except Exception as e:
                            logger.warning("Warning clearing vector_store: %s", e)
                    if hasattr(pipeline, "index"):
                        try:
                            pipeline.index = None
                        except Exception as e:
                            logger.warning("Warning clearing index: %s", e)
                    if hasattr(pipeline, "embeddings"):
                        try:
                            pipeline.embeddings = None
                        except Exception as e:
                            logger.warning("Warning clearing embeddings: %s", e)
                    if hasattr(pipeline, "cache"):
                        try:
                            pipeline.cache.clear()
                        except Exception:
                            pipeline.cache = None
        except Exception as e:
            logger.warning("Warning: error while trying to clear pipeline state: %s", e)
        gc.collect()
    return {"message": "Session reset (db cleared; pipeline state cleared where possible)."}

@router.post("/upload-and-index")
async def upload_documents(files: List[UploadFile] = File(...)):
    responses = []
    for file in files:
        if file.content_type != "application/pdf":
            continue
        doc_id = str(uuid.uuid4())
        file_content = await file.read()
        outline_data = process_pdf_for_api(file_content)
        db[doc_id] = {
            "metadata": outline_data,
            "content": file_content,
            "filename": file.filename,
            "upload_timestamp": datetime.now().isoformat(),
            "file_size": len(file_content)
        }
        response_data = {"doc_id": doc_id, **outline_data}
        responses.append(response_data)
    if not responses:
        raise HTTPException(status_code=400, detail="No valid PDF files were processed.")
    return responses

@router.get("/outline/{doc_id}")
async def get_document_outline(doc_id: str):
    if doc_id not in db or "metadata" not in db[doc_id]:
        raise HTTPException(status_code=404, detail="Document not found or not processed correctly.")
    return db[doc_id]["metadata"]

@router.post("/find-connections")
async def find_connections(request: FindConnectionsRequest):
    if not db:
        raise HTTPException(status_code=400, detail="No documents have been uploaded to search.")
    temp_dir = tempfile.mkdtemp()
    try:
        temp_file_paths = []
        doc_ids_to_process = [doc_id for doc_id in db.keys() if doc_id != request.current_doc_id]
        if not doc_ids_to_process:
            return {"connections": []}
        for doc_id in doc_ids_to_process:
            doc_data = db[doc_id]
            filename_from_db = doc_data["metadata"].get("title", doc_id)
            filename = f"{filename_from_db}.pdf" if not filename_from_db.lower().endswith(".pdf") else filename_from_db
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            temp_file_path = Path(temp_dir) / filename
            temp_file_path.write_bytes(doc_data["content"])
            temp_file_paths.append(str(temp_file_path))
        all_chunks = []
        for path in temp_file_paths:
            all_chunks.extend(pipeline.processor.extract_chunks_from_pdf(path))
        connection_results = pipeline.find_connections_enhanced(
            selected_text=request.selected_text,
            chunks=all_chunks,
            top_k=request.max_results
        )
        return {"connections": connection_results}
    except Exception as e:
        logger.exception("An unexpected error occurred during connection finding: %s", e)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning("Failed to cleanup temp dir: %s", e)

@router.post("/analyze")
async def analyze_document(request: AnalyzeRequest, return_insights: bool = Query(False, description="Return structured insights alongside analysis")):
    logger.info("Analyzing document %s with query: %s", request.doc_id, request.query[:120])
    if request.doc_id not in db:
        raise HTTPException(status_code=404, detail="Document not found.")
    temp_dir = tempfile.mkdtemp()
    try:
        doc_data = db[request.doc_id]
        filename_from_db = doc_data["metadata"].get("title", request.doc_id)
        filename = f"{filename_from_db}.pdf" if not filename_from_db.lower().endswith(".pdf") else filename_from_db
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        temp_file_path = Path(temp_dir) / filename
        temp_file_path.write_bytes(doc_data["content"])
        analysis_result = pipeline.process_api_request(
            task=request.query,
            pdf_file_paths=[str(temp_file_path)],
            collection_name=filename_from_db
        )

        if return_insights:
            selected_preview = ""
            if isinstance(analysis_result, dict):
                selected_preview = str(analysis_result.get("summary") or analysis_result.get("text") or request.query)[:1000]
            else:
                selected_preview = request.query[:1000]

            chunks = pipeline.processor.extract_chunks_from_pdf(str(temp_file_path))
            connections = []
            for ch in (chunks or [])[:8]:
                text = ch.get("text") if isinstance(ch, dict) else str(ch)
                doc_name = filename_from_db
                page = None
                if isinstance(ch, dict):
                    page = ch.get("page_number") or ch.get("page")
                connections.append({"text": text, "document_name": doc_name, "page_number": page})

            insight_req = InsightRequest(
                selected_text=selected_preview,
                connections=[ConnectionData(**c) for c in connections],
                source_document=filename_from_db,
                context={"origin": "analyze_endpoint", "doc_id": request.doc_id}
            )
            insights = await process_insights_with_validation(insight_req)
            if isinstance(analysis_result, dict):
                analysis_result["insights"] = insights
            else:
                analysis_result = {"result": analysis_result, "insights": insights}

        return analysis_result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during analysis: %s", e)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning("Failed to cleanup temp directory: %s", e)

@router.post("/analyze-collection")
async def analyze_collection(request: AnalyzeCollectionRequest, return_insights: bool = Query(False, description="Return structured insights computed across the collection")):
    logger.info("Analyzing collection of %d documents", len(request.doc_ids))
    missing = [doc_id for doc_id in request.doc_ids if doc_id not in db]
    if missing:
        raise HTTPException(status_code=404, detail=f"Documents not found: {', '.join(missing[:3])}")
    temp_dir = tempfile.mkdtemp()
    try:
        temp_file_paths = []
        for doc_id in request.doc_ids:
            doc_data = db[doc_id]
            filename_from_db = doc_data["metadata"].get("title", doc_id)
            filename = f"{filename_from_db}.pdf" if not filename_from_db.lower().endswith(".pdf") else filename_from_db
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            temp_file_path = Path(temp_dir) / filename
            temp_file_path.write_bytes(doc_data["content"])
            temp_file_paths.append(str(temp_file_path))
        analysis_result = pipeline.process_api_request(
            task=request.query,
            pdf_file_paths=temp_file_paths,
            collection_name=request.collection_name
        )

        if return_insights:
            selected_preview = ""
            if isinstance(analysis_result, dict):
                selected_preview = str(analysis_result.get("summary") or request.query)[:1000]
            else:
                selected_preview = request.query[:1000]

            all_chunks = []
            for p in temp_file_paths:
                all_chunks.extend(pipeline.processor.extract_chunks_from_pdf(p))
            connections = []
            for ch in (all_chunks or [])[:16]:
                text = ch.get("text") if isinstance(ch, dict) else str(ch)
                doc_name = ch.get("document_name") if isinstance(ch, dict) else request.collection_name
                page = ch.get("page_number") if isinstance(ch, dict) else None
                connections.append({"text": text, "document_name": doc_name, "page_number": page})

            insight_req = InsightRequest(
                selected_text=selected_preview,
                connections=[ConnectionData(**c) for c in connections],
                source_document=request.collection_name,
                context={"origin": "analyze_collection", "doc_ids": request.doc_ids}
            )
            insights = await process_insights_with_validation(insight_req)
            if isinstance(analysis_result, dict):
                analysis_result["insights"] = insights
            else:
                analysis_result = {"result": analysis_result, "insights": insights}

        return analysis_result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Collection analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Collection analysis failed: {str(e)}")
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning("Failed to cleanup temp directory: %s", e)

@router.get("/available-models")
async def list_available_models():
    logger.info("Listing available Gemini models")
    try:
        connector = aiohttp.TCPConnector(limit=1)
        async with aiohttp.ClientSession(connector=connector) as session:
            models = await gemini_list_models(session)
        rankings = {m: model_ranker.get_success_rate(m) for m in models[:20]}
        return {
            "models": models,
            "success_rates": rankings,
            "preferred_models": PREFERRED_GEMINI_MODELS,
            "api_configured": bool(GEMINI_API_KEY),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception("Error listing models: %s", e)
        return {"error": str(e), "preferred_models": PREFERRED_GEMINI_MODELS, "api_configured": bool(GEMINI_API_KEY)}

@router.post("/preload-model")
async def preload_model(model_name: str = Query("", description="Model name to preload")):
    if not model_name:
        raise HTTPException(status_code=400, detail="Provide model_name")
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    connector = aiohttp.TCPConnector(limit=1)
    async with aiohttp.ClientSession(connector=connector) as session:
        url = f"{GEMINI_API_BASE}/v1/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        payload = {"contents": [{"parts": [{"text": "Hello - test"}], "role": "user"}]}
        headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
        start = time.time()
        try:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                body = await resp.text()
                result = {"model": model_name, "status": resp.status, "elapsed": round(time.time() - start, 2)}
                if resp.status == 200:
                    model_ranker.record_attempt(model_name, True)
                    result["message"] = "Preload success"
                    result["response_preview"] = body[:400]
                else:
                    model_ranker.record_attempt(model_name, False, f"status_{resp.status}")
                    result["error"] = body[:800]
                return result
        except asyncio.TimeoutError:
            model_ranker.record_attempt(model_name, False, "timeout")
            return {"error": "timeout"}
        except Exception as e:
            model_ranker.record_attempt(model_name, False, "exception")
            logger.exception("Preload model exception: %s", e)
            return {"error": str(e)}

@router.get("/gemini-diagnostics")
async def gemini_diagnostics():
    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "api_key_configured": bool(GEMINI_API_KEY),
        "api_base": GEMINI_API_BASE,
        "preferred_models": PREFERRED_GEMINI_MODELS,
        "max_overall_timeout": MAX_OVERALL_TIMEOUT
    }
    if not GEMINI_API_KEY:
        diagnostics["error"] = "GEMINI_API_KEY not configured"
        return diagnostics
    connector = aiohttp.TCPConnector(limit=1)
    async with aiohttp.ClientSession(connector=connector) as session:
        try:
            start = time.time()
            models = await gemini_list_models(session)
            diagnostics["models_found"] = models[:20]
            diagnostics["list_elapsed"] = round(time.time() - start, 2)
        except Exception as e:
            diagnostics["list_error"] = str(e)
        try:
            test_model = PREFERRED_GEMINI_MODELS[0]
            url = f"{GEMINI_API_BASE}/v1/models/{test_model}:generateContent?key={GEMINI_API_KEY}"
            payload = {"contents": [{"parts": [{"text": "Return exactly: {\"test\":\"ok\"}"}], "role": "user"}]}
            headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
            start = time.time()
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=12)) as resp:
                body = await resp.text()
                diagnostics["test_status"] = resp.status
                diagnostics["test_body_preview"] = body[:800]
                diagnostics["test_elapsed"] = round(time.time() - start, 2)
        except Exception as e:
            diagnostics["test_error"] = str(e)
    return diagnostics


@router.get("/health")
async def health_check():
    gemini_status = "not_configured"
    auth_method = "none"
    
    try:
        if GEMINI_API_KEY or GEMINI_CREDENTIALS_PATH:
            _, auth_method = get_gemini_access_token()
            gemini_status = "configured"
    except Exception as e:
        gemini_status = f"error: {str(e)[:100]}"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "documents_loaded": len(db),
        "gemini_status": gemini_status,
        "gemini_model": GEMINI_MODEL,
        "auth_method": auth_method
    }
@router.post("/generate-podcast")
async def generate_podcast_endpoint(request: PodcastRequest):
    """
    Generate podcast audio from insights or text with connections.
    FIXED: Better error handling, validation, and streaming.
    """
    if not PODCAST_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Podcast service unavailable. Podcast module not loaded."
        )
    
    try:
        logger.info(f"Generating podcast: mode={request.mode}, target={request.target_minutes}min")
        
        # Step 1: Validate input and get insights
        insights_data = None
        
        if request.insights:
            # Validate insights structure
            if not isinstance(request.insights, dict):
                raise HTTPException(status_code=400, detail="Insights must be a dictionary")
            
            insights_data = request.insights
            logger.info("Using provided insights")
            
        else:
            # Generate insights from text + connections
            if not request.selected_text and not request.connections:
                raise HTTPException(
                    status_code=400, 
                    detail="Must provide either 'insights' or 'selected_text'/'connections'"
                )
            
            if not request.selected_text or len(request.selected_text.strip()) < 10:
                if not request.connections or len(request.connections) == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="Selected text is too short and no connections provided"
                    )
            
            logger.info("Generating insights from selected text and connections")
            
            insight_request = InsightRequest(
                selected_text=request.selected_text or "",
                connections=request.connections or [],
                source_document=request.source_document,
                context={"origin": "podcast_generation"}
            )
            
            try:
                insights_data = await process_insights_with_validation(insight_request)
            except Exception as e:
                logger.error(f"Failed to generate insights: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate insights for podcast: {str(e)}"
                )
        
        if not insights_data:
            raise HTTPException(status_code=500, detail="No insights data available")

        # Step 2: Validate insights have minimum required content
        required_keys = ["summary", "key_takeaways"]
        missing_keys = [key for key in required_keys if not insights_data.get(key)]
        
        if missing_keys:
            logger.warning(f"Insights missing required keys: {missing_keys}")
            # Add fallbacks
            if "summary" not in insights_data or not insights_data["summary"]:
                insights_data["summary"] = "Analysis of document content completed"
            if "key_takeaways" not in insights_data or not insights_data["key_takeaways"]:
                insights_data["key_takeaways"] = ["Key information extracted from content"]

        # Step 3: Optimize content for target duration
        if request.target_minutes <= 2:
            # Very short podcast - limit content
            insights_data["key_takeaways"] = insights_data.get("key_takeaways", [])[:2]
            insights_data["did_you_know"] = insights_data.get("did_you_know", [])[:1]
            insights_data["examples"] = insights_data.get("examples", [])[:1]
        elif request.target_minutes >= 5:
            # Longer podcast - ensure sufficient content
            if len(insights_data.get("key_takeaways", [])) < 3:
                # Add filler takeaways if needed
                takeaways = insights_data.get("key_takeaways", [])
                while len(takeaways) < 3:
                    takeaways.append(f"Additional insight from the analysis")
                insights_data["key_takeaways"] = takeaways

        # Step 4: Generate the podcast audio
        logger.info("Creating podcast audio file...")
        
        try:
            output_path = podcast_module.create_podcast_from_insights(
                insights=insights_data,
                filename=None,  # Use temp file
                mode=request.mode
            )
            
            if not os.path.exists(output_path):
                raise RuntimeError("Podcast file was not created")
            
            file_size = os.path.getsize(output_path)
            if file_size == 0:
                raise RuntimeError("Podcast file is empty")
                
            logger.info(f"Podcast created: {output_path} ({file_size} bytes)")
            
        except Exception as e:
            logger.error(f"Podcast creation failed: {e}")
            
            # Provide helpful error messages
            error_msg = str(e)
            if "TTS" in error_msg or "speech" in error_msg.lower():
                detail = "Text-to-speech service unavailable. Please install gTTS, pyttsx3, or configure Azure Speech Services."
            elif "pydub" in error_msg.lower():
                detail = "Audio processing unavailable. Please install pydub: pip install pydub"
            elif "No TTS backend" in error_msg:
                detail = "No text-to-speech backend available. Install with: pip install gtts pyttsx3"
            else:
                detail = f"Podcast generation failed: {error_msg}"
            
            raise HTTPException(status_code=500, detail=detail)

        # Step 5: Stream the file back
        def generate_file_stream(file_path: str, chunk_size: int = 8192):
            """Stream file with proper cleanup."""
            try:
                with open(file_path, "rb") as audio_file:
                    while True:
                        chunk = audio_file.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
            except Exception as e:
                logger.error(f"Error streaming file: {e}")
                raise
            finally:
                # Clean up temp file after streaming
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.debug(f"Cleaned up temp file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {file_path}: {e}")

        # Get file info for headers
        file_size = os.path.getsize(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"podcast_{timestamp}.mp3"
        
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(file_size),
            "Cache-Control": "no-cache"
        }
        
        return StreamingResponse(
            generate_file_stream(output_path),
            media_type="audio/mpeg",
            headers=headers
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception("Unexpected error in podcast generation: %s", e)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during podcast generation: {str(e)}"
        )

# FIXED: JSON endpoint with better error handling
@router.post("/generate-podcast-json")
async def generate_podcast_json_endpoint(request: PodcastRequest):
    """
    Generate podcast and return JSON with base64 audio data.
    FIXED: Better validation and error handling.
    """
    if not PODCAST_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Podcast service unavailable. Required dependencies not installed."
        )
    
    try:
        logger.info(f"Generating podcast (JSON): mode={request.mode}")
        
        # Step 1: Get insights (same validation as streaming endpoint)
        insights_data = None
        
        if request.insights:
            if not isinstance(request.insights, dict):
                raise HTTPException(status_code=400, detail="Insights must be a dictionary")
            insights_data = request.insights
        else:
            if not request.selected_text and not request.connections:
                raise HTTPException(
                    status_code=400,
                    detail="Must provide either insights or selected_text/connections"
                )
            
            insight_request = InsightRequest(
                selected_text=request.selected_text or "",
                connections=request.connections or [],
                source_document=request.source_document,
                context={"origin": "podcast_json"}
            )
            
            try:
                insights_data = await process_insights_with_validation(insight_request)
            except Exception as e:
                logger.error(f"Insight generation failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")
        
        if not insights_data:
            raise HTTPException(status_code=500, detail="No insights available")

        # Step 2: Create podcast
        try:
            output_path = podcast_module.create_podcast_from_insights(
                insights=insights_data,
                filename=None,
                mode=request.mode
            )
            
            if not os.path.exists(output_path):
                raise RuntimeError("No output file created")
                
        except Exception as e:
            logger.error(f"Podcast creation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Podcast generation failed: {str(e)}")

        # Step 3: Read and encode audio
        try:
            import base64
            
            with open(output_path, "rb") as f:
                audio_data = f.read()
            
            if len(audio_data) == 0:
                raise RuntimeError("Generated audio file is empty")
            
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Clean up
            os.remove(output_path)
            
            return {
                "status": "success",
                "audio_base64": audio_b64,
                "mime_type": "audio/mpeg",
                "size_bytes": len(audio_data),
                "mode": request.mode,
                "timestamp": datetime.now().isoformat(),
                "insights_summary": insights_data.get("summary", "")[:200]
            }
            
        except Exception as e:
            logger.error(f"Failed to process audio: {e}")
            # Clean up on error
            try:
                os.remove(output_path)
            except:
                pass
            raise HTTPException(status_code=500, detail=f"Failed to process audio: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in JSON podcast generation: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# FIXED: Enhanced health check
@router.get("/podcast-health")
async def podcast_health_check():
    """Comprehensive podcast service health check."""
    health_info = {
        "timestamp": datetime.now().isoformat(),
        "podcast_module_available": PODCAST_AVAILABLE,
        "tts_backends": {},
        "audio_processing": False,
        "errors": [],
        "recommendations": [],
        "overall_healthy": False  # Initialize as False
    }
    
    if not PODCAST_AVAILABLE:
        health_info["errors"].append("Podcast module failed to import")
        health_info["recommendations"].append("Check podcast.py file and dependencies")
        health_info["overall_healthy"] = False
        return health_info  # This return was present but there was a logic flow issue
    
    try:
        # Check TTS backends
        health_info["tts_backends"] = {
            "coqui": getattr(podcast_module, 'HAS_COQUI', False),
            "azure": getattr(podcast_module, 'HAS_AZURE', False),
            "pyttsx3": getattr(podcast_module, 'HAS_PYTTSX3', False),
            "gtts": getattr(podcast_module, 'HAS_GTTS', False)
        }
        
        # Check audio processing
        try:
            import pydub
            health_info["audio_processing"] = True
        except ImportError:
            health_info["audio_processing"] = False
            health_info["errors"].append("pydub not available for audio processing")
            health_info["recommendations"].append("Install pydub: pip install pydub")
        
        # Check if any TTS backend is available
        available_tts = [name for name, available in health_info["tts_backends"].items() if available]
        
        if not available_tts:
            health_info["errors"].append("No TTS backends available")
            health_info["recommendations"].extend([
                "Install gTTS: pip install gtts",
                "Install pyttsx3: pip install pyttsx3", 
                "Install Azure Speech: pip install azure-cognitiveservices-speech",
                "Configure Azure Speech: set AZURE_TTS_KEY and AZURE_TTS_REGION environment variables"
            ])
        else:
            health_info["available_tts"] = available_tts
        
        # Environment checks
        env_checks = {}
        if health_info["tts_backends"].get("azure"):
            azure_key = os.environ.get("AZURE_TTS_KEY")
            azure_region = os.environ.get("AZURE_TTS_REGION") 
            env_checks["azure_configured"] = bool(azure_key and azure_region)
            if not env_checks["azure_configured"]:
                health_info["recommendations"].append("Set AZURE_TTS_KEY and AZURE_TTS_REGION for Azure Speech")
        
        health_info["environment"] = env_checks
        
        # Overall health determination
        health_info["overall_healthy"] = (
            PODCAST_AVAILABLE and
            health_info["audio_processing"] and
            len(available_tts) > 0
        )
        
        # Performance test - only if basic health is good
        if health_info["overall_healthy"]:
            try:
                test_insights = {
                    "summary": "Test summary for health check",
                    "key_takeaways": ["Test takeaway for validation"],
                    "did_you_know": [],
                    "contradictions": [],
                    "examples": [{"text": "Test example", "document_name": None, "page_number": None}],
                    "additional_insights": []
                }
                
                # Test script generation (lightweight validation)
                if hasattr(podcast_module, 'generate_podcast_script'):
                    segments = podcast_module.generate_podcast_script(test_insights, "overview")
                    health_info["script_generation"] = len(segments) > 0
                    health_info["test_segments_count"] = len(segments)
                else:
                    health_info["script_generation"] = False
                    health_info["errors"].append("generate_podcast_script function not found")
                    
            except Exception as e:
                health_info["script_generation"] = False
                health_info["errors"].append(f"Script generation test failed: {str(e)[:100]}")
                logger.warning("Script generation test failed: %s", e)
        else:
            health_info["script_generation"] = False
            health_info["recommendations"].append("Fix basic health issues before testing script generation")
        
        # Add service status summary
        if health_info["overall_healthy"]:
            health_info["status"] = "healthy"
            health_info["message"] = "Podcast service is fully operational"
        elif available_tts:
            health_info["status"] = "degraded" 
            health_info["message"] = "Podcast service partially operational - some features may not work"
        else:
            health_info["status"] = "unhealthy"
            health_info["message"] = "Podcast service is not operational - missing critical dependencies"
        
    except Exception as e:
        health_info["errors"].append(f"Health check failed: {str(e)[:200]}")
        health_info["overall_healthy"] = False
        health_info["status"] = "error"
        health_info["message"] = "Health check encountered an error"
        logger.exception("Health check error: %s", e)
    
    return health_info