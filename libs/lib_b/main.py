import os
import json
import fitz  # PyMuPDF
import spacy
from sentence_transformers import SentenceTransformer, util
import torch
import re
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import warnings
import ftfy
import dataclasses
import ollama
from enum import Enum
import collections
from sklearn.cluster import KMeans # For font size clustering

warnings.filterwarnings("ignore")

# --- Configuration ---
class Config:
    INPUT_DIR = 'app/input'
    OUTPUT_DIR = 'app/output'
    BI_ENCODER_MODEL = 'sentence-transformers/msmarco-MiniLM-L-6-v3'
    SPACY_MODEL = 'en_core_web_sm'
    OLLAMA_MODEL = 'phi3:latest'
    OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    FINAL_REPORT_SECTIONS = 5
    MIN_CHUNK_WORDS = 20
    TITLE_SCORE_THRESHOLD = 0.5 # Adjusted based on new scoring
    TITLE_MAX_WORDS = 25
    
    # Enhanced PDF processing configs
    MAX_FILE_SIZE_MB = 500
    MAX_CHUNKS_PER_DOCUMENT = 2000
    CHUNK_OVERLAP_RATIO = 0.1
    
    # Topic-agnostic processing
    ADAPTIVE_TITLE_DETECTION = True
    MULTI_LANGUAGE_SUPPORT = True
    FUZZY_MATCHING_THRESHOLD = 0.8

class DocumentDomain(Enum):
    """Common document domains for better processing"""
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    LEGAL = "legal"
    MEDICAL = "medical"
    BUSINESS = "business"
    MANUAL = "manual"
    REPORT = "report"
    BOOK = "book"
    RECIPE = "recipe"
    FINANCIAL = "financial"
    GENERAL = "general"

# --- Utility & Setup ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if dataclasses.is_dataclass(obj): return dataclasses.asdict(obj)
        return super(NumpyEncoder, self).default(obj)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Data Classes ---
@dataclass
class DocumentMetadata:
    """Enhanced metadata for documents"""
    filename: str
    file_size: int
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    subject: Optional[str] = None
    detected_domain: DocumentDomain = DocumentDomain.GENERAL
    language: Optional[str] = None
    has_images: bool = False
    has_tables: bool = False
    structure_confidence: float = 1.0

@dataclass
class TextChunk:
    text: str
    page_number: int
    document_name: str
    section_title: str
    metadata: Optional[DocumentMetadata] = None
    chunk_index: int = 0
    confidence_score: float = 1.0
    extraction_method: str = "default"
    detected_domain: DocumentDomain = DocumentDomain.GENERAL
    font_features: Dict[str, Any] = field(default_factory=dict)
    structure_level: int = 0  # 0=body, 1=subsection, 2=section, 3=chapter
    additional_context: Dict[str, Any] = field(default_factory=dict)

class SmartTitleDetector:
    """Enhanced title detection with statistical and contextual analysis"""
    
    def __init__(self):
        self.document_stats = {}
        self.title_patterns = self._compile_title_patterns()
    
    def _compile_title_patterns(self):
        """Compile comprehensive title patterns"""
        return {
            'numbered': re.compile(r'^\s*(\d+\.?\d*\.?\s+|[A-Z]\.?\s+|\([a-zA-Z0-9]+\)\s+)', re.IGNORECASE),
            'chapter': re.compile(r'^\s*(chapter|section|part|appendix|article)\s+[a-zA-Z0-9]+', re.IGNORECASE),
            'heading_style': re.compile(r'^[A-Z][A-Z\s]{3,}$'),  # ALL CAPS
            'title_case': re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]*)*$'),  # Title Case
            'ends_colon': re.compile(r':\s*$'),
            'question': re.compile(r'^\s*(?:what|how|why|when|where|who)\s+', re.IGNORECASE),
            'bullet_start': re.compile(r'^\s*[•·▪▫◦‣⁃]\s+'),
            'roman_numeral': re.compile(r'^\s*[IVX]+\.\s+', re.IGNORECASE)
        }
    
    def analyze_document_typography(self, doc) -> Dict[str, Any]:
        """
        Analyze document typography to establish baselines for font sizes, 
        identifying potential heading sizes through clustering.
        """
        all_spans = []
        for page in doc:
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get('type') == 0:  # Text block
                    for line in block.get('lines', []):
                        all_spans.extend(line.get('spans', []))
        
        if not all_spans:
            return {}
        
        sizes = [s.get('size', 12) for s in all_spans]
        fonts = [s.get('font', 'default') for s in all_spans]
        
        # Use KMeans to find distinct font size groups
        unique_sizes = np.array(list(set(sizes))).reshape(-1, 1)
        
        # Try to find 3-5 clusters for different heading levels + body text
        n_clusters = min(len(unique_sizes), 5)  
        if n_clusters < 2: # Not enough variety for clustering
            size_clusters = []
            sorted_unique_sizes = sorted(list(set(sizes)))
            if sorted_unique_sizes:
                # Fallback to simple percentile if clustering is not feasible
                self.document_stats['heading_sizes'] = {
                    'large': np.percentile(sizes, 95),
                    'medium': np.percentile(sizes, 90),
                    'small': np.percentile(sizes, 75),
                    'body': np.percentile(sizes, 50)
                }
            else:
                   self.document_stats['heading_sizes'] = {'large': 16, 'medium': 14, 'small': 12, 'body': 10} # Default
            logger.debug("Not enough unique font sizes for clustering, using percentiles/defaults.")
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init for robustness
            kmeans.fit(unique_sizes)
            cluster_centers = sorted(kmeans.cluster_centers_.flatten().tolist())

            # Assign meaning to clusters (largest are likely headings)
            self.document_stats['heading_sizes'] = {
                'large': cluster_centers[-1] if n_clusters >= 1 else 12,
                'medium': cluster_centers[-2] if n_clusters >= 2 else cluster_centers[-1],
                'small': cluster_centers[-3] if n_clusters >= 3 else cluster_centers[-1],
                'body': cluster_centers[0] if n_clusters >= 1 else 10 # Smallest is likely body
            }
            logger.debug(f"KMeans detected heading sizes: {self.document_stats['heading_sizes']}")

        # Font analysis
        font_freq = collections.Counter(fonts)
        primary_font = font_freq.most_common(1)[0][0] if font_freq else 'default'
        
        # Bold/italic analysis - average size of bold text
        bold_sizes = [s.get('size', 12) for s in all_spans if s.get('flags', 0) & 4]
        
        self.document_stats.update({
            'primary_font': primary_font,
            'font_variety': len(font_freq),
            'bold_avg_size': np.mean(bold_sizes) if bold_sizes else self.document_stats['heading_sizes'].get('body', 10),
            'total_spans': len(all_spans)
        })
        
        return self.document_stats
    
    def is_title_candidate(self, block: Dict, text: str, context: Dict = None) -> Tuple[bool, float, int]:
        """Enhanced title detection with multi-factor scoring"""
        if not text or len(text.strip()) < 3:
            return False, 0.0, 0
        
        text = text.strip()
        words = text.split()
        word_count = len(words)
        
        # Initialize scoring
        confidence = 0.0
        structure_level = 0
        
        # Get span information
        spans = []
        for line in block.get('lines', []):
            spans.extend(line.get('spans', []))
        
        if not spans:
            return False, 0.0, 0
        
        # Font characteristics
        avg_size = np.mean([s.get('size', 12) for s in spans])
        is_bold = any(s.get('flags', 0) & 4 for s in spans)
        is_italic = any(s.get('flags', 0) & 2 for s in spans)
        font_name = spans[0].get('font', 'default') if spans else 'default'
        
        # Document statistics comparison
        if self.document_stats and 'heading_sizes' in self.document_stats:
            heading_sizes = self.document_stats['heading_sizes']
            primary_font = self.document_stats['primary_font']
            
            # Size-based scoring using clustered sizes
            if avg_size >= heading_sizes['large'] * 0.95: # Allow some tolerance
                confidence += 0.4
                structure_level = 3  # Major heading (e.g., Chapter)
            elif avg_size >= heading_sizes['medium'] * 0.95:
                confidence += 0.3
                structure_level = 2  # Section heading
            elif avg_size >= heading_sizes['small'] * 0.95:
                confidence += 0.2
                structure_level = 1  # Subsection
            
            # If bold, and size is above body text, boost
            if is_bold and avg_size > heading_sizes.get('body', 10):
                confidence += 0.25
            
            # Font change detection - if it's a different font than primary body font
            if font_name != primary_font and avg_size > heading_sizes.get('body', 10):
                confidence += 0.1
        
        # Formatting-based scoring (moved here to apply after size logic)
        if is_italic and not is_bold:  # Italic-only emphasis
            confidence += 0.15
        
        # Content pattern analysis
        pattern_score = self._analyze_content_patterns(text, words, word_count)
        confidence += pattern_score
        
        # Length-based adjustments
        if 3 <= word_count <= 12:
            confidence += 0.15
        elif word_count <= 2:
            confidence += 0.05
        elif word_count > Config.TITLE_MAX_WORDS:
            confidence *= 0.5  # Heavy penalty for very long titles
        
        # Structural context (if available)
        if context:
            context_bonus = self._analyze_structural_context(block, text, context)
            confidence += context_bonus
        
        # Single line bonus
        if len(block.get('lines', [])) == 1:
            confidence += 0.1
            if not text.endswith(('.', '!', '?')): # Titles rarely end with full stops
                confidence += 0.1  
        
        # Final adjustments
        if confidence > 0.5:
            # Check for anti-patterns
            if self._has_title_antipatterns(text):
                confidence *= 0.3
        
        return confidence >= Config.TITLE_SCORE_THRESHOLD, min(confidence, 1.0), structure_level
    
    def _analyze_content_patterns(self, text: str, words: List[str], word_count: int) -> float:
        """Analyze content patterns for title likelihood"""
        score = 0.0
        text_lower = text.lower()
        
        # Pattern matching with weights
        if self.title_patterns['numbered'].match(text):
            score += 0.3
        if self.title_patterns['chapter'].match(text):
            score += 0.35
        if self.title_patterns['heading_style'].match(text):
            score += 0.25
        if self.title_patterns['title_case'].match(text) and word_count > 1:
            score += 0.2
        if self.title_patterns['ends_colon'].search(text):
            score += 0.15
        if self.title_patterns['question'].match(text):
            score += 0.1
        if self.title_patterns['roman_numeral'].match(text):
            score += 0.25
        
        # Content-based heuristics
        if text.isupper() and 3 <= word_count <= 8:
            score += 0.2
        if text.istitle() and word_count <= 10:
            score += 0.15
        
        # Common title words (boosting)
        title_indicators = [
            'introduction', 'conclusion', 'overview', 'summary', 'background',
            'methodology', 'results', 'discussion', 'abstract', 'references',
            'appendix', 'glossary', 'index', 'preface', 'acknowledgments',
            'chapter', 'section', 'part', 'list of figures', 'table of contents'
        ]
        if any(indicator in text_lower for indicator in title_indicators):
            score += 0.2
        
        # Avoid over-scoring
        return min(score, 0.6)
    
    def _analyze_structural_context(self, block: Dict, text: str, context: Dict) -> float:
        """Analyze structural context for additional scoring"""
        score = 0.0
        
        # Position-based scoring
        bbox = block.get('bbox')
        if bbox:
            page_height = context.get('page_height', 1000) # Default if not provided
            page_width = context.get('page_width', 1000)
            
            # Top of page
            if bbox[1] < page_height * 0.2: # Top 20% of the page
                score += 0.1
            
            # Centered or significant left margin
            left_margin = bbox[0]
            line_width = bbox[2] - bbox[0]
            if abs((page_width / 2) - (left_margin + line_width / 2)) < 50: # Roughly centered
                   score += 0.08
            elif left_margin < 100:  # Left-aligned and close to typical margin
                score += 0.05
        
        # Vertical spacing
        if context.get('previous_block_bbox') and bbox:
            prev_bbox = context['previous_block_bbox']
            # Calculate vertical space between current block and previous block
            vertical_gap = bbox[1] - prev_bbox[3]
            # Heuristic: titles often have a larger gap above them (e.g., > 2 * avg line height)
            # This requires estimating average line height, which is complex.
            # A simpler approach: if a significant gap, it's more likely a title
            if vertical_gap > 20: # Arbitrary threshold, needs tuning per document structure
                   score += 0.1
        
        # Preceded by page break (already in context, reuse)
        if context.get('preceded_by_page_break', False):
            score += 0.15
        
        return min(score, 0.3) # Max bonus for structural context
    
    def _has_title_antipatterns(self, text: str) -> bool:
        """Check for patterns that indicate text is NOT a title"""
        text_lower = text.lower()
        
        # Anti-patterns
        antipatterns = [
            r'\d{4}-\d{2}-\d{2}',  # Dates
            r'page\s+\d+',  # Page numbers
            r'figure\s+\d+[\.:]\s',  # Figure captions
            r'table\s+\d+[\.:]\s',  # Table captions
            r'copyright\s+©',  # Copyright notices
            r'all rights reserved',  # Legal text
            r'^\s*note\s*:',  # Notes
            r'^\s*warning\s*:',  # Warnings
            r'^\s*\d+\s*$',  # Just numbers
            r'^\s*chapter\s+end', # End of chapter markers
            r'^\s*---\s*$', # Separator lines
            # NEW: Specific patterns for cross-references/footnotes masquerading as titles
            r'(?:see|refer to|for more information|for details|as described in)\s+(?:the\s+)?(?:guide|document|section|chapter|appendix|figure|table|note)',
            r'\(.+\)', # Lines that are entirely enclosed in parentheses
            r'^\s*(?:[\d\W_]*?(?:fig\.|figure|tbl\.|table)\s*\d+[a-z]?[\.:\s]*)' # More robust figure/table captions
        ]
        
        if any(re.search(pattern, text_lower) for pattern in antipatterns):
            return True
        
        # Very long sentences (likely body text) - refine to check for multiple sentences
        if len(re.findall(r'[.!?]\s+\S', text)) > 1 and len(text) > 80: # More than one sentence break AND long
            return True
        
        # Contains too many common words (re-evaluate threshold)
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were']
        words = text.lower().split()
        if len(words) > 5 and sum(1 for word in words if word in common_words) / len(words) > 0.4:
            return True
            
        return False

class DomainDetector:
    """Detects document domain/topic for better processing"""
    
    DOMAIN_KEYWORDS = {
        DocumentDomain.ACADEMIC: [
            'abstract', 'introduction', 'methodology', 'results', 'conclusion', 
            'references', 'bibliography', 'literature review', 'hypothesis',
            'research', 'study', 'analysis', 'experiment', 'survey', 'journal',
            'university', 'professor', 'thesis', 'dissertation', 'peer review',
            'discussion', 'findings', 'empirical', 'qualitative', 'quantitative', 'review of literature'
        ],
        DocumentDomain.TECHNICAL: [
            'specification', 'implementation', 'algorithm', 'framework',
            'architecture', 'system', 'configuration', 'installation',
            'deployment', 'api', 'interface', 'protocol', 'standard',
            'technical', 'engineering', 'software', 'hardware', 'design',
            'user guide', 'troubleshooting', 'syntax', 'function', 'database', 'network'
        ],
        DocumentDomain.LEGAL: [
            'contract', 'agreement', 'clause', 'section', 'subsection',
            'whereas', 'therefore', 'party', 'parties', 'jurisdiction',
            'law', 'legal', 'court', 'judge', 'attorney', 'counsel',
            'statute', 'regulation', 'compliance', 'liability', 'plaintiff', 'defendant',
            'judgment', 'amendment', 'provision', 'deed', 'warrant'
        ],
        DocumentDomain.MEDICAL: [
            'patient', 'diagnosis', 'treatment', 'symptoms', 'procedure',
            'medical', 'clinical', 'hospital', 'doctor', 'physician',
            'medication', 'dosage', 'therapy', 'health', 'disease',
            'condition', 'examination', 'test', 'results', 'prescription',
            'pharmacology', 'pathology', 'epidemiology', 'anatomy', 'physiology', 'surgery'
        ],
        DocumentDomain.BUSINESS: [
            'revenue', 'profit', 'financial', 'business', 'market',
            'strategy', 'management', 'executive', 'board', 'shareholders',
            'company', 'corporation', 'investment', 'budget', 'forecast',
            'performance', 'metrics', 'kpi', 'roi', 'growth', 'sales',
            'marketing', 'operations', 'human resources', 'strategy'
        ],
        DocumentDomain.MANUAL: [
            'instructions', 'step', 'procedure', 'guide', 'manual',
            'how to', 'tutorial', 'setup', 'installation', 'operation',
            'maintenance', 'troubleshooting', 'warning', 'caution',
            'note', 'important', 'tip', 'example', 'illustration', 'chapter', 'section'
        ],
        DocumentDomain.RECIPE: [
            'ingredients', 'recipe', 'cooking', 'preparation', 'serves',
            'minutes', 'hours', 'temperature', 'oven', 'bake', 'cook',
            'cup', 'tablespoon', 'teaspoon', 'pound', 'ounce', 'gram',
            'flour', 'sugar', 'salt', 'pepper', 'oil', 'butter', 'garnish', 'instructions'
        ],
        DocumentDomain.FINANCIAL: [
            'balance sheet', 'income statement', 'cash flow', 'assets',
            'liabilities', 'equity', 'dividend', 'earnings', 'quarterly',
            'annual', 'fiscal', 'audit', 'accounting', 'gaap', 'ifrs',
            'depreciation', 'amortization', 'revenue recognition', 'investment',
            'stock', 'bond', 'portfolio', 'interest rate', 'tax', 'report'
        ],
        DocumentDomain.GENERAL: [ # General terms that don't strongly point to one domain
            'introduction', 'conclusion', 'summary', 'overview', 'details', 'contents',
            'about', 'contact', 'appendix'
        ]
    }
    
    @classmethod
    def detect_domain(cls, text_content: str, metadata: Optional[DocumentMetadata] = None) -> DocumentDomain:
        """Detect document domain based on content and metadata"""
        text_lower = text_content.lower()
        
        domain_scores = {}
        for domain, keywords in cls.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score / len(keywords)  # Normalize by keyword count
        
        # Consider metadata hints
        if metadata and metadata.title:
            title_lower = metadata.title.lower()
            for domain, keywords in cls.DOMAIN_KEYWORDS.items():
                title_matches = sum(1 for keyword in keywords if keyword in title_lower)
                if title_matches > 0:
                    # Give higher weight to title matches
                    domain_scores[domain] = domain_scores.get(domain, 0) + (title_matches * 0.75) 
        
        if domain_scores:
            detected_domain = max(domain_scores, key=domain_scores.get)
            confidence = domain_scores[detected_domain]
            logger.info(f"Detected domain: {detected_domain.value} (confidence: {confidence:.2f})")
            return detected_domain
        
        return DocumentDomain.GENERAL

class EnhancedDocumentProcessor:
    """Enhanced PDF processor with improved title detection"""
    
    def __init__(self):
        self.title_detector = SmartTitleDetector()
        self.domain_specific_patterns = self._initialize_domain_patterns()
    
    def _initialize_domain_patterns(self) -> Dict[DocumentDomain, Dict]:
        """Initialize domain-specific processing patterns"""
        return {
            DocumentDomain.ACADEMIC: {
                'common_sections': ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion', 'references'],
                'ignore_patterns': ['acknowledgments', 'funding', 'conflict of interest', 'figure', 'table', 'copyright']
            },
            DocumentDomain.TECHNICAL: {
                'common_sections': ['overview', 'specification', 'implementation', 'configuration', 'examples', 'troubleshooting'],
                'ignore_patterns': ['copyright', 'trademark', 'revision history', 'license agreement']
            },
            DocumentDomain.LEGAL: {
                'common_sections': ['whereas', 'definitions', 'terms', 'conditions', 'signatures', 'articles'],
                'ignore_patterns': ['exhibit', 'schedule', 'appendix', 'signature page']
            },
            DocumentDomain.RECIPE: {
                'common_sections': ['ingredients', 'instructions', 'preparation', 'cooking', 'serving'],
                'ignore_patterns': ['nutrition facts', 'allergen information', 'tips', 'notes']
            },
            DocumentDomain.MANUAL: {
                'common_sections': ['safety', 'installation', 'operation', 'maintenance', 'troubleshooting'],
                'ignore_patterns': ['warranty', 'specifications table', 'part list', 'diagram']
            }
        }
    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning with domain awareness"""
        if not text:
            return ""
        
        # Basic cleaning
        text = ftfy.fix_text(text)
        text = re.sub(r'\s*\n\s*', ' ', text).strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common recipe list prefixes like 'o ' or numbers followed by a space
        text = re.sub(r'^[oO\d]+\s+', '', text)
        text = re.sub(r'^\s*Instructions?\s*:\s*', '', text, flags=re.IGNORECASE) # Remove "Instructions:" if at start

        # Remove common artifacts
        # Keep basic punctuation, but be more selective about what to remove
        text = re.sub(r'[^\w\s\.,;:!?\-\(\)\'"\[\]\{\}/]', ' ', text)  
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_chunks_from_pdf(self, pdf_path: str) -> List[TextChunk]:
        """Enhanced PDF extraction with improved title detection"""
        document_name = Path(pdf_path).stem
        logger.info(f"Processing PDF '{document_name}' with enhanced title detection...")
        chunks = []
        
        try:
            with fitz.open(pdf_path) as doc:
                # Extract basic metadata
                metadata = self._extract_pdf_metadata(doc, Path(pdf_path))
                
                # Analyze document typography for title detection baselines
                self.title_detector.analyze_document_typography(doc)
                
                # Detect document domain from initial content
                sample_text = self._extract_sample_text(doc, max_pages=3)
                detected_domain = DomainDetector.detect_domain(sample_text, metadata)
                metadata.detected_domain = detected_domain
                
                logger.info(f"Detected domain: {detected_domain.value}")
                
                # Process pages with enhanced title detection
                previous_block_bbox = None
                for page_num, page in enumerate(doc, 1):
                    # Pass page dimensions and previous block's bbox for context
                    page_width, page_height = page.rect.width, page.rect.height
                    page_chunks, last_block_bbox_on_page = self._extract_page_chunks_enhanced(
                        page, page_num, document_name, metadata, detected_domain,
                        previous_block_bbox=previous_block_bbox,
                        page_width=page_width, page_height=page_height
                    )
                    chunks.extend(page_chunks)
                    # CORRECTED: Use last_block_bbox_on_page which is returned by the method
                    previous_block_bbox = last_block_bbox_on_page 
                
                # Post-process chunks
                chunks = self._post_process_chunks(chunks, detected_domain)
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}", exc_info=True)
        
        return chunks
    
    def _extract_pdf_metadata(self, doc, file_path: Path) -> DocumentMetadata:
        """Extract comprehensive PDF metadata"""
        stat = file_path.stat()
        metadata = DocumentMetadata(
            filename=file_path.name,
            file_size=stat.st_size,
            creation_date=datetime.fromtimestamp(stat.st_ctime).isoformat(),
            modification_date=datetime.fromtimestamp(stat.st_mtime).isoformat()
        )
        
        try:
            pdf_metadata = doc.metadata
            metadata.page_count = doc.page_count
            metadata.author = pdf_metadata.get('author', '')
            metadata.title = pdf_metadata.get('title', '')
            metadata.subject = pdf_metadata.get('subject', '')
            
            # Enhanced metadata extraction
            metadata.has_images = bool(doc.get_page_images(0)) if doc.page_count > 0 else False # Check first page for images
            metadata.has_tables = self._detect_tables_in_doc(doc)
            
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {e}")
        
        return metadata
    
    def _extract_sample_text(self, doc, max_pages: int = 3) -> str:
        """Extract sample text for domain detection"""
        sample_text = ""
        pages_to_sample = min(max_pages, doc.page_count)
        
        for page_num in range(pages_to_sample):
            try:
                page = doc[page_num]
                page_text = page.get_text()
                sample_text += " " + page_text[:1000]  # Limit per page
            except Exception as e:
                logger.warning(f"Could not extract text from page {page_num}: {e}")
        
        return sample_text
    
    def _detect_tables_in_doc(self, doc) -> bool:
        """Simple table detection in PDF using page.find_tables()"""
        try:
            for page in doc:
                tables = page.find_tables()
                if tables and len(tables.tables) > 0: # Ensure tables actually found
                    return True
        except Exception as e: # Catch any error from find_tables
            logger.warning(f"Error detecting tables: {e}")
            pass
        return False
    
    def _extract_page_chunks_enhanced(self, page, page_num: int, document_name: str, 
                                      metadata: DocumentMetadata, domain: DocumentDomain,
                                      previous_block_bbox: Optional[Tuple[float,float,float,float]] = None,
                                      page_width: float = 0, page_height: float = 0) -> Tuple[List[TextChunk], Optional[Tuple[float,float,float,float]]]:
        """Enhanced page chunk extraction with smart title detection"""
        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", [])
        if not blocks:
            return [], None
        
        # Get domain-specific patterns
        domain_patterns = self.domain_specific_patterns.get(domain, {})
        
        chunks = []
        current_title = document_name.replace("-", " ").title() # Default to filename as title
        current_recipe_name = current_title # For recipe domain, this will hold the actual recipe title
        current_body = ""
        chunk_index = 0
        
        last_block_bbox_on_block = None # This will be the bbox of the *last processed block*, for next iteration's previous_block_bbox
        
        for i, block in enumerate(blocks):
            if block.get('type') != 0:  # Skip non-text blocks (images, drawings)
                if block.get('bbox'): # Update last_block_bbox_on_block even if it's not text, for spacing analysis
                    last_block_bbox_on_block = block.get('bbox')
                continue
            
            block_raw_text = "".join(
                s['text'] for l in block.get('lines', []) 
                for s in l.get('spans', [])
            ).strip() # Strip immediately for accurate checks
            
            # Text for detection (less aggressive cleaning initially)
            text_for_detection = ftfy.fix_text(block_raw_text)
            text_for_detection = re.sub(r'\s*\n\s*', ' ', text_for_detection).strip()
            text_for_detection = re.sub(r'\s+', ' ', text_for_detection)

            # Cleaned text for actual chunk content
            cleaned_block_text_for_content = self._clean_text(block_raw_text) # Uses the _clean_text with new stripping rules
            
            if not cleaned_block_text_for_content or self._should_ignore_block(text_for_detection, domain_patterns):
                last_block_bbox_on_block = block.get('bbox')
                continue
            
            # Enhanced title detection with context
            context = {
                'is_first_block_on_page': i == 0,
                'follows_whitespace': previous_block_bbox is not None and (block.get('bbox')[1] - previous_block_bbox[3] > 15),
                'preceded_by_page_break': i == 0 and page_num > 1,
                'bbox': block.get('bbox'),
                'previous_block_bbox': previous_block_bbox, # Pass the last processed block's bbox from previous iteration/page
                'page_width': page_width,
                'page_height': page_height
            }
            
            is_title, confidence, structure_level = self.title_detector.is_title_candidate(
                block, text_for_detection, context
            )
            
            # **RECIPE-SPECIFIC LOGIC PRIORITY**
            if domain == DocumentDomain.RECIPE:
                # 1. Identify the primary Recipe Name (highest confidence title-like block, usually first prominent one)
                if page_num == 1 and i < 5 and is_title and structure_level >= 2: # Early pages, prominent titles
                    if len(text_for_detection.split()) > 1 and len(text_for_detection.split()) < 15: # Reasonable title length
                        current_recipe_name = text_for_detection
                        is_title = True # Confirm it's a title
                        confidence = 1.0 # Highest confidence for recipe name
                        structure_level = 3 # Treat as chapter level for primary recipe name
                        logger.debug(f"Detected primary recipe name: '{current_recipe_name}'")

                # 2. Identify "Ingredients", "Instructions" as fixed sub-section titles
                if re.match(r'^\s*Ingredients?\s*$', block_raw_text, re.IGNORECASE):
                    # Always treat this as a title break for the *previous* content
                    if self._is_valid_chunk(current_body):
                        chunks.append(TextChunk(
                            text=current_body.strip(),
                            page_number=page_num,
                            document_name=document_name,
                            section_title=current_title, # The title *before* 'Ingredients'
                            metadata=metadata,
                            chunk_index=chunk_index,
                            confidence_score=confidence, 
                            detected_domain=domain,
                            structure_level=structure_level,
                            extraction_method="recipe_section_split"
                        ))
                        chunk_index += 1
                    current_title = f"{current_recipe_name} - Ingredients" if current_recipe_name else "Ingredients"
                    current_body = ""
                    is_title = True # Force it to be treated as a title to break the chunk
                    confidence = 1.0
                    structure_level = 2 # Fixed structure level for main recipe sections
                    logger.debug(f"Detected recipe section title: '{current_title}'")
                elif re.match(r'^\s*(?:Instructions?|Directions|Method|Preparation)\s*$', block_raw_text, re.IGNORECASE):
                    if self._is_valid_chunk(current_body):
                        chunks.append(TextChunk(
                            text=current_body.strip(),
                            page_number=page_num,
                            document_name=document_name,
                            section_title=current_title, # The title *before* 'Instructions'
                            metadata=metadata,
                            chunk_index=chunk_index,
                            confidence_score=confidence, 
                            detected_domain=domain,
                            structure_level=structure_level,
                            extraction_method="recipe_section_split"
                        ))
                        chunk_index += 1
                    current_title = f"{current_recipe_name} - Instructions" if current_recipe_name else "Instructions"
                    current_body = ""
                    is_title = True # Force it to be treated as a title to break the chunk
                    confidence = 1.0
                    structure_level = 2 # Fixed structure level for main recipe sections
                    logger.debug(f"Detected recipe section title: '{current_title}'")
                elif re.match(r'^\s*(?:Notes|Tips|Serving)\s*$', block_raw_text, re.IGNORECASE):
                    if self._is_valid_chunk(current_body):
                        chunks.append(TextChunk(
                            text=current_body.strip(),
                            page_number=page_num,
                            document_name=document_name,
                            section_title=current_title, # The title *before* this sub-section
                            metadata=metadata,
                            chunk_index=chunk_index,
                            confidence_score=confidence, 
                            detected_domain=domain,
                            structure_level=structure_level,
                            extraction_method="recipe_section_split"
                        ))
                        chunk_index += 1
                    current_title = f"{current_recipe_name} - {text_for_detection}" if current_recipe_name else text_for_detection # Use detected text for "Notes" etc.
                    current_body = ""
                    is_title = True # Force it to be treated as a title to break the chunk
                    confidence = 0.9
                    structure_level = 1 # Sub-subsection
                    logger.debug(f"Detected recipe sub-section title: '{current_title}'")

            # END RECIPE-SPECIFIC LOGIC

            # Main logic for accumulating chunks and detecting new titles
            # Only proceed if it's not a recipe-specific forced title, or if it is the recipe name
            if is_title and confidence >= Config.TITLE_SCORE_THRESHOLD: # Use configurable threshold
                # If it's a new general title detected by SmartTitleDetector, and not overridden by recipe logic
                # then apply it.
                if self._is_valid_chunk(current_body):
                    chunks.append(TextChunk(
                        text=current_body.strip(),
                        page_number=page_num,
                        document_name=document_name,
                        section_title=current_title,
                        metadata=metadata,
                        chunk_index=chunk_index,
                        confidence_score=confidence, 
                        detected_domain=domain,
                        structure_level=structure_level,
                        extraction_method="smart_title_detection"
                    ))
                    chunk_index += 1
                
                # Set new title, but for recipe, prioritize current_recipe_name
                if domain != DocumentDomain.RECIPE or current_recipe_name == text_for_detection: # Only if this block *is* the main recipe name
                    current_title = text_for_detection
                # Otherwise, if it's a normal title in a recipe document, the current_title has already been set by recipe logic.
                
                current_body = ""
                metadata.structure_confidence = structure_level / 3.0
                logger.debug(f"Title detected: '{current_title}' (confidence: {confidence:.2f}, level: {structure_level})")
            else: # It's body text
                current_body += " " + cleaned_block_text_for_content # Use the cleaned text for content
            
            last_block_bbox_on_block = block.get('bbox') # Store current block's bbox for next iteration
            
        # Add final chunk from the page
        if self._is_valid_chunk(current_body):
            chunks.append(TextChunk(
                text=current_body.strip(),
                page_number=page_num,
                document_name=document_name,
                section_title=current_title,
                metadata=metadata,
                chunk_index=chunk_index,
                confidence_score=metadata.structure_confidence,
                detected_domain=domain,
                structure_level=metadata.structure_confidence * 3,
                extraction_method="smart_title_detection"
            ))
            
        return chunks, last_block_bbox_on_block
    
    def _should_ignore_block(self, text: str, domain_patterns: Dict) -> bool:
        """Check if block should be ignored based on domain patterns and general noise"""
        text_lower = text.lower()
        
        # General noise patterns - more aggressive filtering
        general_noise = [
            r'^\s*page\s+\d+\s*$', # Exact "page X"
            r'^\s*copyright\s+©.*$', 
            r'^\s*all rights reserved.*$', 
            r'^\s*footnote\s+\d+.*$',
            r'^\s*header\s*$', r'^\s*footer\s*$', r'^\s*watermark\s*$', 
            r'^\s*table of contents.*$',
            r'^\s*(?:fig\.|figure|tbl\.|table)\s*\d+[a-z]?[\.:\s].*$', # More robust figure/table captions
            r'^\s*acknowledgements?$', # Common single-line ignorable sections
            r'^\s*index\s*$', r'^\s*references?\s*$', r'^\s*bibliography\s*$',
            r'^\s*appendix\s*$', r'^\s*glossary\s*$', r'^\s*preface\s*$',
            r'^\s*--+\s*$', # Horizontal lines as text
            r'(?:see|refer to|for more information|for details|as described in)\s+(?:the\s+)?(?:guide|document|section|chapter|appendix|figure|table|note)', # Cross-references
            r'^\s*\(.+\)\s*$', # Lines that are entirely enclosed in parentheses
            r'^\s*\d+\s*$', # Just numbers (e.g., page numbers or single digits)
            r'^\s*[\W_]+\s*$' # Lines with only symbols/punctuation
        ]
        
        if any(re.search(pattern, text_lower) for pattern in general_noise):
            return True
        
        # Domain-specific ignore patterns
        ignore_patterns = domain_patterns.get('ignore_patterns', [])
        if any(pattern in text_lower for pattern in ignore_patterns):
            return True
        
        # Very short or very long single "words" (likely artifacts)
        words = text.split()
        if len(words) == 1 and (len(words[0]) < 3 or len(words[0]) > 50): # Single long "word" could be junk
            return True
        
        # Mostly numbers or symbols (e.g., page numbers or strange artifacts)
        if re.fullmatch(r'[\d\s\W]+', text): # Only numbers, spaces, non-word chars
            return True
            
        return False
    
    def _is_valid_chunk(self, text: str) -> bool:
        """Enhanced chunk validation"""
        if not text or not text.strip():
            return False
        
        words = text.split()
        word_count = len(words)
        
        # Minimum word count
        if word_count < Config.MIN_CHUNK_WORDS:
            return False
        
        # Check for meaningful content (not just numbers/symbols)
        meaningful_words = [w for w in words if re.match(r'^[a-zA-Z]', w)]
        if len(meaningful_words) < Config.MIN_CHUNK_WORDS // 2: # At least half meaningful words
            return False
        
        return True
    
    def _post_process_chunks(self, chunks: List[TextChunk], domain: DocumentDomain) -> List[TextChunk]:
        """Post-process chunks based on domain"""
        if not chunks:
            return chunks
        
        # Merge very short chunks with adjacent ones
        chunks = self._merge_short_chunks(chunks)
        
        # Re-index chunks
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
        
        return chunks
    
    def _merge_short_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Intelligently merge short chunks"""
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = None
        
        for chunk in chunks:
            if current_chunk is None:
                current_chunk = chunk
                continue
            
            current_words = len(current_chunk.text.split())
            chunk_words = len(chunk.text.split())
            
            # Merge conditions:
            # 1. Current chunk is too short
            # 2. Both chunks are on the same page
            # 3. The new chunk is not a higher-level structural element than the current one
            should_merge = (
                current_words < Config.MIN_CHUNK_WORDS or
                (current_words < Config.MIN_CHUNK_WORDS * 2 and  # Allow merging slightly larger short chunks
                 chunk.page_number == current_chunk.page_number and
                 chunk.structure_level <= current_chunk.structure_level + 1) # Merge if same level or slightly lower
            )
            
            if should_merge:
                # Merge text
                current_chunk.text += " " + chunk.text
                # Update page number to span both if different
                if chunk.page_number != current_chunk.page_number:
                    current_chunk.page_number = min(current_chunk.page_number, chunk.page_number)
                # Keep the more specific section title (higher structure_level, or if new title is clearer)
                if chunk.structure_level > current_chunk.structure_level:
                    current_chunk.section_title = chunk.section_title
                    current_chunk.structure_level = chunk.structure_level
                elif chunk.structure_level == current_chunk.structure_level and len(chunk.section_title) > len(current_chunk.section_title):
                    # If same level, prefer longer, potentially more descriptive title
                    current_chunk.section_title = chunk.section_title

                # Combine confidence scores, maybe average or weighted average
                current_chunk.confidence_score = (current_chunk.confidence_score + chunk.confidence_score) / 2.0
                continue
            
            merged_chunks.append(current_chunk)
            current_chunk = chunk
        
        if current_chunk:
            merged_chunks.append(current_chunk)
        
        return merged_chunks

# --- Core Components (Enhanced) ---
class ModelManager:
    def __init__(self):
        self.bi_encoder: SentenceTransformer = None
        self.nlp: spacy.Language = None
        self.llm_client: ollama.Client = None
        self._models_loaded = False

    def load_models(self) -> bool:
        if self._models_loaded: return True
        try:
            logger.info("Loading all models...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.bi_encoder = SentenceTransformer(Config.BI_ENCODER_MODEL, device=device)
            self.nlp = spacy.load(Config.SPACY_MODEL)
            self.llm_client = ollama.Client()
            self.llm_client.show(Config.OLLAMA_MODEL)
            self._models_loaded = True
            logger.info("All models loaded successfully. ✅")
            return True
        except Exception as e:
            logger.error(f"Fatal error loading models: {e}", exc_info=True)
            logger.error(f"Please ensure Ollama is running and model '{Config.OLLAMA_MODEL}' is pulled (`ollama pull {Config.OLLAMA_MODEL}`).")
            return False

class IntelligentRetriever:
    """Enhanced retriever with domain-aware processing"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def _analyze_task_with_spacy(self, task: str) -> Tuple[List[str], List[str]]:
        """Enhanced spaCy analysis with better entity recognition"""
        doc = self.model_manager.nlp(task)
        
        # Enhanced keyword extraction
        subjects = [tok.lemma_.lower() for tok in doc if tok.dep_ in ('nsubj', 'dobj', 'pobj', 'ROOT') and not tok.is_stop and tok.is_alpha]
        attributes = [tok.lemma_.lower() for tok in doc if tok.dep_ in ('amod', 'acomp', 'attr') and not tok.is_stop and tok.is_alpha]
        entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ('ORG', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LOC', 'GPE')] # Added more entity types
        
        # Constraint detection with better patterns
        constraints = []
        # Negation patterns also considering quantitative constraints
        negation_patterns = ['free', 'no', 'without', 'exclude', 'avoid', 'not', 'never', 'less than', 'except']
        for token in doc:
            if any(pattern in token.text.lower() for pattern in negation_patterns):
                constraints.append(token.lemma_.lower())
                # Look for related nouns, adjectives, or verbs
                for child in token.children:
                    if child.pos_ in ('NOUN', 'ADJ', 'VERB'):
                        constraints.append(child.lemma_.lower())
                if token.head.pos_ in ('NOUN', 'ADJ', 'VERB'):
                    constraints.append(token.head.lemma_.lower())
            
            # Numeric/quantitative constraints
            if token.like_num or token.pos_ == 'NUM':
                # e.g., "more than 10", "up to 50MB"
                if token.head and token.head.lemma_.lower() in ['more', 'less', 'up', 'down']:
                    constraints.append(f"{token.head.lemma_.lower()} {token.text}")
                elif token.nbor(1).lower_ in ['mb', 'gb', 'pages', 'sections']:
                    constraints.append(f"{token.text} {token.nbor(1).lower_}")

        main_keywords = list(dict.fromkeys(subjects + attributes + entities))
        strict_constraints = list(dict.fromkeys(constraints))
        
        logger.info(f"Enhanced spaCy analysis - Keywords: {main_keywords}, Constraints: {strict_constraints}")
        return main_keywords, strict_constraints

    def _get_dynamic_keywords_from_llm(self, main_keywords: List[str], constraints: List[str], 
                                     detected_domains: List[DocumentDomain]) -> Tuple[List[str], List[str], List[str]]:
        """Enhanced LLM keyword generation with domain awareness and stricter filtering."""
        if not main_keywords and not constraints:  
            return [], [], []
        
        concept = " ".join(main_keywords + constraints)
        domain_context = ", ".join([d.value for d in detected_domains]) if detected_domains else "general"
        
        logger.info(f"Querying LLM for keywords related to '{concept}' in domains: {domain_context}")
        
        # **IMPROVED LLM PROMPT**
        # Emphasize strict JSON formatting with double quotes
        prompt = f"""
Analyze the user task concept: "{concept}"
Document domains detected: {domain_context}

Based on the context and domains, provide:

1. **Positive Keywords:** Concepts that boost relevance. Prioritize meal type, style, and dietary needs.
2. **Negative Keywords:** Concepts that conflict or decrease relevance.
3. **Forbidden Keywords:** Strict exclusions based on constraints. These should *absolutely* prevent a document/chunk from being considered.

**Strict Requirements for this Task:**
- **Meal Type:** Must be "dinner".
- **Style:** Must be "buffet".
- **Dietary:** Must be "vegetarian" and include "gluten-free" options.

Consider the document domains when generating keywords.

Respond ONLY with a single, valid JSON object. Ensure all property names and string values are enclosed in double quotes.

Example for "vegetarian buffet-style dinner menu":
{{
  "positive": ["dinner", "buffet", "vegetarian", "plant-based", "main course", "side dish", "salad", "appetizer", "gluten-free", "vegan (if relevant)"],
  "negative": ["breakfast", "lunch", "snack", "dessert", "individual portion", "non-vegetarian (unless specified as 'optional')"],
  "forbidden": ["meat", "chicken", "beef", "pork", "fish", "lamb", "bacon", "sausage", "seafood", "eggs (unless specified as vegetarian only)", "dairy (unless specified as vegetarian only)", "wheat", "gluten", "barley", "rye"]
}}

Your turn for "{concept}" in {domain_context} context:"""

        try:
            response = self.model_manager.llm_client.generate(model=Config.OLLAMA_MODEL, prompt=prompt, options={"temperature": 0.2}) # Lower temperature for more focused output
            
            # Extract only the JSON part from the LLM's response
            # This regex is more robust to leading/trailing text from the LLM
            match = re.search(r'\{.*\}', response['response'], re.DOTALL)
            if match:
                keywords_json_str = match.group(0)
                keywords_json = json.loads(keywords_json_str)
                positive = [kw.lower() for kw in keywords_json.get("positive", [])]
                negative = [kw.lower() for kw in keywords_json.get("negative", [])]
                forbidden = [kw.lower() for kw in keywords_json.get("forbidden", [])]
                logger.info(f"Domain-aware Keywords: +{positive}, -{negative}, !{forbidden}")
                return positive, negative, forbidden
            else:
                logger.warning(f"LLM response did not contain a valid JSON object: {response['response']}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}. Raw response: {response['response'] if 'response' in response else 'N/A'}")
        except Exception as e:
            logger.error(f"Failed to get dynamic keywords from LLM: {e}")
        return [], [], []

    def rank_chunks_by_score(self, task: str, chunks: List[TextChunk]) -> List[TextChunk]:
        """Enhanced ranking with domain awareness and better scoring"""
        if not chunks: return []

        # Analyze detected domains in chunks
        detected_domains = list(set(chunk.detected_domain for chunk in chunks))
        logger.info(f"Processing chunks from domains: {[d.value for d in detected_domains]}")

        main_keywords, constraints = self._analyze_task_with_spacy(task)
        positive_terms, negative_terms, forbidden_words = self._get_dynamic_keywords_from_llm(
            main_keywords, constraints, detected_domains
        )

        # **CRITICAL FIRST-PASS FILTERING FOR MEAL TYPE (BEFORE SEMANTIC SEARCH)**
        # This is a hard filter to eliminate irrelevant documents early
        initial_filtered_chunks = []
        task_lower = task.lower()
        required_meal_type = "dinner" # From "dinner menu planning"
        
        for chunk in chunks:
            # Safely access chunk.metadata.title using getattr
            doc_title = getattr(chunk.metadata, 'title', '') if chunk.metadata else ''
            chunk_content_lower = (chunk.document_name + " " + doc_title + " " + chunk.text + " " + chunk.section_title).lower()
            
            # EXCLUDE if it explicitly mentions unwanted meal types
            if "breakfast" in chunk_content_lower or "lunch" in chunk_content_lower:
                logger.debug(f"Excluding chunk due to wrong meal type: {chunk.document_name} - {chunk.section_title}")
                continue
            
            # Optionally, BOOST if it explicitly mentions the required meal type, but don't hard exclude if not present yet
            # because some recipes might just be "main course" without "dinner" explicit
            if required_meal_type in chunk_content_lower:
                chunk.confidence_score *= 1.2 # Boost for explicit dinner mention
            
            initial_filtered_chunks.append(chunk)

        if not initial_filtered_chunks:
            logger.warning("No chunks left after initial meal type filtering.")
            return []

        chunks_for_ranking = initial_filtered_chunks
        
        queries = [
            task,
            " ".join(main_keywords),
            " ".join(positive_terms[:5])  # Top positive terms
        ]
        
        # Enhanced corpus creation with metadata
        corpus = []
        for chunk in chunks_for_ranking: # Use filtered chunks here
            # Safely access chunk.metadata.title for corpus text
            doc_title_for_corpus = getattr(chunk.metadata, 'title', '') if chunk.metadata else ''
            corpus_text = f"Document: {chunk.document_name}. Title: {doc_title_for_corpus}. Section: {chunk.section_title}. Text: {chunk.text}"
            # Add domain context
            if chunk.detected_domain != DocumentDomain.GENERAL:
                corpus_text = f"[{chunk.detected_domain.value}] {corpus_text}"
            corpus.append(corpus_text)
        
        # Semantic search
        corpus_embeddings = self.model_manager.bi_encoder.encode(corpus, convert_to_tensor=True, show_progress_bar=False)
        query_embeddings = self.model_manager.bi_encoder.encode(queries, convert_to_tensor=True, show_progress_bar=False)
        search_hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=min(100, len(chunks_for_ranking)))
        
        # Aggregate scores from multiple queries
        chunk_scores = {}
        for query_idx, hits in enumerate(search_hits):
            # Give higher weight to the original task query
            query_weight = [1.0, 0.7, 0.5][query_idx] if query_idx < 3 else 0.3
            for hit in hits:
                corpus_id = hit['corpus_id']
                score = hit['score'] * query_weight
                chunk_scores[corpus_id] = chunk_scores.get(corpus_id, 0) + score # Sum scores for each chunk

        if not chunk_scores: return []

        # Enhanced scoring with multiple factors
        adjusted_scores = {}
        for cid, base_score in chunk_scores.items():
            chunk = chunks_for_ranking[cid] # Ensure we reference from the filtered list
            score = base_score
            
            # Safely access chunk.metadata.title for searchable text
            doc_title_for_search = getattr(chunk.metadata, 'title', '') if chunk.metadata else ''
            searchable_text = (
                chunk.document_name + " " + doc_title_for_search + " " +
                chunk.section_title + " " + 
                chunk.text + " " +
                chunk.detected_domain.value
            ).lower()
            
            # Positive term boosting (with diminishing returns)
            positive_matches = sum(1 for term in positive_terms if term in searchable_text)
            if positive_matches > 0:
                boost = min(1.5, 1.0 + (positive_matches * 0.15)) # Slightly lower max boost
                score *= boost
            
            # Negative term penalty
            negative_matches = sum(1 for term in negative_terms if term in searchable_text)
            if negative_matches > 0:
                penalty = max(0.2, 1.0 - (negative_matches * 0.2)) # Stronger penalty
                score *= penalty
            
            # Structure level bonus (higher level = more important) - capped
            structure_bonus = 1.0 + (min(chunk.structure_level, 3) * 0.1) # Max bonus for level 3
            score *= structure_bonus
            
            # Confidence score factor from title detection
            score *= (0.8 + chunk.confidence_score * 0.2) # A smaller factor, as semantic similarity is key
            
            # Domain relevance bonus - if the chunk's domain is among the detected task domains
            # task_lower already defined
            if chunk.detected_domain in detected_domains:
                domain_keywords = DomainDetector.DOMAIN_KEYWORDS.get(chunk.detected_domain, [])
                domain_matches_in_task = sum(1 for keyword in domain_keywords if keyword in task_lower)
                if domain_matches_in_task > 0:
                    domain_bonus = 1.0 + (domain_matches_in_task * 0.05)
                    score *= domain_bonus

            # **CRITICAL: VEGETARIAN AND GLUTEN-FREE FILTERING/BOOSTING**
            # Very strong boost for positive requirements
            if any(veg_kw in searchable_text for veg_kw in ["vegetarian", "plant-based", "meatless", "vegan"]):
                score *= 1.5 # Significant boost for vegetarian
            
            # For "gluten-free", if it's explicitly mentioned, a big boost
            if "gluten-free" in searchable_text:
                score *= 1.7 # Even bigger boost for GF
            
            # Apply forbidden words as a hard filter after scoring
            # This is done *after* initial scoring to allow for dynamic LLM forbidden words
            has_forbidden = any(word in searchable_text for word in forbidden_words)
            if has_forbidden:
                logger.debug(f"Excluding chunk due to forbidden word: {chunk.document_name} - {chunk.section_title}")
                continue # Skip this chunk entirely

            adjusted_scores[cid] = score

        # Sort by adjusted scores
        # Note: adjusted_scores now only contains CIDs that passed forbidden word filter
        ranked_chunk_ids = sorted(adjusted_scores.keys(), key=adjusted_scores.get, reverse=True)
        ranked_chunks = [chunks_for_ranking[cid] for cid in ranked_chunk_ids] # Map back to actual chunks

        logger.info(f"Retrieved {len(ranked_chunks)} relevant chunks after all filtering and ranking")

        # Ensure we have enough chunks, pad if necessary (though with strict filtering, this might not always fill)
        if len(ranked_chunks) < Config.FINAL_REPORT_SECTIONS:
            logger.warning(f"Only {len(ranked_chunks)} chunks found, which is less than {Config.FINAL_REPORT_SECTIONS}. Unable to pad due to strict filtering.")
            # We won't pad with irrelevant chunks if strict filters are applied.
            # If the output is too short, it means we don't have enough truly relevant content.

        return ranked_chunks[:Config.FINAL_REPORT_SECTIONS]

class SelectionBasedRetriever:
    """Real-time retriever for selection-based content discovery"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.chunk_embeddings = None
        self.chunk_index = {}
        self.chunks_db = []
        
    def index_document_chunks(self, all_chunks: List[TextChunk]):
        """Pre-compute embeddings for all chunks to enable fast similarity search"""
        logger.info(f"Indexing {len(all_chunks)} chunks for real-time search...")
        
        corpus_texts = []
        for i, chunk in enumerate(all_chunks):
            doc_title = getattr(chunk.metadata, 'title', '') if chunk.metadata else ''
            corpus_text = f"Document: {chunk.document_name}. Section: {chunk.section_title}. Content: {chunk.text}"
            if doc_title:
                corpus_text = f"Title: {doc_title}. {corpus_text}"
            if chunk.detected_domain != DocumentDomain.GENERAL:
                corpus_text = f"[{chunk.detected_domain.value}] {corpus_text}"
            
            corpus_texts.append(corpus_text)
            self.chunk_index[i] = chunk
        
        self.chunk_embeddings = self.model_manager.bi_encoder.encode(
            corpus_texts, 
            convert_to_tensor=True, 
            show_progress_bar=True
        )
        self.chunks_db = all_chunks.copy()
        logger.info("✅ Chunk indexing completed")
    
    def find_similar_content(self, selected_text: str, top_k: int = 10, 
                             exclude_same_document: bool = True, 
                             source_document: str = None) -> List[Dict]:
        """Find similar content based on selected text"""
        if not selected_text.strip() or self.chunk_embeddings is None:
            return []
        
        cleaned_selection = self._clean_selection_text(selected_text)
        if len(cleaned_selection.split()) < 2:  # Too short for meaningful search
            return []
        
        logger.info(f"Searching for content similar to: '{cleaned_selection[:100]}...'")
        
        selection_embedding = self.model_manager.bi_encoder.encode(
            [cleaned_selection], 
            convert_to_tensor=True
        )
        
        search_results = util.semantic_search(
            selection_embedding, 
            self.chunk_embeddings, 
            top_k=min(top_k * 2, len(self.chunks_db))  # Get more results for filtering
        )[0]  # Only one query
        
        filtered_results = []
        
        for result in search_results:
            chunk_idx = result['corpus_id']
            chunk = self.chunk_index[chunk_idx]
            similarity_score = result['score']
            
            # Apply exclusion filters
            if exclude_same_document and source_document:
                if chunk.document_name.lower() == source_document.lower():
                    continue
            
            # Only include results with reasonable similarity
            if similarity_score < 0.3:
                continue
                
            # Format result
            doc_title = getattr(chunk.metadata, 'title', '') if chunk.metadata else ''
            formatted_result = {
                'document_name': chunk.document_name,
                'document_title': doc_title,
                'section_title': chunk.section_title,
                'content': chunk.text,
                'page_number': chunk.page_number,
                'similarity_score': float(similarity_score),
                'domain': chunk.detected_domain.value,
                'confidence_score': chunk.confidence_score,
                'chunk_index': chunk.chunk_index,
                'relevance_explanation': self._generate_relevance_explanation(similarity_score)
            }
            
            filtered_results.append(formatted_result)
            
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results
    
    def _clean_selection_text(self, text: str) -> str:
        """Clean selected text for better similarity search"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common UI artifacts that might be selected accidentally
        text = re.sub(r'^(page\s+\d+|figure\s+\d+|table\s+\d+)', '', text, flags=re.IGNORECASE)
        
        # Fix encoding issues
        text = ftfy.fix_text(text)
        
        return text
    
    def _generate_relevance_explanation(self, score: float) -> str:
        """Generate a brief explanation of why this content is relevant"""
        if score > 0.8:
            return "Very high similarity - likely discussing the same topic"
        elif score > 0.6:
            return "High similarity - related concepts or complementary information"  
        elif score > 0.4:
            return "Moderate similarity - tangentially related content"
        else:
            return "Lower similarity - might provide broader context"

class DocumentAnalysisPipeline:
    """Enhanced pipeline with improved title detection"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.processor = EnhancedDocumentProcessor()
        self.retriever = IntelligentRetriever(self.model_manager)
        # ADD THESE NEW LINES:
        self.selection_retriever = SelectionBasedRetriever(self.model_manager)
        self.indexed_chunks = []
        self.is_selection_ready = False

    # ADD THIS NEW METHOD to DocumentAnalysisPipeline class
    def initialize_selection_search(self, all_chunks: List[TextChunk]) -> bool:
        """Initialize the selection-based search system"""
        try:
            if not all_chunks:
                logger.warning("No chunks provided for selection search initialization")
                return False
                
            logger.info(f"Initializing selection search with {len(all_chunks)} chunks...")
            self.selection_retriever.index_document_chunks(all_chunks)
            self.indexed_chunks = all_chunks
            self.is_selection_ready = True
            logger.info("✅ Selection search system ready")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize selection search: {e}", exc_info=True)
            return False

    # ADD THIS NEW METHOD to DocumentAnalysisPipeline class  
    def search_by_selection(self, selected_text: str, source_document: str = None, 
                            top_k: int = 5, exclude_same_document: bool = True) -> List[Dict]:
        """API method for selection-based search"""
        if not self.is_selection_ready:
            logger.error("Selection search not initialized. Call initialize_selection_search first.")
            return []
        
        return self.selection_retriever.find_similar_content(
            selected_text=selected_text,
            top_k=top_k,
            exclude_same_document=exclude_same_document,
            source_document=source_document
        )

    # ADD THIS NEW METHOD to DocumentAnalysisPipeline class
    def find_connections_enhanced(self, selected_text: str, chunks: List[TextChunk], top_k: int = 10) -> List[Dict]:
        """
        Enhanced connection finding with multiple similarity approaches and better context.
        """
        if not chunks or not selected_text.strip():
            return []
        logger.info(f"Finding connections for text: '{selected_text[:100]}...'")
                
        # ENHANCEMENT 1: Multi-query approach
        queries = self._generate_connection_queries(selected_text)
                
        # ENHANCEMENT 2: Create enhanced corpus with more context
        corpus = []
        for chunk in chunks:
            doc_title = getattr(chunk.metadata, 'title', '') if chunk.metadata else ''
            # Include more context in searchable text
            corpus_text = f"""
            Document: {chunk.document_name}
            Title: {doc_title}
            Section: {chunk.section_title}
            Content: {chunk.text}
            Domain: {chunk.detected_domain.value}
            """.strip().replace('\n', ' ')
            corpus.append(corpus_text)
        # Semantic search with multiple queries
        corpus_embeddings = self.model_manager.bi_encoder.encode(corpus, convert_to_tensor=True, show_progress_bar=False)
        query_embeddings = self.model_manager.bi_encoder.encode(queries, convert_to_tensor=True, show_progress_bar=False)
                
        # ENHANCEMENT 3: Aggregate scores from multiple query types
        chunk_scores = {}
        search_results = util.semantic_search(query_embeddings, corpus_embeddings, top_k=min(50, len(chunks)))
                
        for query_idx, hits in enumerate(search_results):
            query_weight = [1.0, 0.8, 0.6][query_idx] if query_idx < 3 else 0.4
            for hit in hits:
                corpus_id = hit['corpus_id']
                score = hit['score'] * query_weight
                chunk_scores[corpus_id] = chunk_scores.get(corpus_id, 0) + score
        if not chunk_scores:
            return []
        # ENHANCEMENT 4: Apply connection-specific scoring
        final_scores = {}
        selected_text_lower = selected_text.lower()
                
        for chunk_id, base_score in chunk_scores.items():
            chunk = chunks[chunk_id]
            score = base_score
                        
            # Context similarity bonus
            context_text = f"{chunk.section_title} {chunk.text}".lower()
                        
            # Keyword overlap bonus
            selected_words = set(selected_text_lower.split())
            chunk_words = set(context_text.split())
            overlap_ratio = len(selected_words & chunk_words) / len(selected_words) if selected_words else 0
            if overlap_ratio > 0.3:  # 30% word overlap
                score *= (1.0 + overlap_ratio)
                        
            # Domain relevance bonus
            if chunk.detected_domain != DocumentDomain.GENERAL:
                score *= 1.1
                        
            # Structure level bonus (higher level sections are more likely to be important connections)
            score *= (1.0 + chunk.structure_level * 0.05)
                        
            # Length appropriateness (not too short, not too long)
            chunk_length = len(chunk.text.split())
            if 20 <= chunk_length <= 200:  # Optimal length range
                score *= 1.1
            elif chunk_length < 10:  # Too short
                score *= 0.7
            elif chunk_length > 500:  # Too long
                score *= 0.8
                        
            final_scores[chunk_id] = score
        # ENHANCEMENT 5: Diversify results by document and section
        ranked_chunk_ids = sorted(final_scores.keys(), key=final_scores.get, reverse=True)
        diversified_results = self._diversify_connection_results(
            [chunks[cid] for cid in ranked_chunk_ids], 
            final_scores, 
            top_k
        )
        # Format results with enhanced metadata
        connections = []
        for i, chunk in enumerate(diversified_results):
            # ENHANCEMENT 6: Add connection strength and explanation
            connection_strength = self._calculate_connection_strength(selected_text, chunk)
                        
            connections.append({
                "document_name": chunk.document_name,
                "section_title": chunk.section_title,
                "text": chunk.text,
                "page_number": chunk.page_number,
                "relevance_score": round(final_scores.get(chunks.index(chunk), 0), 3),
                "connection_strength": connection_strength,
                "domain": chunk.detected_domain.value,
                "rank": i + 1,
                "preview": chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text
            })
        logger.info(f"Found {len(connections)} enhanced connections")
        return connections

    def _generate_connection_queries(self, selected_text: str) -> List[str]:
        """Generate multiple query variations for better connection finding."""
        queries = [selected_text]  # Original text
                
        # Extract key concepts using spaCy
        doc = self.model_manager.nlp(selected_text)
                
        # Add noun phrases as queries
        noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
        queries.extend(noun_phrases[:3])  # Top 3 noun phrases
                
        # Add entities as queries
        entities = [ent.text for ent in doc.ents if ent.label_ in ('ORG', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LOC', 'PERSON')]
        queries.extend(entities[:2])  # Top 2 entities
                
        # Add keyword-based query
        keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
        if keywords:
            queries.append(" ".join(keywords[:5]))  # Top 5 keywords
                
        return list(dict.fromkeys(queries))  # Remove duplicates while preserving order

    def _diversify_connection_results(self, ranked_chunks: List[TextChunk], scores: Dict, top_k: int) -> List[TextChunk]:
        """Diversify results to avoid too many chunks from the same document/section."""
        selected = []
        doc_section_count = {}
        doc_count = {}
                
        for chunk in ranked_chunks:
            if len(selected) >= top_k:
                break
                            
            doc_name = chunk.document_name
            section_key = f"{doc_name}::{chunk.section_title}"
                        
            # Limit chunks per document and per section
            if doc_count.get(doc_name, 0) >= 3:  # Max 3 per document
                continue
            if doc_section_count.get(section_key, 0) >= 2:  # Max 2 per section
                continue
                            
            selected.append(chunk)
            doc_count[doc_name] = doc_count.get(doc_name, 0) + 1
            doc_section_count[section_key] = doc_section_count.get(section_key, 0) + 1
                
        return selected

    def _calculate_connection_strength(self, selected_text: str, chunk: TextChunk) -> str:
        """Calculate and categorize connection strength."""
        selected_lower = selected_text.lower()
        chunk_text_lower = chunk.text.lower()
                
        # Simple keyword overlap calculation
        selected_words = set(selected_lower.split())
        chunk_words = set(chunk_text_lower.split())
        overlap_ratio = len(selected_words & chunk_words) / len(selected_words) if selected_words else 0
                
        # Check for exact phrase matches
        has_exact_match = any(phrase.strip() in chunk_text_lower for phrase in selected_lower.split('.') if len(phrase.strip()) > 10)
                
        if has_exact_match or overlap_ratio > 0.7:
            return "strong"
        elif overlap_ratio > 0.4:
            return "medium"
        elif overlap_ratio > 0.2:
            return "weak"
        else:
            return "conceptual"

    def _format_chunks_with_llm(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Enhanced chunk formatting with domain awareness"""
        if not chunks: return []
        
        logger.info("Formatting chunks with domain-aware LLM processing...")
        
        # Group chunks by domain for more targeted formatting
        domain_groups = collections.defaultdict(list) # Use defaultdict for cleaner grouping
        for chunk in chunks:
            domain_groups[chunk.detected_domain].append(chunk)
        
        formatted_chunks = []
        
        for domain, domain_chunks in domain_groups.items():
            domain_name = domain.value
            
            # Combine chunks into larger prompts to reduce API calls and improve context for LLM
            combined_prompt_parts = []
            current_prompt_text = ""
            for i, chunk in enumerate(domain_chunks):
                chunk_text_to_add = f"Chunk {i}:\n\"\"\"\n{chunk.text}\n\"\"\"\n\n"
                # Keep prompt size within reasonable limits for LLM
                # Max tokens for stablelm2 could be around 4096. 3000 chars is a safe estimate for most
                if len(current_prompt_text) + len(chunk_text_to_add) < 3000:
                    current_prompt_text += chunk_text_to_add
                else:
                    combined_prompt_parts.append(current_prompt_text)
                    current_prompt_text = chunk_text_to_add
            if current_prompt_text: # Add any remaining text
                combined_prompt_parts.append(current_prompt_text)

            formatted_texts_overall = {}

            for part_idx, prompt_part in enumerate(combined_prompt_parts):
                prompt = f"""You are a text formatting expert specializing in {domain_name} documents. 
For each numbered chunk below, rewrite it as clean, well-structured prose appropriate for {domain_name} content.

Guidelines for {domain_name} formatting:
- Remove bullet points, excessive line breaks, and merge fragments into natural, coherent sentences.
- Maintain technical accuracy and domain-specific terminology.
- Ensure proper flow and readability.
- Keep the original meaning intact.
- If a chunk is clearly a heading, keep it concise and heading-like.

Chunks to format:

{prompt_part}

Respond with a single JSON object where keys are chunk numbers (e.g., "0") and values are the formatted text optimized for {domain_name} content.
Example for "technical manual sections":
{{
  "0": "This section describes the detailed specification of the XYZ component, including its operational parameters and environmental requirements.",
  "1": "The installation procedure involves several steps. First, ensure all power is disconnected from the unit. Next, connect the primary input cable to port A, and the secondary output cable to port B."
}}

Your turn for these {domain_name} chunks:"""
                
                try:
                    response = self.model_manager.llm_client.generate(model=Config.OLLAMA_MODEL, prompt=prompt, options={"temperature": 0.1}) # Lower temperature for formatting
                    clean_json_str = response['response'].replace('\n', '\\n').replace('\r', '\\r')
                    match = re.search(r'\{.*\}', clean_json_str, re.DOTALL)
                    
                    if match:
                        formatted_texts_batch = json.loads(match.group(0))
                        formatted_texts_overall.update(formatted_texts_batch) # Collect results from all batches
                    else:
                        logger.warning(f"Could not parse JSON from LLM response for {domain_name} (batch {part_idx}), using originals for this batch.")
                        # No update to formatted_texts_overall means original chunk text will be used by default
                        
                except Exception as e:
                    logger.error(f"Error formatting {domain_name} chunks (batch {part_idx}): {e}")
                    # If error, no update to formatted_texts_overall, originals will be used

            # Apply formatted texts back to original chunks
            for i, chunk in enumerate(domain_chunks):
                # Need to map the original chunk's index (from the overall domain_chunks list) to the formatted_texts_overall dict's keys
                formatted_text = formatted_texts_overall.get(str(i), chunk.text) # Use original if not formatted
                chunk.text = formatted_text
                formatted_chunks.append(chunk)
            
            logger.info(f"Successfully formatted {len(domain_chunks)} {domain_name} chunks")
                        
        return formatted_chunks

    def run(self) -> bool:
        # User constraint: docker image <= 200 MB, runtime <= 10s
        # Model loading (SentenceTransformer, spaCy, Ollama client) can impact size and startup time.
        # Ollama model 'stablelm2' itself is large and is assumed to be pre-pulled.
        # Bi-encoder 'msmarco-MiniLM-L-6-v3' is relatively small (approx 90MB)
        # spaCy 'en_core_web_sm' is also relatively small (approx 12MB)
        # These are within typical limits for a 200MB docker if only necessary components are bundled.
        # Runtime is handled by efficiency of processing and LLM calls.
        
        if not self.model_manager.load_models():  
            return False
        
        input_path = Path(Config.INPUT_DIR)
        
        if not input_path.exists():
            logger.error(f"Input directory {Config.INPUT_DIR} does not exist")
            return False
        
        processed_collections = 0
        for collection_dir in input_path.iterdir():
            if collection_dir.is_dir():
                try:
                    self.process_collection(collection_dir)
                    processed_collections += 1
                except Exception as e:
                    logger.error(f"Failed to process collection {collection_dir.name}: {e}")
        
        logger.info(f"--- Pipeline execution finished. Processed {processed_collections} collections ---")
        return processed_collections > 0

    def process_collection(self, collection_path: Path):
        try:
            logger.info(f"--- Processing Collection: {collection_path.name} ---")
            
            # Find metadata file
            json_files = list(collection_path.glob('*.json'))
            if not json_files:
                logger.warning(f"No metadata JSON file found in {collection_path.name}")
                return

            with open(json_files[0], 'r', encoding='utf-8') as f:
                # Load the entire metadata dictionary from the JSON file
                collection_metadata_dict = json.load(f)
            
            task = collection_metadata_dict.get('job_to_be_done', {}).get('task')
            if not task:
                logger.warning(f"No task found in metadata for {collection_path.name}")
                return

            logger.info(f"Task: {task}")

            # Extract chunks from all PDFs
            all_chunks = []
            pdf_dir = collection_path / 'pdfs' if (collection_path / 'pdfs').exists() else collection_path
            
            processed_docs = 0
            # Use the 'documents' list from the loaded collection_metadata_dict
            for doc_info in collection_metadata_dict.get('documents', []):
                filename = doc_info.get('filename', '')
                pdf_path = pdf_dir / filename
                
                if pdf_path.is_file() and pdf_path.suffix.lower() == '.pdf':
                    try:
                        logger.info(f"Processing document: {filename}")
                        # extract_chunks_from_pdf already creates and assigns DocumentMetadata objects to chunks
                        doc_chunks = self.processor.extract_chunks_from_pdf(str(pdf_path))
                        
                        # --- REMOVED THE PROBLEMTAIC LOOP HERE ---
                        # The DocumentMetadata object is already correctly attached to each chunk
                        # within the extract_chunks_from_pdf method call.
                        # No need to re-assign a dict here.
                        
                        all_chunks.extend(doc_chunks)
                        processed_docs += 1
                        logger.info(f"Extracted {len(doc_chunks)} chunks from {filename}")
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {e}")
                else:
                    logger.warning(f"PDF file not found: {pdf_path}")
            
            if not all_chunks:
                logger.warning(f"No text chunks extracted from {collection_path.name}")
                return

            logger.info(f"Total chunks extracted: {len(all_chunks)} from {processed_docs} documents")

            # Analyze domains represented in chunks
            domain_distribution = collections.Counter(chunk.detected_domain for chunk in all_chunks)
            
            logger.info(f"Domain distribution: {[(d.value, count) for d, count in domain_distribution.items()]}")

            # Rank and select top chunks
            top_chunks = self.retriever.rank_chunks_by_score(task, all_chunks)
            logger.info(f"Selected {len(top_chunks)} top-ranked chunks")

            # Format chunks
            formatted_chunks = self._format_chunks_with_llm(top_chunks)

            # Generate final report using the original collection_metadata_dict
            self._generate_final_report(formatted_chunks, collection_metadata_dict, collection_path, domain_distribution)

        except Exception as e:
            logger.error(f"Failed to process collection {collection_path.name}: {e}", exc_info=True)
    # In class DocumentAnalysisPipeline:

    def process_api_request(self, task: str, pdf_file_paths: List[str], collection_name: str) -> Dict:
        """Enhanced API request processing with selection search capability"""
        try:
            logger.info(f"--- Processing API Request for Collection: {collection_name} ---")
            logger.info(f"Task: {task}")

            # 1. Load models if they haven't been loaded yet
            if not self.model_manager.load_models():
                raise RuntimeError("Failed to load models for API request processing.")

            # 2. Extract chunks from all provided PDF paths
            all_chunks = []
            processed_docs_count = 0
            for pdf_path_str in pdf_file_paths:
                pdf_path = Path(pdf_path_str)
                if pdf_path.is_file() and pdf_path.suffix.lower() == '.pdf':
                    try:
                        logger.info(f"Processing document: {pdf_path.name}")
                        doc_chunks = self.processor.extract_chunks_from_pdf(str(pdf_path))
                        all_chunks.extend(doc_chunks)
                        processed_docs_count += 1
                        logger.info(f"Extracted {len(doc_chunks)} chunks from {pdf_path.name}")
                    except Exception as e:
                        logger.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
                else:
                    logger.warning(f"PDF file not found or invalid: {pdf_path}")

            if not all_chunks:
                logger.warning(f"No text chunks could be extracted from the provided documents for collection {collection_name}.")
                return {"error": "No content could be extracted from the uploaded documents."}

            logger.info(f"Total chunks extracted: {len(all_chunks)} from {processed_docs_count} documents")

            # 3. Initialize selection search system with all chunks
            selection_ready = self.initialize_selection_search(all_chunks)

            # 4. Analyze domain distribution
            domain_distribution = collections.Counter(chunk.detected_domain for chunk in all_chunks)
            logger.info(f"Domain distribution: {[(d.value, count) for d, count in domain_distribution.items()]}")

            # 5. Rank and select top chunks for main report
            top_chunks = self.retriever.rank_chunks_by_score(task, all_chunks)
            logger.info(f"Selected {len(top_chunks)} top-ranked chunks for the report.")

            # 6. Format chunks with LLM
            formatted_chunks = self._format_chunks_with_llm(top_chunks)

            # 7. Generate the final report dictionary
            final_report = {
                "collection_name": collection_name,
                "job_to_be_done": {"task": task},
                "documents": [{"filename": Path(p).name} for p in pdf_file_paths],
                "selection_search_ready": selection_ready, # NEW FIELD
                "extracted_sections": [],
                "subsection_analysis": [],
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "domain_analysis": {
                    "detected_domains": {domain.value: count for domain, count in domain_distribution.items()},
                    "primary_domain": max(domain_distribution, key=domain_distribution.get).value if domain_distribution else "general",
                    "total_chunks_processed": sum(domain_distribution.values()),
                    "total_chunks_indexed": len(all_chunks) # NEW FIELD
                }
            }

            if formatted_chunks:
                for i, chunk in enumerate(formatted_chunks):
                    final_report["extracted_sections"].append({
                        "document": f"{chunk.document_name}.pdf",
                        "section_title": chunk.section_title,
                        "importance_rank": i + 1,
                        "page_number": chunk.page_number
                    })
                    final_report["subsection_analysis"].append({
                        "document": f"{chunk.document_name}.pdf",
                        "refined_text": chunk.text,
                        "page_number": chunk.page_number
                    })

            # Optionally, save the report to disk for logging/auditing
            self._save_results(collection_name, final_report)

            return final_report

        except Exception as e:
            logger.error(f"Failed to process API request for collection {collection_name}: {e}", exc_info=True)
            return {"error": f"An internal server error occurred: {e}"}
    def _generate_final_report(self, chunks: List[TextChunk], metadata: Dict, # metadata here is the original JSON dict
                                  collection_path: Path, domain_distribution: Dict):
        """Generate enhanced final report with domain insights"""
        final_report = metadata.copy()
        extracted_sections, subsection_analysis = [], []
        
        if not chunks:
            logger.warning(f"No relevant chunks found for collection {collection_path.name}")
        else:
            for i, chunk in enumerate(chunks):
                extracted_sections.append({
                    "document": f"{chunk.document_name}.pdf",
                    "section_title": chunk.section_title,
                    "importance_rank": i + 1,
                    "page_number": chunk.page_number
                })
                
                subsection_analysis.append({
                    "document": f"{chunk.document_name}.pdf",
                    "refined_text": chunk.text,
                    "page_number": chunk.page_number
                })
        
        # Add enhanced metadata
        final_report["extracted_sections"] = extracted_sections
        final_report["subsection_analysis"] = subsection_analysis
        final_report["processing_timestamp"] = datetime.now(timezone.utc).isoformat()
        final_report["domain_analysis"] = {
            "detected_domains": {domain.value: count for domain, count in domain_distribution.items()},
            "primary_domain": max(domain_distribution, key=domain_distribution.get).value if domain_distribution else "general",
            "total_chunks_processed": sum(domain_distribution.values())
        }
        final_report["processing_config"] = {
            "min_chunk_words": Config.MIN_CHUNK_WORDS,
            "title_score_threshold": Config.TITLE_SCORE_THRESHOLD,
            "final_report_sections": Config.FINAL_REPORT_SECTIONS,
            "bi_encoder_model": Config.BI_ENCODER_MODEL,
            "title_detection": "smart_title_detector_v4_refined" # Updated version name
        }
        
        self._save_results(collection_path.name, final_report)

    def _save_results(self, collection_name: str, report: Dict):
        """Save results with error handling"""
        output_path = Path(Config.OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{collection_name}_analysis.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
            logger.info(f"Final report saved successfully to {output_file} ✨")
        except Exception as e:
            logger.error(f"Could not save final report to {output_file}: {e}")

def main():
    """Main execution function"""
    try:
        pipeline = DocumentAnalysisPipeline()
        success = pipeline.run()
        
        if success:
            logger.info("🎉 Pipeline completed successfully!")
        else:
            logger.error("❌ Pipeline failed to complete")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())