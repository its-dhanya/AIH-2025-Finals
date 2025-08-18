import fitz
import json
import os
import re
import argparse
import logging
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import List, Optional, Set, Tuple, Dict, Any
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import math
import sys
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration Constants (moved to top for clarity) ---
DEFAULT_WEIGHTS = {
    "relative_size": 35,
    "is_bold": 25,
    "text_case": 18,
    "is_centered": 20,
    "starts_with_number": 30,
    "preceding_whitespace": 22,
    "font_style_change": 18,
    "in_toc": 65,
    "position_weight": 25,
    "isolation_weight": 35,
    "font_uniqueness": 20,
    "indentation": 15,
    "line_spacing": 12,
    "capitalization_ratio": 10,
    "punctuation_penalty": -15,
    "length_penalty": -5,
    "semantic_consistency": 20,
}

POSTER_WEIGHTS = {
    "relative_size": 20,
    "is_bold": 25,
    "text_case": 15,
    "is_centered": 10,
    "starts_with_number": -10,
    "preceding_whitespace": 10,
    "font_style_change": 15,
    "in_toc": 0,
    "position_weight": 20,
    "isolation_weight": 50,
    "font_uniqueness": 25,
    "indentation": 5,
    "line_spacing": 5,
    "capitalization_ratio": 10,
    "punctuation_penalty": -15,
    "length_penalty": -10,
    "semantic_consistency": 15,
}

DEFAULT_THRESHOLDS = {
    "heading_confidence": 25,
    "semantic_similarity": 0.85,
    "min_heading_length": 4,
    "max_heading_length": 400,
    "max_word_count": 40,
    "title_min_score": 45,
    "vertical_spacing_threshold": 12,
    "min_font_size": 8,
    "max_heading_lines": 3,
    "capitalization_threshold": 0.7,
    "poster_font_variance_threshold": 50,
    "title_font_size_multiplier": 1.2,
    "title_position_min_y": 0.05,
    "title_position_max_y": 0.30,
    "title_is_document_type_penalty": 20,
}

DOCUMENT_TYPE_PATTERNS = {
    "academic": [
        r"\babstract\b",
        r"\bintroduction\b",
        r"\bmethodology\b",
        r"\bresults?\b",
        r"\bconclusions?\b",
        r"\breferences?\b",
        r"\bbibliography\b",
        r"\backnowledg",
        r"\bappendix\b",
        r"\bliterature review\b",
        r"\bdiscussion\b",
    ],
    "legal": [
        r"\bwhereas\b",
        r"\btherefore\b",
        r"\bpursuant to\b",
        r"\bnotwithstanding\b",
        r"\baffidavit\b",
        r"\bagreement\b",
        r"\bcontract\b",
        r"\blicense\b",
        r"\bterms and conditions\b",
        r"\bexhibit\b",
        r"\bschedule\b",
    ],
    "technical": [
        r"\bspecification\b",
        r"\brequirements?\b",
        r"\barchitecture\b",
        r"\bapi\b",
        r"\bdocumentation\b",
        r"\binstallation\b",
        r"\bconfiguration\b",
        r"\btroubleshooting\b",
        r"\bversion\b",
        r"\brelease notes?\b",
    ],
    "financial": [
        r"\bbalance sheet\b",
        r"\bincome statement\b",
        r"\bcash flow\b",
        r"\baudit\b",
        r"\bfinancial\b",
        r"\brevenue\b",
        r"\bexpenses?\b",
        r"\bassets?\b",
        r"\bliabilities\b",
        r"\bequity\b",
        r"\bprofit\b",
        r"\bloss\b",
    ],
    "manual": [
        r"\buser guide\b",
        r"\bmanual\b",
        r"\binstructions?\b",
        r"\bhow to\b",
        r"\bstep\s+\d+\b",
        r"\bprocedure\b",
        r"\boperating\b",
        r"\bmaintenance\b",
        r"\bsafety\b",
        r"\bwarning\b",
        r"\bcaution\b",
    ],
    "report": [
        r"\bexecutive summary\b",
        r"\boverview\b",
        r"\bfindings?\b",
        r"\brecommendations?\b",
        r"\banalysis\b",
        r"\bsurvey\b",
        r"\bstudy\b",
        r"\bassessment\b",
        r"\bevaluation\b",
        r"\bperformance\b",
    ],
}

NON_HEADING_STOP_WORDS = {
    "common": {
        "version",
        "date",
        "remarks",
        "identifier",
        "reference",
        "days",
        "designation",
        "pay",
        "syllabus",
        "copyright",
        "confidential",
        "page",
        "fig",
        "figure",
        "table",
        "whether",
        "signature",
        "tel",
        "fax",
        "email",
        "phone",
        "address",
        "contact",
        "home town",
        "name",
        "title",
        "position",
        "department",
        "organization",
        "company",
        "website",
        "please",
        "fill",
        "complete",
        "submit",
        "enter",
        "select",
        "check",
        "note",
        "yes",
        "no",
        "n/a",
        "not applicable",
        "optional",
        "required",
        "instructions",
        "guidelines",
        "comments",
        "description",
        "amount",
        "quantity",
        "total",
        "subtotal",
        "tax",
        "fee",
        "cost",
        "www",
        "http",
        "https",
        ".com",
        ".org",
        ".edu",
        ".gov",
        "inc",
        "grant",
        "advance",
        "concession",
        "fare",
        "visit",
        "form",
        "application",
        "print",
        "clear",
        "reset",
        "save",
        "continue",
        "next",
        "previous",
        "back",
        "home",
        "availed",
        "servant",
        "route",
    },
    "academic": {
        "corresponding author",
        "keywords",
        "received",
        "accepted",
        "published",
        "doi",
        "issn",
        "isbn",
        "vol",
        "pp",
        "et al",
        "ibid",
    },
    "legal": {
        "plaintiff",
        "defendant",
        "court",
        "judge",
        "attorney",
        "counsel",
        "case no",
        "docket",
        "filed",
        "served",
    },
    "general_section_titles": {
        "overview",
        "introduction",
        "table of contents",
        "contents",
        "abstract",
        "summary",
        "preface",
        "acknowledgements",
        "chapter",
        "section",
        "appendix",
        "references",
        "bibliography",
        "glossary",
        "index",
    }
}

COMPILED_FORM_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.MULTILINE)
    for p in [
        r".*:\s*_{3,}\s*$",
        r"^\s*\[\s*\]",
        r"^\s*\(\s*\)",
        r"^\s*_{10,}$",
        r"\b(signature|date)\b\s*:\s*$",
        r"circle one",
    ]
]

HEADING_PATTERNS = {
    "numbered_decimal": re.compile(r"^(\d+(?:\.\d+)*\.?)\s+(.+)$"),
    "numbered_simple": re.compile(r"^(\d+\.?)\s+(.+)$"),
    "roman_numerals": re.compile(r"^([IVXLCDM]+\.?)\s+(.+)$", re.IGNORECASE),
    "letters": re.compile(r"^([A-Z]\.?)\s+(.+)$"),
    "bullets": re.compile(r"^([•·▪▫‣⁃]\s*)(.+)$"),
    "chapter": re.compile(r"^(chapter\s+\d+|ch\.\s*\d+)\s*:?\s*(.*)$", re.IGNORECASE),
    "section": re.compile(
        r"^(section\s+\d+(?:\.\d+)*|sec\.\s*\d+(?:\.\d+)*)\s*:?\s*(.*)$", re.IGNORECASE
    ),
    "part": re.compile(r"^(part\s+[IVXLCDM]+|part\s+\d+)\s*:?\s*(.*)$", re.IGNORECASE),
    "appendix": re.compile(
        r"^(appendix\s+[A-Z]|appendix\s+\d+)\s*:?\s*(.*)$", re.IGNORECASE
    ),
}

# --- Data Classes ---
@dataclass
class Block:
    text: str
    page_num: int
    bbox: Tuple[float, float, float, float]
    size: float
    font: str
    is_bold: bool
    is_italic: bool = False
    color: Tuple[float, float, float] = (0, 0, 0)
    line_height: float = 0.0
    score: float = 0.0
    font_family: str = ""
    indentation: float = 0.0
    line_count: int = 1
    word_count: int = 0
    char_count: int = 0
    capitalization_ratio: float = 0.0
    punctuation_ratio: float = 0.0
    whitespace_before: float = 0.0
    whitespace_after: float = 0.0
    level: int = 0

    def __post_init__(self):
        self.font_family = self.extract_font_family()
        self.word_count = len(self.text.split())
        self.char_count = len(self.text)
        self.calculate_text_metrics()

    def extract_font_family(self) -> str:
        if not self.font:
            return ""
        font_clean = re.sub(
            r"[-+](Bold|Italic|Regular|Light|Medium|Heavy|Black|Thin|ExtraLight|SemiBold|ExtraBold).*$",
            "",
            self.font,
            flags=re.IGNORECASE,
        )
        font_clean = re.sub(r"(MT|PS|TT|OT)$", "", font_clean, flags=re.IGNORECASE)
        return (
            font_clean.split("-")[0].split(",")[0].strip()
            if any(sep in font_clean for sep in ["-", ","])
            else font_clean.strip()
        )

    def calculate_text_metrics(self):
        if not self.text:
            return
        alpha_chars = [c for c in self.text if c.isalpha()]
        if alpha_chars:
            self.capitalization_ratio = sum(
                1 for c in alpha_chars if c.isupper()
            ) / len(alpha_chars)
        if self.text:
            self.punctuation_ratio = sum(
                1 for c in self.text if c in ".,;:!?()[]{}\"'-"
            ) / len(self.text)


@dataclass
class DocumentAnalysis:
    font_sizes: Counter
    font_families: Counter
    common_fonts: Set[str]
    page_dimensions: Tuple[float, float]
    document_type: str = "unknown"
    is_form: bool = False
    is_poster: bool = False
    is_report: bool = False
    is_academic: bool = False
    is_legal: bool = False
    is_technical: bool = False
    is_manual: bool = False
    is_financial: bool = False
    language: str = "en"
    reading_order: str = "ltr"
    column_count: int = 1
    avg_line_spacing: float = 0.0
    text_density: float = 0.0


# --- Utility Functions ---
def normalize_unicode(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u00a0", " ")
    return text


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_unicode(text)
    text = re.sub(r"[\t\n\r\f\v\u2000-\u200B\u2028\u2029]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    return text.strip()


def detect_language(text: str) -> str:
    """Simple language detection based on character patterns"""
    if not text:
        return "en"
    latin_count = sum(1 for c in text if "\u0000" <= c <= "\u024f")
    arabic_count = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    chinese_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    cyrillic_count = sum(1 for c in text if "\u0400" <= c <= "\u04ff")

    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "en"

    if arabic_count / total_alpha > 0.3:
        return "ar"
    elif chinese_count / total_alpha > 0.3:
        return "zh"
    elif cyrillic_count / total_alpha > 0.3:
        return "ru"
    else:
        return "en"

def is_garbled_ocr(text: str) -> bool:
    """Enhanced OCR garbling detection with better heuristics"""
    if len(text) < 3:
        return False
    if len(text) > 15 and re.search(r"(.{2,10})\1{3,}", text):
        return True
    alphanum_chars = sum(1 for char in text if char.isalnum() or char.isspace())
    if len(text) > 10 and (alphanum_chars / len(text)) < 0.4:
        return True
    impossible_clusters = ["qx", "qz", "jx", "jz", "vx", "vz", "wx", "wz"]
    text_lower = text.lower()
    if any(cluster in text_lower for cluster in impossible_clusters):
        return True
    punct_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if len(text) > 5 and punct_count / len(text) > 0.4:
        return True
    if len(text) > 10:
        case_changes = sum(
            1
            for i in range(1, len(text))
            if text[i].isalpha()
            and text[i - 1].isalpha()
            and text[i].isupper() != text[i - 1].isupper()
        )
        if case_changes > len(text) * 0.3:
            return True

    return False

def analyze_document_type(
    doc: fitz.Document, thresholds: Dict
) -> DocumentAnalysis:
    """Enhanced document analysis with better type and poster detection"""
    font_sizes = Counter()
    font_families = Counter()
    all_text = ""
    page_dims = []

    type_scores = {doc_type: 0 for doc_type in DOCUMENT_TYPE_PATTERNS}

    try:
        for page_num in range(min(10, doc.page_count)):
            try:
                page = doc.load_page(page_num)
                page_dims.append((page.rect.width, page.rect.height))
                page_text = page.get_text().lower()
                all_text += page_text + " "

                for doc_type, patterns in DOCUMENT_TYPE_PATTERNS.items():
                    for pattern in patterns:
                        type_scores[doc_type] += len(
                            re.findall(pattern, page_text, re.IGNORECASE)
                        )

                blocks_dict = page.get_text("dict")
                if blocks_dict and "blocks" in blocks_dict:
                    for block in blocks_dict["blocks"]:
                        if block.get("type") == 0:
                            for line in block.get("lines", []):
                                for span in line.get("spans", []):
                                    font_sizes[round(span["size"])] += 1
                                    font_families[span["font"]] += 1
            except Exception as e:
                logging.warning(f"Failed to process page {page_num}: {e}")
    except Exception as e:
        logging.warning(f"Error in document analysis: {e}")

    dominant_type = (
        max(type_scores, key=type_scores.get)
        if type_scores and type_scores[max(type_scores, key=type_scores.get)] > 0
        else "unknown"
    )
    language = detect_language(all_text[:1000])
    reading_order = "rtl" if language in ["ar", "he"] else "ltr"
    avg_dims = (
        (
            sum(w for w, h in page_dims) / len(page_dims),
            sum(h for w, h in page_dims) / len(page_dims),
        )
        if page_dims
        else (612, 792)
    )

    font_size_list = list(font_sizes.keys())
    font_size_variance = np.var(font_size_list) if len(font_size_list) > 1 else 0
    is_poster = (
        doc.page_count <= 2
        and font_size_variance > thresholds["poster_font_variance_threshold"]
    ) or (
        avg_dims[0] > avg_dims[1] and doc.page_count <= 3
    )

    analysis = DocumentAnalysis(
        font_sizes=font_sizes,
        font_families=font_families,
        common_fonts=set(dict(font_families.most_common(5)).keys()),
        page_dimensions=avg_dims,
        document_type=dominant_type,
        language=language,
        reading_order=reading_order,
        is_academic=dominant_type == "academic" or type_scores["academic"] > 5,
        is_legal=dominant_type == "legal" or type_scores["legal"] > 3,
        is_technical=dominant_type == "technical" or type_scores["technical"] > 4,
        is_manual=dominant_type == "manual" or type_scores["manual"] > 4,
        is_financial=dominant_type == "financial" or type_scores["financial"] > 3,
        is_report=dominant_type == "report" or type_scores["report"] > 4,
        is_form="form" in all_text.lower() or "application" in all_text.lower(),
        is_poster=is_poster,
    )
    return analysis


def get_text_case_advanced(text: str) -> str:
    """Advanced text case detection with better heuristics"""
    if not text or len(text) <= 1:
        return "SENTENCE"
    alpha_text = "".join(c for c in text if c.isalpha())
    if not alpha_text:
        return "SENTENCE"
    upper_count = sum(1 for c in alpha_text if c.isupper())
    lower_count = sum(1 for c in alpha_text if c.islower())
    total_alpha = len(alpha_text)
    upper_ratio = upper_count / total_alpha
    if upper_ratio >= 0.95:
        return "UPPER"
    words = text.split()
    if len(words) > 1:
        capitalized_words = sum(1 for word in words if word and word[0].isupper())
        if capitalized_words >= len(words) * 0.8 and upper_ratio < 0.7:
            return "TITLE"
    if text[0].isupper() and upper_ratio < 0.3:
        return "SENTENCE"
    return "MIXED"


def is_likely_sentence_advanced(text: str) -> bool:
    """Enhanced sentence detection with better linguistic heuristics"""
    text = text.strip()
    if not text:
        return False

    word_count = len(text.split())
    if ":" in text[:25] and word_count < 5:
        return False
    if text.endswith((".", "?", "!")) and word_count > 8:
        return True
    legal_indicators = [
        "pursuant to",
        "notwithstanding",
        "whereas",
        "therefore",
        "hereby",
    ]
    if any(indicator in text.lower() for indicator in legal_indicators):
        return True
    academic_indicators = [
        "this study",
        "our research",
        "we found",
        "the results show",
        "in conclusion",
    ]
    if any(indicator in text.lower() for indicator in academic_indicators):
        return True

    tech_indicators = ["this section", "the following", "to configure", "note that"]
    if any(indicator in text.lower() for indicator in tech_indicators):
        return True

    if word_count > 12:
        connecting_words = [
            "and",
            "but",
            "or",
            "however",
            "therefore",
            "because",
            "although",
            "since",
            "while",
        ]
        if any(word in text.lower().split() for word in connecting_words):
            return True

    if re.match(r"^\(\w\)\s", text.lower()) and word_count > 6:
        return True

    if any(symbol in text for symbol in ["©", "®", "™"]) and word_count > 4:
        return True
    return False

def detect_and_parse_toc_advanced(
    doc: fitz.Document,
) -> Tuple[Set[int], Dict[str, int]]:
    """Advanced TOC detection with improved pattern recognition"""
    toc_pages, toc_entries = set(), {}
    max_toc_page = min(25, int(doc.page_count * 0.3))
    toc_patterns = [
        re.compile(r"^(.*?)\s*[._\s]{4,}\s*(\d+)$"),
        re.compile(r"^(.*?)\s{3,}(\d+)$"),
        re.compile(r"^(\d+\.?\d*\.?\s+.*?)\s{2,}(\d+)$"),
        re.compile(r"^([A-Z\s]+.*?)\s{3,}(\d+)$"),
        re.compile(r"^(Chapter\s+\d+.*?)\s+(\d+)$", re.IGNORECASE),
        re.compile(r"^(Section\s+\d+.*?)\s+(\d+)$", re.IGNORECASE),
        re.compile(r"^(Appendix\s+[A-Z].*?)\s+(\d+)$", re.IGNORECASE),
    ]

    try:
        for page_num in range(max_toc_page):
            try:
                page = doc.load_page(page_num)
                text = page.get_text().lower()
                blocks = page.get_text("blocks")

                toc_indicators = [
                    "table of contents" in text or "contents" in text,
                    len(re.findall(r"\.{4,}", text)) > 3,
                    len(re.findall(r"\d+\s*$", text, re.MULTILINE)) > 5,
                    len(re.findall(r"chapter\s+\d+", text)) > 2,
                    len(re.findall(r"section\s+\d+", text)) > 3,
                    page_num < 5
                    and "index" not in text,
                ]

                toc_score = sum(toc_indicators)

                if toc_score >= 2:
                    toc_pages.add(page_num)

                    for block in blocks:
                        try:
                            line_text = clean_text(str(block[4]))
                            if not line_text or len(line_text) < 5:
                                continue

                            for pattern in toc_patterns:
                                match = pattern.match(line_text)
                                if match:
                                    entry_text = match.group(1).strip()
                                    entry_text = re.sub(
                                        r"^[\d\.\sA-Za-z\)]+\s*", "", entry_text
                                    ).strip()
                                    entry_text = re.sub(r"\s+", " ", entry_text)

                                    if len(entry_text) > 3 and not entry_text.isdigit():
                                        try:
                                            page_ref = int(match.group(2))
                                            if 1 <= page_ref <= doc.page_count:
                                                toc_entries[entry_text] = page_ref
                                        except (ValueError, IndexError):
                                            continue
                                    break

                        except Exception as e:
                            logging.debug(f"Error processing TOC block: {e}")

            except Exception as e:
                logging.warning(f"Error processing page {page_num} for TOC: {e}")

    except Exception as e:
        logging.warning(f"Error in TOC detection: {e}")

    return toc_pages, toc_entries


def identify_headers_footers_advanced(doc: fitz.Document) -> Set[str]:
    """Advanced header/footer identification with better pattern recognition"""
    text_counter = Counter()
    position_counter = defaultdict(list)
    margin = 0.12
    try:
        sample_pages = min(doc.page_count, 15)

        for page_num in range(sample_pages):
            try:
                page = doc.load_page(page_num)
                h = page.rect.height
                blocks = page.get_text("blocks") or []

                for block in blocks:
                    try:
                        text = clean_text(str(block[4]))
                        if not text or len(text) < 3:
                            continue

                        y0, y1 = block[1], block[3]

                        in_header = y0 < h * margin
                        in_footer = y1 > h * (1 - margin)

                        if in_header or in_footer:
                            normalized_text = re.sub(
                                r"[^\w\s]", "", text.lower()
                            ).strip()
                            normalized_text = re.sub(r"\s+", " ", normalized_text)

                            if (
                                not re.match(r"^\d+$", normalized_text)
                                and not re.match(
                                    r"^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$", text
                                )
                                and len(normalized_text) > 2
                            ):

                                text_counter[normalized_text] += 1
                                position_type = "header" if in_header else "footer"
                                position_counter[normalized_text].append(
                                    (page_num, y0, position_type)
                                )

                    except Exception as e:
                        logging.debug(f"Error processing header/footer block: {e}")

            except Exception as e:
                logging.warning(
                    f"Error processing page {page_num} for headers/footers: {e}"
                )

    except Exception as e:
        logging.warning(f"Error in header/footer identification: {e}")

    min_occurrences = max(2, sample_pages // 4)
    repeating_texts = set()

    for norm_text, count in text_counter.items():
        if count >= min_occurrences:
            positions = position_counter[norm_text]

            y_positions = [pos[1] for pos in positions]
            position_types = [pos[2] for pos in positions]

            if len(set(round(y, -1) for y in y_positions)) <= 3:
                if len(set(position_types)) <= 2:
                    repeating_texts.add(norm_text)

    return repeating_texts


def get_page_stats(page: fitz.Page) -> Dict[str, float]:
    """Enhanced page statistics with error handling"""
    sizes, fonts = [], []
    try:
        blocks_dict = page.get_text("dict")
        if not blocks_dict or "blocks" not in blocks_dict:
            return {"mode_size": 12, "mean_size": 12, "max_size": 12}
        for block in blocks_dict["blocks"]:
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        sizes.append(round(span["size"]))
                        fonts.append(span["font"])
    except Exception as e:
        logging.debug(f"Error getting page stats: {e}")
    if not sizes:
        return {"mode_size": 12, "mean_size": 12, "max_size": 12}
    size_counter = Counter(sizes)
    mode_size = size_counter.most_common(1)[0][0]
    mean_size = sum(sizes) / len(sizes)
    max_size = max(sizes)
    return {
        "mode_size": mode_size,
        "mean_size": mean_size,
        "max_size": max_size,
        "size_variance": np.var(sizes) if len(sizes) > 1 else 0,
    }


def extract_headings_from_pdf_advanced(
    pdf_path: str,
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    confidence_threshold: Optional[float] = None,
    apply_semantic_filtering: bool = True,
) -> Dict[str, Any]:
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()
    if confidence_threshold is None:
        confidence_threshold = thresholds["heading_confidence"]

    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            return {"error": "PDF document has no pages"}

        doc_analysis = analyze_document_type(doc, thresholds)
        toc_pages, toc_entries = detect_and_parse_toc_advanced(doc)
        headers_footers = identify_headers_footers_advanced(doc)

        all_blocks = []
        page_block_map = defaultdict(list)

        semantic_model = None
        if apply_semantic_filtering:
            try:
                # Ensure model is loaded from the environment variable specified in Dockerfile
                # SENTENCE_TRANSFORMERS_HOME will be /app/models
                MODEL_DIR = Path(__file__).parent / "all-MiniLM-L6-v2-local"
                semantic_model = SentenceTransformer(str(MODEL_DIR))
            except Exception as e:
                logging.warning(f"Could not load semantic model: {e}")
                apply_semantic_filtering = False

        for page_num in range(doc.page_count):
            try:
                page = doc.load_page(page_num)
                blocks_dict = page.get_text("dict")
                if not blocks_dict or "blocks" not in blocks_dict:
                    continue
                
                prev_block_bbox = None
                for block_dict in blocks_dict["blocks"]:
                    if block_dict.get("type") != 0:
                        continue
                    
                    spans = []
                    for line in block_dict.get("lines", []):
                        spans.extend(line.get("spans", []))

                    if not spans:
                        continue

                    # Consolidate text from spans
                    clean_full_text = clean_text("".join(s["text"] for s in spans))

                    if (
                        not clean_full_text
                        or len(clean_full_text) < 2
                        or is_garbled_ocr(clean_full_text)
                    ):
                        continue

                    # Determine dominant font and size for the block
                    dominant_font = Counter(s["font"] for s in spans).most_common(1)[0][0]
                    dominant_size = Counter(s["size"] for s in spans).most_common(1)[0][0]
                    is_bold = any(s["flags"] & 16 for s in spans)
                    is_italic = any(s["flags"] & 2 for s in spans) # Although not directly used for scoring, good to capture

                    text_block = Block(
                        text=clean_full_text,
                        page_num=page_num,
                        bbox=tuple(block_dict["bbox"]),
                        size=float(dominant_size),
                        font=dominant_font,
                        is_bold=is_bold,
                        is_italic=is_italic,
                        indentation=float(block_dict["bbox"][0]),
                    )
                    if prev_block_bbox:
                        text_block.whitespace_before = float(
                            block_dict["bbox"][1] - prev_block_bbox[3]
                        )
                    all_blocks.append(text_block)
                    page_block_map[page_num].append(text_block)
                    prev_block_bbox = block_dict["bbox"]
            except Exception as e:
                logging.warning(f"Error processing page {page_num}: {e}")

        if not all_blocks:
            return {"error": "No text blocks found in PDF"}
        
        title = extract_document_title(all_blocks, doc_analysis, thresholds)
        
        scored_blocks = []

        for block in all_blocks:
            if block.text == title and block.page_num <= 1:
                continue
            is_pre_cleaned = False
            if "●" in block.text and doc_analysis.is_poster:
                first_part = block.text.split("●", 1)[0].strip()
                if 1 < len(first_part.split()) < 5:
                    block.text = first_part
                    is_pre_cleaned = True

            if should_skip_block(block, doc_analysis, headers_footers, thresholds):
                continue

            # Load page_stats on demand to avoid processing all pages if only first few are needed
            # For each block, get page stats from its respective page
            page_stats = get_page_stats(doc.load_page(block.page_num))
            score = calculate_heading_score(
                block,
                page_block_map[block.page_num],
                doc_analysis,
                toc_entries,
                weights,
                thresholds,
                page_stats,
            )

            if is_pre_cleaned:
                score += 50

            if score >= confidence_threshold:
                block.score = score
                scored_blocks.append(block)

        scored_blocks.sort(key=lambda x: x.score, reverse=True)
        if apply_semantic_filtering and semantic_model and len(scored_blocks) > 1:
            scored_blocks = apply_semantic_consistency_filter(
                scored_blocks, semantic_model, thresholds, weights
            )
        headings = assign_heading_levels(scored_blocks, doc_analysis)
        results = {
            "headings": [
                {
                    "text": h.text,
                    "page": h.page_num,
                    "level": getattr(h, "level", 1),
                    "confidence_score": h.score,
                    "font_size": h.size,
                    "font_family": h.font_family,
                    "is_bold": h.is_bold,
                    "bbox": h.bbox,
                }
                for h in headings
            ],
            "title": title,
            "document_info": {
                "total_pages": doc.page_count,
                "document_type": doc_analysis.document_type,
                "language": doc_analysis.language,
                "has_toc": len(toc_pages) > 0,
                "total_blocks_analyzed": len(all_blocks),
                "headings_found": len(headings),
            },
            "extraction_metadata": {
                "confidence_threshold": confidence_threshold,
                "weights_used": weights,
                "thresholds_used": thresholds,
                "semantic_filtering_applied": apply_semantic_filtering,
            },
        }
        doc.close()
        return results

    except Exception as e:
        logging.error(f"Error extracting headings from {pdf_path}: {e}")
        return {"error": f"Failed to process PDF: {str(e)}"}


def should_skip_block(
    block: Block,
    doc_analysis: DocumentAnalysis,
    headers_footers: Set[str],
    thresholds: Dict[str, float],
) -> bool:
    text = block.text.strip()
    text_lower = text.lower()

    if not text or len(text) < thresholds["min_heading_length"]:
        return True
    if len(text) > thresholds["max_heading_length"]:
        return True

    if doc_analysis.is_form:
        if block.word_count <= 2 and block.size < 14 and not block.is_bold:
            return True
        for pattern in COMPILED_FORM_PATTERNS:
            if pattern.search(text):
                return True
        alpha_chars = sum(1 for char in text if char.isalpha())
        if len(text) > 0 and (alpha_chars / len(text)) < 0.4:
            return True

    normalized_text = re.sub(r"[^\w\s]", "", text_lower).strip()
    if normalized_text in headers_footers:
        return True

    stop_words = NON_HEADING_STOP_WORDS.get(
        "common", set()
    ) | NON_HEADING_STOP_WORDS.get(doc_analysis.document_type, set())
    if any(word in text_lower for word in stop_words):
        return True

    if is_likely_sentence_advanced(text):
        return True

    return False


def _calculate_isolation(current_block: Block, page_blocks: List[Block]) -> float:
    """Calculates the minimum distance from a block to any other block on the page."""
    min_dist = float("inf")
    if len(page_blocks) <= 1:
        return 50.0

    cb = current_block.bbox
    for other_block in page_blocks:
        if other_block is current_block:
            continue
        ob = other_block.bbox

        dx = max(0, ob[0] - cb[2], cb[0] - ob[2])
        dy = max(0, ob[1] - cb[3], cb[1] - ob[3])
        dist = math.sqrt(dx**2 + dy**2)

        if dist < min_dist:
            min_dist = dist

    return min_dist


def calculate_heading_score(
    block: Block,
    page_blocks: List[Block],
    doc_analysis: DocumentAnalysis,
    toc_entries: Dict[str, int],
    weights: Dict[str, float],
    thresholds: Dict[str, float],
    page_stats: Dict[str, float],
) -> float:
    """Calculate comprehensive heading score, with special logic for posters."""

    active_weights = POSTER_WEIGHTS if doc_analysis.is_poster else weights
    score = 0.0

    if page_stats["mode_size"] > 0:
        size_ratio = block.size / page_stats["mode_size"]
        score += active_weights.get("relative_size", 35) * min(size_ratio, 2.5)

    if block.is_bold:
        score += active_weights.get("is_bold", 25)

    text_case = get_text_case_advanced(block.text)
    case_bonus = {
        "UPPER": active_weights.get("text_case", 15),
        "TITLE": active_weights.get("text_case", 15) * 0.8,
    }
    score += case_bonus.get(text_case, 0)

    page_width, page_height = doc_analysis.page_dimensions
    is_centered = (
        abs(block.bbox[0] + (block.bbox[2] - block.bbox[0]) / 2 - page_width / 2)
        < page_width * 0.15
    )
    if is_centered:
        score += active_weights.get("is_centered", 20)

    if block.bbox[1] < page_height * 0.5:
        score += active_weights.get("position_weight", 25)

    if re.match(r"^\d+\.?\s+", block.text):
        score += active_weights.get("starts_with_number", 30)

    text_clean = re.sub(r"[^\w\s]", "", block.text.lower()).strip()
    if text_clean in toc_entries:
        score += active_weights.get("in_toc", 65)

    if block.whitespace_before > thresholds.get("vertical_spacing_threshold", 8):
        score += active_weights.get("preceding_whitespace", 22)

    if active_weights.get("isolation_weight", 0) > 0:
        isolation_dist = _calculate_isolation(block, page_blocks)
        isolation_bonus = min(isolation_dist / 36.0, 1.0)
        score += active_weights.get("isolation_weight") * isolation_bonus

    if len(block.text) < 5:
        score += active_weights.get("length_penalty", -5) * 2
    elif len(block.text) > 100:
        score += active_weights.get("length_penalty", -5) * (len(block.text) / 100)

    stripped_text = block.text.strip()
    if stripped_text.endswith((".", "?", "!")) and block.word_count > 8:
        score += active_weights.get("punctuation_penalty", -15)
    elif stripped_text.endswith(":"):
        score += 10

    if block.capitalization_ratio > thresholds.get("capitalization_threshold", 0.7):
        score += active_weights.get("capitalization_ratio", 10)

    return max(0, score)


def apply_semantic_consistency_filter(
    blocks: List[Block],
    model: Any,
    thresholds: Dict[str, float],
    weights: Dict[str, float],
) -> List[Block]:
    if len(blocks) <= 2:
        return blocks
    try:
        texts = [block.text for block in blocks]
        embeddings = model.encode(texts)
        similarities = cosine_similarity(embeddings)
        avg_similarities = []
        for i in range(len(blocks)):
            other_sims = [similarities[i][j] for j in range(len(blocks)) if i != j]
            avg_sim = sum(other_sims) / len(other_sims) if other_sims else 0
            avg_similarities.append(avg_sim)
        filtered_blocks = []
        for block, avg_sim in zip(blocks, avg_similarities):
            if block.score > 60 or avg_sim >= thresholds.get(
                "semantic_similarity", 0.85
            ):
                filtered_blocks.append(block)
            elif avg_sim >= thresholds.get("semantic_similarity", 0.85) * 0.8:
                block.score += weights.get("semantic_consistency", 20) * avg_sim
                filtered_blocks.append(block)
        return filtered_blocks
    except Exception as e:
        logging.warning(f"Semantic filtering failed: {e}")
        return blocks


def assign_heading_levels(
    blocks: List[Block], doc_analysis: DocumentAnalysis
) -> List[Block]:
    """
    Assign hierarchical levels to heading blocks
    """
    if not blocks:
        return []

    blocks_by_position = sorted(blocks, key=lambda b: (b.page_num, b.bbox[1]))

    font_groups = defaultdict(list)
    for block in blocks_by_position:
        font_key = (round(block.size), block.is_bold, block.font_family)
        font_groups[font_key].append(block)

    sorted_font_groups = sorted(
        font_groups.items(),
        key=lambda x: (x[0][0], -len(x[1])),
        reverse=True,
    )

    level_map = {}
    current_level = 1

    for font_key, group_blocks in sorted_font_groups:
        if current_level <= 6:
            level_map[font_key] = current_level
            current_level += 1
        else:
            level_map[font_key] = 6

    for block in blocks_by_position:
        font_key = (round(block.size), block.is_bold, block.font_family)
        block.level = level_map.get(font_key, 1)

    return blocks_by_position


def extract_document_title(blocks: List[Block], doc_analysis: DocumentAnalysis, thresholds: Dict) -> str:
    """
    Extract document title from blocks with improved heuristics.
    Prioritizes blocks that are most visually distinct AND are not common section titles.
    """
    first_page_blocks = [b for b in blocks if b.page_num == 0]
    if not first_page_blocks:
        return "Untitled Document"

    first_page_sizes = [b.size for b in first_page_blocks]
    if not first_page_sizes:
        return "Untitled Document"
    body_text_size_mode = Counter(first_page_sizes).most_common(1)[0][0]

    title_candidates = []

    common_section_titles = set()
    for category in NON_HEADING_STOP_WORDS.values():
        common_section_titles.update(category)
    common_section_titles.update(set(doc_analysis.document_type.split()))

    for block in first_page_blocks:
        text_lower = block.text.lower().strip()
        title_score = 0.0

        if block.size > body_text_size_mode * thresholds.get("title_font_size_multiplier", 1.2):
            title_score += (block.size / body_text_size_mode) * 20
        elif block.size > body_text_size_mode:
            title_score += 10

        if block.is_bold:
            title_score += 25

        page_height = doc_analysis.page_dimensions[1]
        y_ratio = block.bbox[1] / page_height
        if thresholds["title_position_min_y"] <= y_ratio <= thresholds["title_position_max_y"]:
            title_score += 30
        elif y_ratio < thresholds["title_position_min_y"]:
             title_score += 10
        elif y_ratio > thresholds["title_position_max_y"]:
            title_score -= 10

        page_width = doc_analysis.page_dimensions[0]
        is_centered = (
            abs(block.bbox[0] + (block.bbox[2] - block.bbox[0]) / 2 - page_width / 2)
            < page_width * 0.1
        )
        if is_centered:
            title_score += 25
        isolation_dist = _calculate_isolation(block, first_page_blocks)
        isolation_bonus = min(isolation_dist / 50.0, 1.0) * 15
        title_score += isolation_bonus
        if len(block.text) < 10:
            title_score -= 10
        if len(block.text) > 150 or block.word_count > 30:
            title_score -= 20
        is_common_section_title = False
        for common_term in common_section_titles:
            if common_term == text_lower or text_lower.startswith(common_term + " ") or text_lower.endswith(" " + common_term):
                is_common_section_title = True
                break
        if is_common_section_title:
            title_score -= thresholds.get("title_is_document_type_penalty", 20)
            logging.debug(f"Title candidate '{block.text[:50]}' penalized for being a common section title.")

        if re.match(r"^\s*\d+\s*$", text_lower) or re.match(r"^\s*page\s+\d+\s*$", text_lower):
            title_score -= 50
        if re.match(r"^(?:confidential|draft|internal document)\b", text_lower, re.IGNORECASE):
            title_score -= 30

        text_case = get_text_case_advanced(block.text)
        if text_case == "SENTENCE":
            title_score -= 5
        elif text_case == "MIXED":
            title_score -= 10

        if title_score >= thresholds.get("title_min_score", 45):
            title_candidates.append((block, title_score))
            logging.debug(f"Candidate: '{block.text}' (Score: {title_score:.2f}, Size: {block.size}, Bold: {block.is_bold}, Centered: {is_centered}, Y-ratio: {y_ratio:.2f})")

    if title_candidates:
        best_title = max(title_candidates, key=lambda x: (
            x[1],
            x[0].size,
            -abs((x[0].bbox[1] + x[0].bbox[3]) / 2 - doc_analysis.page_dimensions[1] / 2)
        ))
        logging.info(f"Selected Title: '{best_title[0].text}' with score {best_title[1]:.2f}")
        return best_title[0].text
    second_page_blocks = [b for b in blocks if b.page_num == 1]
    if second_page_blocks:
        logging.debug("No strong title on page 1. Checking page 2 for potential title.")
        second_page_candidates = []
        for block in second_page_blocks:
            title_score = 0.0
            text_lower = block.text.lower().strip()

            if block.size > body_text_size_mode * 1.1:
                title_score += (block.size / body_text_size_mode) * 15
            if block.is_bold:
                title_score += 20
            page_height = doc_analysis.page_dimensions[1]
            if block.bbox[1] < page_height * 0.25:
                title_score += 20
            page_width = doc_analysis.page_dimensions[0]
            is_centered = abs(block.bbox[0] + (block.bbox[2] - block.bbox[0]) / 2 - page_width / 2) < page_width * 0.15
            if is_centered:
                title_score += 15
            is_common_section_title = False
            for common_term in common_section_titles:
                if common_term == text_lower or text_lower.startswith(common_term + " ") or text_lower.endswith(" " + common_term):
                    is_common_section_title = True
                    break
            if is_common_section_title:
                title_score -= 30
                logging.debug(f"Page 2 candidate '{block.text[:50]}' penalized for being common section title.")

            if title_score >= thresholds.get("title_min_score", 45) * 0.8:
                second_page_candidates.append((block, title_score))

        if second_page_candidates:
            best_title = max(second_page_candidates, key=lambda x: (x[1], x[0].size))
            logging.info(f"Selected Title from Page 2: '{best_title[0].text}' with score {best_title[1]:.2f}")
            return best_title[0].text

    return "Untitled Document"


def create_hierarchical_outline(
    headings: List[Dict], title: str = None
) -> Dict[str, Any]:
    """
    Convert flat heading list to hierarchical outline structure
    """
    outline = {
        "title": title or "Untitled Document",
        "outline": [], # Changed to a list to directly store H1s
    }

    heading_stack = [] # Stores (level, reference to the dict in the outline)

    for heading in headings:
        level = heading.get("level", 1)
        heading_item = {
            "text": heading["text"],
            "page": heading["page"],
            "confidence": heading.get("confidence_score", 0.0),
            "children": [],
        }

        # Pop from stack until we find a parent or become an H1
        while heading_stack and heading_stack[-1]["level"] >= level:
            heading_stack.pop()

        if level == 1:
            outline["outline"].append(heading_item)
            heading_stack = [{"level": 1, "item": heading_item}]
        else:
            if heading_stack:
                # Append to the children of the last suitable parent
                heading_stack[-1]["item"]["children"].append(heading_item)
            else:
                # If no parent found (e.g., H2 before any H1), append to a default top-level list
                # This scenario should ideally be rare with good heading detection
                outline["outline"].append(heading_item) # Fallback, appending as a new top-level item

            heading_stack.append({"level": level, "item": heading_item})

    return outline

def create_flat_outline(headings: List[Dict], title: str = None) -> Dict[str, Any]:
    """
    Create a flat outline structure with H1, H2, H3 levels as requested,
    now including a unique ID for each item.
    """
    outline_list = []

    # Use 'enumerate' to get a unique index 'i' for each heading
    for i, heading in enumerate(headings):
        level = heading.get("level", 1)
        if 1 <= level <= 3:  # Only include H1, H2, H3
            heading_item = {
                # --- CHANGES START HERE ---

                # 1. Added a unique ID for React's 'key' prop
                "id": f"heading-{i}-{heading['page']}",
                
                # 2. Renamed 'text' to 'title' to match the frontend component
                "title": heading["text"],
                
                # --- CHANGES END HERE ---
                
                "level": f"H{level}", # Format as "H1", "H2", "H3"
                "page": heading["page"],
            }
            outline_list.append(heading_item)

    return {
        "title": title or "Untitled Document",
        "outline": outline_list,
    }
def format_json_output(
    results: Dict[str, Any], format_type: str = "flat"
) -> Dict[str, Any]:
    """
    Format extraction results as clean JSON output
    """
    headings = results.get("headings", [])
    title = results.get("title", "Untitled Document")

    # Use the modified create_flat_outline for the 'flat' type
    # The 'hierarchical' type remains as is, though it might not be chosen if 'flat' is default.
    if format_type == "hierarchical":
        return create_hierarchical_outline(headings, title)
    else: # This covers 'flat' and any other default/unrecognized type
        return create_flat_outline(headings, title)

# --- New: Centralized PDF Processing Function ---
def process_all_pdfs_in_directories(
    input_dir_path: str = "/input",
    output_dir_path: str = "/output",
    confidence: float = 25.0,
    format_type: str = "flat",
    no_semantic: bool = False,
    include_metadata: bool = False,
    pretty: bool = True,
    quiet: bool = False,
    verbose: bool = False,
):
    """
    Processes all PDF files from the input directory and saves JSON outputs.
    """
    if quiet:
        logging.basicConfig(level=logging.ERROR)
    elif verbose:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
        )
    else:
        logging.basicConfig(level=logging.INFO) # Changed to INFO for general messages

    input_dir = Path(input_dir_path)
    output_dir = Path(output_dir_path)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        logging.info(f"No PDF files found in {input_dir_path}. Exiting.")
        return 0

    logging.info(f"Starting processing of {len(pdf_files)} PDF(s) from {input_dir_path}")

    for pdf_file in pdf_files:
        output_file = output_dir / f"{pdf_file.stem}.json"
        logging.info(f"Processing '{pdf_file.name}' -> '{output_file.name}'")

        try:
            results = extract_headings_from_pdf_advanced(
                str(pdf_file),
                confidence_threshold=confidence,
                apply_semantic_filtering=not no_semantic,
            )

            if "error" in results:
                logging.error(f"Error processing {pdf_file.name}: {results['error']}")
                continue # Skip to the next file

            # Ensure the format_type passed here aligns with the desired output.
            # Since the user explicitly provided a *flat* list example,
            # we ensure create_flat_outline is used.
            json_output = format_json_output(results, format_type='flat') # Force 'flat' as per user's example

            if include_metadata:
                json_output["metadata"] = {
                    "total_headings": len(results.get("headings", [])),
                    "document_pages": results.get("document_info", {}).get("total_pages", 0),
                    "extraction_method": "advanced_pdf_analysis",
                    "confidence_threshold": confidence,
                    "format_type": "flat", # Reflect the actual format used
                    "processing_timestamp": f"{os.getenv('CURRENT_TIME', 'N/A')}", # Add timestamp
                }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_output, f, indent=2 if pretty else None, ensure_ascii=False)
            logging.info(f"Successfully saved outline for '{pdf_file.name}'.")

        except Exception as e:
            logging.error(f"Critical error processing '{pdf_file.name}': {e}")
            # Optionally, you might want to write an error JSON or log more details.

    logging.info("Completed processing all PDFs.")
    return 0

def process_pdf_for_api(pdf_bytes: bytes) -> dict:
    """
    A special wrapper function for the FastAPI backend to call.
    It takes PDF file content as bytes, processes it using your existing logic,
    and returns the final formatted JSON outline.
    """
    import tempfile
    
    # Create a temporary file to hold the PDF content
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_bytes)
        temp_pdf.flush() # Ensure all data is written to the file
        
        # Now, call your main extraction function with the path to this temporary file
        results = extract_headings_from_pdf_advanced(temp_pdf.name)
        
        if "error" in results:
            return results
            
        # Format the results into the clean, flat outline
        formatted_output = format_json_output(results, format_type='flat')
        
        return formatted_output
# --- Main function (Adjusted to call the new centralized processor) ---
def main():
    parser = argparse.ArgumentParser(
        description="PDF Heading Extractor with JSON Outline Output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script processes all PDF files found in the specified input directory
and generates a JSON outline for each in the specified output directory.

Examples:
  python main.py # Processes from /app/input to /app/output (Docker default)
  python main.py --input /my/pdfs --output /my/results --format hierarchical
  python main.py --confidence 30 --no-semantic
        """,
    )

    parser.add_argument(
        "--input",
        default="/app/input",
        help="Input directory containing PDF files (default: /app/input)",
    )
    parser.add_argument(
        "--output",
        default="/app/output",
        help="Output directory for JSON results (default: /app/output)",
    )
    parser.add_argument(
        "--format",
        choices=["flat", "hierarchical"],
        default="flat", # Default to 'flat' which now produces the requested H1/H2/H3 list
        help="Output format: flat (H1/H2/H3 list) or hierarchical (nested)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=25.0,
        help="Confidence threshold for heading detection (default: 25.0)",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic consistency filtering",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include extraction metadata in output",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True, # Explicitly default to True for more readable output
        help="Pretty print JSON output (default: True)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging (DEBUG level)")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress all output except JSON errors (ERROR level)"
    )

    args = parser.parse_args()

    # Call the new centralized processing function
    return process_all_pdfs_in_directories(
        input_dir_path=args.input,
        output_dir_path=args.output,
        confidence=args.confidence,
        format_type=args.format, # This will effectively be 'flat' by default as per argparse
        no_semantic=args.no_semantic,
        include_metadata=args.include_metadata,
        pretty=args.pretty,
        quiet=args.quiet,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    sys.exit(main())