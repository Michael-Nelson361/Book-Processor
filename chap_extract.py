import re
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
import fitz  # PyMuPDF
from dataclasses import dataclass, field
import json
from datetime import datetime
from collections import defaultdict, Counter
import logging
import traceback


def setup_logging(log_file: Optional[str] = None, verbose: bool = True):
    """Setup dual logging - detailed to file, minimal to console"""
    # Create logger
    logger = logging.getLogger('PDFChapterExtractor')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter('%(message)s')
    
    # Console handler - always set this up first
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(console_formatter)
    # Filter to only show high-level progress messages
    ch.addFilter(lambda record: record.levelno >= logging.INFO and 
                               not record.getMessage().startswith('  '))
    logger.addHandler(ch)
    
    # File handler - try to set up but handle failures gracefully
    if log_file:
        try:
            # Create parent directory if needed
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(file_formatter)
            logger.addHandler(fh)
            logger.info(f"Log file created: {log_file}")
        except Exception as e:
            # Log failure to console only
            logger.warning(f"Failed to create log file: {e}")
            logger.warning("Continuing with console logging only")
    
    return logger


# Global logger instance
logger = logging.getLogger('PDFChapterExtractor')


@dataclass
class ChapterInfo:
    """Stores information about a detected chapter or section"""
    chapter_number: Optional[int]
    title: str
    start_page: int
    end_page: Optional[int] = None
    is_intro: bool = False
    is_end: bool = False
    is_toc: bool = False
    is_section: bool = False
    section_number: Optional[int] = None
    parent_section: Optional[str] = None
    detection_method: str = "pattern"  # pattern, format, or custom
    confidence: float = 1.0
    font_size: Optional[float] = None
    is_bold: bool = False


@dataclass
class TextBlock:
    """Represents a text block with formatting info"""
    text: str
    font_size: float
    font_name: str
    flags: int  # Bold, italic, etc.
    bbox: Tuple[float, float, float, float]  # Bounding box
    page_num: int
    page_width: float  # Added for accurate centering calculation
    
    @property
    def is_bold(self) -> bool:
        return bool(self.flags & 2**4)
    
    @property
    def is_italic(self) -> bool:
        return bool(self.flags & 2**1)
    
    @property
    def is_centered(self) -> bool:
        # Text block's horizontal center
        block_center_x = (self.bbox[0] + self.bbox[2]) / 2
        # Page's horizontal center
        page_center_x = self.page_width / 2
        # Allow a tolerance (e.g., 10% of page width)
        tolerance = self.page_width * 0.10
        return abs(block_center_x - page_center_x) < tolerance


@dataclass
class DocumentStats:
    """Document formatting statistics"""
    body_font_size: float
    body_font_name: str
    heading_sizes: List[float]
    avg_font_size: float
    font_size_histogram: Dict[float, int]
    font_name_histogram: Dict[str, int]
    pages_sampled: int


class ChapterDetector:
    """Enhanced chapter detection with multiple strategies"""
    
    def __init__(self, patterns: Optional[List[str]] = None, verbose: bool = True):
        self.verbose = verbose
        
        # Standard chapter patterns - FIXED SYNTAX ERRORS
        self.chapter_patterns = patterns if patterns else [
            r'^\s*CHAPTER\s+(\d+)\s*[:.]?\s*(.*)$',  # CHAPTER 1: Title
            r'^\s*Chapter\s+(\d+)\s*[:.]?\s*(.*)$',   # Chapter 1: Title
            r'^\s*CHAPTER\s+([IVXLCDM]+)\s*[:.]?\s*(.*)$',  # CHAPTER IV: Title
            r'^\s*(\d+)\s*[:.]?\s*(.*)$',  # Matches "1", "1.", "1 Title", "1. Title"
            r'^\s*Part\s+(\d+)\s*[:.]?\s*(.*)$',  # Part 1: Title
            r'^\s*IRON\s+RULE\s+([IVXLCDM]+)\s*[:.]?\s*(.*)$',  # IRON RULE I
            # Pattern for common literary sections (empty group for title consistency)
            r'^\s*(EPILOGUE|PROLOGUE|APPENDIX|FOREWORD|PREFACE|INTRODUCTION|ABOUT THE AUTHOR)\s*()$',
        ]
        
        # Section patterns
        self.section_patterns = [
            r'^\s*SECTION\s+(\d+)\s*[:.]?\s*(.*)$',  # SECTION 1: Title
            r'^\s*Section\s+(\d+)\s*[:.]?\s*(.*)$',   # Section 1: Title
            r'^\s*PART\s+([IVXLCDM]+)\s*[:.]?\s*(.*)$',  # PART IV: Title
            r'^\s*Part\s+([IVXLCDM]+)\s*[:.]?\s*(.*)$',   # Part IV: Title
        ]
        
        # Title-only patterns (for books without chapter markers)
        # Made more restrictive to avoid false positives
        self.title_patterns = [
            r'^([A-Z][A-Z\s]{4,60})$',  # ALL CAPS TITLE (4-60 chars)
            r'^(\d+)\s*[-–—]\s*(.+)$',  # 1 - Title or 1 – Title
            # Removed Title Case pattern as it's too broad
        ]
        
        # Compile all patterns
        self.compiled_chapter_patterns = [re.compile(p, re.IGNORECASE | re.MULTILINE) 
                                         for p in self.chapter_patterns]
        self.compiled_section_patterns = [re.compile(p, re.IGNORECASE | re.MULTILINE) 
                                        for p in self.section_patterns]
        self.compiled_title_patterns = [re.compile(p, re.MULTILINE) 
                                      for p in self.title_patterns]
        
        # TOC detection patterns
        self.toc_patterns = [
            re.compile(r'table\s+of\s+contents', re.IGNORECASE),
            re.compile(r'^\s*contents\s*$', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^\s*index\s*$', re.IGNORECASE | re.MULTILINE),
        ]
        
        # Track formatting statistics
        self.font_sizes: Set[float] = set()
        self.font_names: Set[str] = set()
        self.heading_candidates: List[TextBlock] = []
    
    def is_toc_page(self, text: str) -> bool:
        """Check if the page is likely a table of contents"""
        for pattern in self.toc_patterns:
            if pattern.search(text):
                return True
        return False
    
    def detect_chapter(self, text: str, page_num: int = None) -> Optional[Tuple[str, str, str]]:
        """
        Detect if text contains a chapter heading
        Returns: (chapter_number, title, type) or None
        """
        # Try standard chapter patterns
        for i, pattern in enumerate(self.compiled_chapter_patterns):
            match = pattern.match(text.strip())
            if match:
                logger.debug(f"  ✓ Found chapter match on page {page_num + 1} using pattern {i + 1}")
                return match.group(1), match.group(2).strip() if len(match.groups()) > 1 else "", "chapter"
        
        # Try section patterns
        for i, pattern in enumerate(self.compiled_section_patterns):
            match = pattern.match(text.strip())
            if match:
                logger.debug(f"  ✓ Found section match on page {page_num + 1}")
                return match.group(1), match.group(2).strip() if len(match.groups()) > 1 else "", "section"
        
        return None
    
    def extract_text_blocks(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """Extract text blocks with formatting information"""
        blocks = []
        
        try:
            # Get actual page width
            actual_page_width = page.rect.width
            
            # Get text blocks with formatting
            block_dict = page.get_text("dict")
            
            for block in block_dict["blocks"]:
                if block["type"] == 0:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text and len(text) > 2:  # Skip very short text
                                tb = TextBlock(
                                    text=text,
                                    font_size=span["size"],
                                    font_name=span["font"],
                                    flags=span["flags"],
                                    bbox=span["bbox"],
                                    page_num=page_num,
                                    page_width=actual_page_width  # Pass actual page width
                                )
                                blocks.append(tb)
                                
                                # Track font statistics
                                self.font_sizes.add(span["size"])
                                self.font_names.add(span["font"])
        except Exception as e:
            logger.error(f"Error extracting text blocks from page {page_num + 1}: {str(e)}")
            logger.debug(traceback.format_exc())
        
        return blocks
    
    def is_likely_heading(self, block: TextBlock, doc_stats: DocumentStats) -> Tuple[bool, float]:
        """
        Determine if a text block is likely a heading based on formatting
        Returns: (is_heading, confidence)
        """
        confidence = 0.0
        factors = []
        
        # Factor 1: Font size compared to body text
        size_ratio = block.font_size / doc_stats.body_font_size if doc_stats.body_font_size > 0 else 1.0
        if size_ratio >= 1.5:  # Significantly larger than body text
            confidence += 0.4
            factors.append("large_font")
        elif size_ratio >= 1.3:
            confidence += 0.2
            factors.append("medium_font")
        
        # Factor 2: Bold text
        if block.is_bold:
            confidence += 0.3
            factors.append("bold")
        
        # Factor 3: Different font from body text
        if block.font_name != doc_stats.body_font_name:
            confidence += 0.1
            factors.append("different_font")
        
        # Factor 4: Centered text (only valuable if combined with other factors)
        if block.is_centered and len(factors) > 0:
            confidence += 0.1
            factors.append("centered")
        
        # Factor 5: ALL CAPS (but restrict length to avoid false positives)
        if block.text.isupper() and 4 <= len(block.text) <= 60:
            confidence += 0.2
            factors.append("all_caps")
        
        # Factor 6: Matches title patterns (reduced weight)
        for pattern in self.compiled_title_patterns:
            if pattern.match(block.text):
                confidence += 0.1
                factors.append("title_pattern")
                break
        
        # Special handling for standalone numerals (like in "The Time Machine")
        if block.text.isdigit() and len(block.text) <= 2:  # e.g., "1" to "99"
            if block.is_centered and size_ratio >= 1.8:  # Significantly larger & centered
                if "large_centered_numeral" not in factors:
                    factors.append("large_centered_numeral")
                confidence = max(confidence, 0.85)  # Boost confidence for strong candidates
            elif size_ratio >= 1.5:  # Still quite large but maybe not centered
                if "large_numeral" not in factors:
                    factors.append("large_numeral")
                confidence = max(confidence, 0.75)
        
        # Require multiple factors for high confidence
        if len(factors) < 2:
            confidence *= 0.5  # Reduce confidence if only one factor
        
        # Special case: if it's just ALL CAPS without size change, very low confidence
        if factors == ["all_caps"] or factors == ["title_pattern"]:
            confidence = 0.3
        
        # Cap confidence at 1.0
        confidence = min(confidence, 1.0)
        
        return confidence >= 0.6, confidence


class PDFChapterExtractor:
    """Enhanced PDF chapter extractor with format-based detection"""
    
    def __init__(self, 
                 chapter_detector: Optional[ChapterDetector] = None,
                 min_text_length: int = 20,
                 output_dir: Optional[str] = None,
                 verbose: bool = True,
                 skip_toc_pages: int = 0,
                 use_format_detection: bool = True,
                 format_sensitivity: str = "medium",
                 min_chapter_pages: int = 3,
                 custom_patterns: Optional[List[str]] = None,
                 log_file: Optional[str] = None):
        """
        Initialize the extractor
        
        Args:
            chapter_detector: Custom chapter detector
            min_text_length: Minimum text length to consider a page non-empty
            output_dir: Output directory
            verbose: Whether to print detailed progress
            skip_toc_pages: Pages to skip after TOC
            use_format_detection: Use font-based detection for chapters
            format_sensitivity: Sensitivity for format detection (low/medium/high)
            min_chapter_pages: Minimum pages for a format-detected chapter
            custom_patterns: Additional custom patterns to use
            log_file: Path to log file (created if not exists)
        """
        self.verbose = verbose
        self.detector = chapter_detector or ChapterDetector(patterns=custom_patterns, verbose=verbose)
        self.min_text_length = min_text_length
        self.output_dir = output_dir
        self.skip_toc_pages = skip_toc_pages
        self.use_format_detection = use_format_detection
        self.min_chapter_pages = min_chapter_pages
        self.log_file = log_file
        
        # Set confidence thresholds based on sensitivity
        self.confidence_thresholds = {
            "low": 0.8,
            "medium": 0.7,
            "high": 0.6
        }
        self.confidence_threshold = self.confidence_thresholds.get(format_sensitivity, 0.7)
    
    def log(self, message: str, level: str = "INFO"):
        """Log message with appropriate level"""
        level_map = {
            "INFO": logging.INFO,
            "SUCCESS": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "PROGRESS": logging.INFO,
            "DEBUG": logging.DEBUG
        }
        log_level = level_map.get(level, logging.INFO)
        
        # Add emoji prefix for file logging
        prefix = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✅",
            "WARNING": "⚠️ ",
            "ERROR": "❌",
            "PROGRESS": "🔄",
            "DEBUG": "🔍"
        }.get(level, "  ")
        
        logger.log(log_level, f"{prefix} {message}")
    
    def extract_book_name(self, filepath: str) -> str:
        """Extract book name from filepath"""
        path = Path(filepath)
        book_name = path.stem
        self.log(f"Extracted book name: '{book_name}'", "DEBUG")
        return book_name
    
    def sanitize_filename(self, filename: str, max_length: int = 80) -> str:
        """Sanitize and truncate filename for filesystem compatibility"""
        # Remove/replace invalid characters for Windows/Unix filesystems
        invalid_chars = r'<>:"/\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remove multiple spaces and trim
        filename = ' '.join(filename.split())
        
        # Truncate if too long, keeping extension if present
        if len(filename) > max_length:
            # Try to truncate at a word boundary
            truncated = filename[:max_length]
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.7:  # If space is reasonably far in
                filename = truncated[:last_space]
            else:
                filename = truncated
            filename = filename.rstrip(' ._-')  # Clean up trailing chars
        
        return filename
    
    def analyze_document_structure(self, doc: fitz.Document) -> DocumentStats:
        """Analyze document to understand its structure"""
        self.log("Analyzing document structure...", "PROGRESS")
        
        # Collect font statistics
        font_sizes = defaultdict(int)
        font_names = defaultdict(int)
        total_blocks = 0
        
        # Sample first 50 pages or 20% of document
        sample_size = min(50, max(10, int(len(doc) * 0.2)))
        
        for page_num in range(sample_size):
            try:
                page = doc[page_num]
                blocks = self.detector.extract_text_blocks(page, page_num)
                
                for block in blocks:
                    # Only count substantial text blocks
                    if len(block.text) > 20:
                        font_sizes[block.font_size] += 1
                        font_names[block.font_name] += 1
                        total_blocks += 1
            except Exception as e:
                logger.error(f"Error analyzing page {page_num + 1}: {str(e)}")
        
        # Find the dominant body font (most common font size)
        body_font_size = 12.0  # Default
        body_font_name = ""
        
        if font_sizes:
            # Get the most common font size (likely body text)
            body_font_size = max(font_sizes.items(), key=lambda x: x[1])[0]
            
            # Get the most common font name
            if font_names:
                body_font_name = max(font_names.items(), key=lambda x: x[1])[0]
        
        # Calculate average font size
        if total_blocks > 0:
            weighted_sum = sum(size * count for size, count in font_sizes.items())
            avg_font_size = weighted_sum / total_blocks
        else:
            avg_font_size = body_font_size
        
        # Find likely heading sizes (significantly larger than body text)
        heading_sizes = [size for size in font_sizes.keys() 
                        if size >= body_font_size * 1.3]
        
        logger.debug(f"Body font: {body_font_size:.1f}pt ({body_font_name})")
        logger.debug(f"Average font size: {avg_font_size:.1f}pt")
        logger.debug(f"Detected {len(heading_sizes)} potential heading sizes")
        
        return DocumentStats(
            body_font_size=body_font_size,
            body_font_name=body_font_name,
            heading_sizes=sorted(heading_sizes, reverse=True),
            avg_font_size=avg_font_size,
            font_size_histogram=dict(font_sizes),
            font_name_histogram=dict(font_names),
            pages_sampled=sample_size
        )
    
    def detect_chapters_with_format(self, doc: fitz.Document, doc_stats: DocumentStats) -> List[ChapterInfo]:
        """Detect chapters using formatting information"""
        chapters = []
        current_section = None
        chapter_count = 0
        section_count = 0
        last_heading_page = -10  # Track last heading to avoid adjacent headings
        
        self.log("Using format-based detection for chapters...", "DEBUG")
        logger.debug(f"Confidence threshold: {self.confidence_threshold}")
        
        for page_num in range(len(doc)):
            if page_num % 10 == 0:
                logger.debug(f"  Analyzing formatting on page {page_num + 1}/{len(doc)}...")
            
            try:
                page = doc[page_num]
                blocks = self.detector.extract_text_blocks(page, page_num)
                
                # Look for potential headings on this page
                best_heading = None
                best_confidence = 0.0
                
                for block in blocks:
                    if len(block.text) < 3 or len(block.text) > 100:  # Skip very short or very long text
                        continue
                    
                    is_heading, confidence = self.detector.is_likely_heading(block, doc_stats)
                    
                    if is_heading and confidence > best_confidence:
                        best_heading = block
                        best_confidence = confidence
                
                # Process the best heading if it meets threshold
                if best_heading and best_confidence >= self.confidence_threshold:
                    # Check proximity to last heading
                    pages_since_last = page_num - last_heading_page
                    
                    if pages_since_last < 2:
                        logger.debug(f"  ⚠️ Skipping heading on page {page_num + 1} (too close to previous)")
                        continue
                    
                    # Check if it matches any patterns first
                    pattern_match = self.detector.detect_chapter(best_heading.text, page_num)
                    
                    if pattern_match:
                        num, title, ch_type = pattern_match
                        if ch_type == "section":
                            section_count += 1
                            current_section = title or f"Section {section_count}"
                            chapters.append(ChapterInfo(
                                chapter_number=None,
                                section_number=section_count,
                                title=current_section,
                                start_page=page_num,
                                is_section=True,
                                detection_method="pattern",
                                font_size=best_heading.font_size,
                                is_bold=best_heading.is_bold
                            ))
                        else:
                            chapter_count += 1
                            chapters.append(ChapterInfo(
                                chapter_number=chapter_count,
                                title=title or best_heading.text,
                                start_page=page_num,
                                parent_section=current_section,
                                detection_method="pattern",
                                font_size=best_heading.font_size,
                                is_bold=best_heading.is_bold
                            ))
                    else:
                        # Format-based detection only
                        # Be more conservative with format-only detection
                        if best_confidence >= self.confidence_threshold + 0.1:
                            chapter_count += 1
                            chapters.append(ChapterInfo(
                                chapter_number=chapter_count,
                                title=best_heading.text,
                                start_page=page_num,
                                parent_section=current_section,
                                detection_method="format",
                                confidence=best_confidence,
                                font_size=best_heading.font_size,
                                is_bold=best_heading.is_bold
                            ))
                            
                            logger.debug(f"  📄 Detected by format: '{best_heading.text}' at page {page_num + 1} (confidence: {best_confidence:.0%})")
                    
                    last_heading_page = page_num
            except Exception as e:
                logger.error(f"Error detecting chapters on page {page_num + 1}: {str(e)}")
                logger.debug(traceback.format_exc())
        
        return chapters
    
    def post_process_chapters(self, chapters: List[ChapterInfo], total_pages: int) -> List[ChapterInfo]:
        """Post-process chapters to merge false positives and validate structure"""
        if not chapters:
            return chapters
        
        self.log("Post-processing chapters...", "DEBUG")
        
        # Sort by start page
        chapters.sort(key=lambda x: x.start_page)
        
        # Calculate end pages
        for i in range(len(chapters) - 1):
            chapters[i].end_page = chapters[i + 1].start_page - 1
        chapters[-1].end_page = total_pages - 1
        
        # Filter out chapters that are too short (format-detected only)
        filtered_chapters = []
        merged_count = 0
        
        for i, chapter in enumerate(chapters):
            page_count = (chapter.end_page - chapter.start_page + 1) if chapter.end_page else 1
            
            # Keep pattern-detected chapters regardless of length
            if chapter.detection_method == "pattern":
                filtered_chapters.append(chapter)
            # For format-detected chapters, check minimum page count
            elif page_count >= self.min_chapter_pages:
                filtered_chapters.append(chapter)
            # Low confidence format detections with few pages are likely false positives
            elif chapter.confidence < 0.7 and page_count < 3:
                merged_count += 1
                logger.debug(f"  🔀 Merging short chapter: '{chapter.title}' ({page_count} pages)")
        
        # Check for excessive format-detected chapters in sequence
        format_sequence = []
        final_chapters = []
        
        for chapter in filtered_chapters:
            if chapter.detection_method == "format":
                format_sequence.append(chapter)
            else:
                # Process any accumulated format sequence
                if len(format_sequence) >= 5:
                    # Too many consecutive format detections - keep only highest confidence
                    logger.warning(f"Found {len(format_sequence)} consecutive format-detected chapters - filtering")
                    format_sequence.sort(key=lambda x: x.confidence, reverse=True)
                    # Keep top 20% or at least 1
                    keep_count = max(1, len(format_sequence) // 5)
                    final_chapters.extend(format_sequence[:keep_count])
                    merged_count += len(format_sequence) - keep_count
                else:
                    final_chapters.extend(format_sequence)
                
                format_sequence = []
                final_chapters.append(chapter)
        
        # Handle remaining format sequence
        if format_sequence:
            if len(format_sequence) >= 5:
                format_sequence.sort(key=lambda x: x.confidence, reverse=True)
                keep_count = max(1, len(format_sequence) // 5)
                final_chapters.extend(format_sequence[:keep_count])
                merged_count += len(format_sequence) - keep_count
            else:
                final_chapters.extend(format_sequence)
        
        # Re-sort and recalculate end pages
        final_chapters.sort(key=lambda x: x.start_page)
        for i in range(len(final_chapters) - 1):
            final_chapters[i].end_page = final_chapters[i + 1].start_page - 1
        if final_chapters:
            final_chapters[-1].end_page = total_pages - 1
        
        if merged_count > 0:
            logger.debug(f"Merged/removed {merged_count} false positive chapters")
        
        return final_chapters
    
    def analyze_pdf(self, filepath: str) -> Tuple[List[ChapterInfo], fitz.Document]:
        """Analyze PDF and detect all chapters/sections"""
        self.log("Opening PDF file...", "INFO")
        
        try:
            doc = fitz.open(filepath)
            total_pages = len(doc)
            self.log(f"PDF loaded. Total pages: {total_pages}", "INFO")
        except Exception as e:
            logger.error(f"Failed to open PDF: {str(e)}")
            raise
        
        chapters = []
        toc_end_page = -1
        
        # First, analyze document structure if format detection is enabled
        doc_stats = None
        if self.use_format_detection:
            try:
                doc_stats = self.analyze_document_structure(doc)
            except Exception as e:
                logger.error(f"Error analyzing document structure: {str(e)}")
                logger.debug(traceback.format_exc())
        
        self.log("Scanning for chapters...", "INFO")
        
        # Standard pattern-based detection first
        for page_num in range(total_pages):
            if page_num % 50 == 0 and page_num > 0:
                logger.debug(f"  Progress: {page_num}/{total_pages} pages scanned")
            
            try:
                page = doc[page_num]
                text = page.get_text()
                
                # Skip very short pages
                if len(text.strip()) < self.min_text_length:
                    continue
                
                # Check if this is a TOC page
                if self.detector.is_toc_page(text):
                    logger.debug(f"  📑 Table of Contents detected at page {page_num + 1}")
                    toc_end_page = page_num + self.skip_toc_pages
                    continue
                
                # Skip TOC pages
                if page_num <= toc_end_page:
                    continue
                
                # Check for chapters/sections using patterns
                lines = text.strip().split('\n')[:10]  # Check first 10 lines
                
                for line in lines:
                    result = self.detector.detect_chapter(line, page_num)
                    if result:
                        num_val, title_val, ch_type = result
                        
                        # Parse chapter/section number
                        parsed_num = None
                        try:
                            roman_map = {
                                'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
                                'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
                                'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
                                'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20
                            }
                            if num_val.upper() in roman_map:
                                parsed_num = roman_map[num_val.upper()]
                            else:
                                parsed_num = int(num_val)
                        except:
                            parsed_num = None
                        
                        # Check if this is a special section (EPILOGUE, PROLOGUE, etc.)
                        special_sections = ["EPILOGUE", "PROLOGUE", "APPENDIX", "FOREWORD", 
                                          "PREFACE", "INTRODUCTION", "ABOUT THE AUTHOR"]
                        is_special_section = num_val.upper() in special_sections
                        
                        if is_special_section:
                            # Handle special sections
                            actual_title = num_val.strip().title()  # e.g., "Epilogue"
                            chapter_info = ChapterInfo(
                                chapter_number=None,
                                title=actual_title,
                                start_page=page_num,
                                detection_method="pattern",
                                is_intro=(actual_title.upper() in ["INTRODUCTION", "PREFACE", "FOREWORD"]),
                                is_end=(actual_title.upper() in ["EPILOGUE", "ABOUT THE AUTHOR", "APPENDIX"])
                            )
                        elif ch_type == "section":
                            chapter_info = ChapterInfo(
                                chapter_number=None,
                                section_number=parsed_num,
                                title=title_val or f"Section {parsed_num or len(chapters) + 1}",
                                start_page=page_num,
                                is_section=True,
                                detection_method="pattern"
                            )
                        else:
                            # Standard chapter
                            chapter_count = len([c for c in chapters if not c.is_section and not c.is_intro and not c.is_end])
                            chapter_info = ChapterInfo(
                                chapter_number=parsed_num,
                                title=title_val or f"Chapter {parsed_num or chapter_count + 1}",
                                start_page=page_num,
                                detection_method="pattern"
                            )
                        
                        # Avoid duplicate chapters on the same page
                        if not any(c.start_page == page_num for c in chapters):
                            chapters.append(chapter_info)
                            
                            if is_special_section:
                                type_str = "Special Section"
                                num_display = chapter_info.title
                            else:
                                type_str = "Section" if ch_type == "section" else "Chapter"
                                num_display = parsed_num or '?'
                            logger.debug(f"  📖 {type_str} {num_display} found: '{chapter_info.title}' at page {page_num + 1}")
                        break
            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # Count pattern-detected chapters
        pattern_chapters = [ch for ch in chapters if ch.detection_method == "pattern"]
        logger.debug(f"Found {len(pattern_chapters)} chapters using patterns")
        
        # If few chapters found and format detection is enabled, try that
        if len(pattern_chapters) < 5 and self.use_format_detection and doc_stats:
            self.log("Attempting format-based detection...", "DEBUG")
            try:
                format_chapters = self.detect_chapters_with_format(doc, doc_stats)
                
                # Merge results, avoiding duplicates
                existing_pages = {ch.start_page for ch in chapters}
                added_format = 0
                for ch in format_chapters:
                    if ch.start_page not in existing_pages:
                        chapters.append(ch)
                        added_format += 1
                
                if added_format > 0:
                    logger.debug(f"Added {added_format} chapters using format detection")
            except Exception as e:
                logger.error(f"Error in format-based detection: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # Post-process chapters
        try:
            chapters = self.post_process_chapters(chapters, total_pages)
        except Exception as e:
            logger.error(f"Error in post-processing: {str(e)}")
            logger.debug(traceback.format_exc())
        
        self.log(f"Chapter detection complete. Found {len(chapters)} chapters/sections", "SUCCESS")
        
        # Process chapter boundaries
        if chapters:
            logger.debug("Processing chapter boundaries...")
            
            # Add intro section if needed
            first_content_start = chapters[0].start_page if chapters else total_pages
            if first_content_start > 0:
                # Check for substantial intro content
                has_content = False
                for page_num in range(first_content_start):
                    try:
                        page = doc[page_num]
                        text = page.get_text().strip()
                        if len(text) > self.min_text_length * 2:
                            has_content = True
                            break
                    except Exception as e:
                        logger.error(f"Error checking intro content on page {page_num + 1}: {str(e)}")
                
                if has_content:
                    intro = ChapterInfo(
                        chapter_number=None,
                        title="Introduction",
                        start_page=0,
                        end_page=first_content_start - 1,
                        is_intro=True
                    )
                    chapters.insert(0, intro)
                    logger.debug(f"Added intro section: pages 1-{intro.end_page + 1}")
        
        return chapters, doc
    
    def generate_filename(self, chapter: ChapterInfo, book_name: str) -> str:
        """Generate filename based on naming convention"""
        start_page = chapter.start_page + 1
        end_page = chapter.end_page + 1 if chapter.end_page is not None else start_page
        
        # Don't include book name in filename since it's already in the folder name
        if chapter.is_intro:
            base_name = f"[INTRO] {chapter.title}"
        elif chapter.is_end:
            base_name = f"[END] {chapter.title}"
        elif chapter.is_section:
            section_num = chapter.section_number or "?"
            base_name = f"[SECTION{section_num}] {chapter.title}"
        else:
            ch_num = chapter.chapter_number or "?"
            if chapter.parent_section:
                base_name = f"[SECTION{chapter.parent_section}-CH{ch_num}] {chapter.title}"
            else:
                base_name = f"[CH{ch_num}] {chapter.title}"
        
        # Add page range
        filename = f"{base_name} [{start_page}-{end_page}].pdf"
        
        # Sanitize the final filename (but less aggressively since no book name)
        return self.sanitize_filename(filename, max_length=120)
    
    def extract_chapters(self, filepath: str, save_metadata: bool = True) -> Dict[str, any]:
        """Main method to extract all chapters from a PDF"""
        logger.info(f"Starting extraction for: {os.path.basename(filepath)}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDF file not found: {filepath}")
        
        # Extract and sanitize book name
        book_name_full = self.extract_book_name(filepath)
        book_name_safe = self.sanitize_filename(book_name_full, max_length=80)
        
        if book_name_full != book_name_safe:
            logger.info(f"Book name sanitized: '{book_name_safe}'")
        
        output_dir = self.output_dir or os.path.dirname(filepath)
        
        output_path = Path(output_dir) / book_name_safe
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_path}")
        
        try:
            chapters, doc = self.analyze_pdf(filepath)
            total_pages = len(doc)
        except Exception as e:
            logger.error(f"Failed to analyze PDF: {str(e)}")
            raise
        
        if not chapters:
            doc.close()
            raise ValueError("No chapters detected in the PDF")
        
        logger.info(f"Extracting {len(chapters)} sections...")
        extracted_files = []
        
        for i, chapter in enumerate(chapters):
            if chapter.end_page is None:
                continue
            
            # Use full book name for display but safe name for files
            filename = self.generate_filename(chapter, book_name_full)
            full_path = output_path / filename
            
            # Log detailed progress to file only
            if chapter.is_intro:
                ch_type = "intro"
            elif chapter.is_section:
                ch_type = f"section {chapter.section_number}"
            else:
                ch_type = f"chapter {chapter.chapter_number}"
            
            pages = chapter.end_page - chapter.start_page + 1
            method = f" ({chapter.detection_method})" if chapter.detection_method != "pattern" else ""
            confidence = f" [{chapter.confidence:.0%}]" if hasattr(chapter, 'confidence') and chapter.confidence < 1 else ""
            logger.debug(f"  [{i+1}/{len(chapters)}] Extracting {ch_type}: {pages} pages -> {filename}{method}{confidence}")
            
            try:
                chapter_doc = fitz.open()
                
                for page_num in range(chapter.start_page, chapter.end_page + 1):
                    chapter_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                
                chapter_doc.save(str(full_path))
                chapter_doc.close()
                
                extracted_files.append({
                    'filename': filename,
                    'path': str(full_path),
                    'start_page': chapter.start_page + 1,
                    'end_page': chapter.end_page + 1,
                    'chapter_number': chapter.chapter_number,
                    'section_number': chapter.section_number,
                    'title': chapter.title,
                    'is_intro': chapter.is_intro,
                    'is_section': chapter.is_section,
                    'detection_method': chapter.detection_method,
                    'confidence': getattr(chapter, 'confidence', 1.0)
                })
            except Exception as e:
                logger.error(f"Failed to extract {ch_type}: {str(e)}")
                logger.debug(traceback.format_exc())
        
        doc.close()
        
        # Count chapters and sections
        chapter_count = len([c for c in chapters if not c.is_intro and not c.is_end and not c.is_section])
        section_count = len([c for c in chapters if c.is_section])
        
        metadata = {
            'source_file': filepath,
            'book_name': book_name_full,  # Keep full name in metadata
            'book_name_safe': book_name_safe,  # Also store safe name
            'total_pages': total_pages,
            'chapters_found': chapter_count,
            'sections_found': section_count,
            'files_created': len(extracted_files),
            'output_directory': str(output_path),
            'extracted_files': extracted_files
        }
        
        if save_metadata:
            metadata_path = output_path / f"{book_name_safe}_extraction_metadata.json"
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.debug(f"Metadata saved to: {metadata_path}")
            except Exception as e:
                logger.error(f"Failed to save metadata: {str(e)}")
        
        logger.info(f"Extraction complete! {len(extracted_files)} files created")
        
        return metadata


def main():
    """Main function to handle command-line execution"""
    parser = argparse.ArgumentParser(
        description="Extract chapters from a PDF book into separate PDF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s book.pdf                        # Extract with format detection (medium sensitivity)
  %(prog)s book.pdf --patterns-only        # Use only pattern matching
  %(prog)s book.pdf --sensitivity low      # Conservative format detection
  %(prog)s book.pdf --min-pages 5          # Require 5+ pages per chapter
  %(prog)s book.pdf --pattern "^Lesson"    # Add custom pattern
  %(prog)s book.pdf --skip-toc-pages 5     # Skip pages after TOC
  %(prog)s book.pdf --quiet                # Minimal output
  %(prog)s book.pdf --log-file extract.log # Save detailed log
        """
    )
    
    parser.add_argument('filepath', help='Path to the PDF file')
    parser.add_argument('-o', '--output', help='Output directory')
    parser.add_argument('-q', '--quiet', action='store_true', help='Minimal console output')
    parser.add_argument('--no-metadata', action='store_true', help='Don\'t save metadata')
    parser.add_argument('--min-text-length', type=int, default=20, help='Min text length for non-empty page')
    parser.add_argument('--skip-toc-pages', type=int, default=0, help='Pages to skip after TOC')
    parser.add_argument('--patterns-only', action='store_true', help='Disable format-based detection')
    parser.add_argument('--pattern', action='append', help='Add custom chapter pattern (can use multiple times)')
    parser.add_argument('--sensitivity', choices=['low', 'medium', 'high'], default='medium',
                       help='Format detection sensitivity (default: medium)')
    parser.add_argument('--min-pages', type=int, default=3, 
                       help='Minimum pages for format-detected chapters (default: 3)')
    parser.add_argument('--log-file', help='Path to log file (default: <book_name>_extraction.log)')
    
    args = parser.parse_args()
    
    # Setup log file
    if args.log_file:
        log_file = args.log_file
    else:
        # Generate log filename based on input file
        book_path = Path(args.filepath)
        log_file = book_path.parent / f"{book_path.stem}_extraction.log"
    
    # Setup logging
    setup_logging(log_file=str(log_file), verbose=not args.quiet)
    
    # Log startup
    logger.info("="*60)
    logger.info("PDF Chapter Extractor - Starting")
    logger.info("="*60)
    logger.info(f"Log file: {log_file}")
    
    try:
        extractor = PDFChapterExtractor(
            output_dir=args.output,
            min_text_length=args.min_text_length,
            verbose=not args.quiet,
            skip_toc_pages=args.skip_toc_pages,
            use_format_detection=not args.patterns_only,
            format_sensitivity=args.sensitivity,
            min_chapter_pages=args.min_pages,
            custom_patterns=args.pattern,
            log_file=str(log_file)
        )
        
        results = extractor.extract_chapters(
            args.filepath,
            save_metadata=not args.no_metadata
        )
        
        # Summary
        logger.info("="*60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("="*60)
        logger.info(f"📚 Book: {results['book_name']}")
        logger.info(f"📄 Total pages: {results['total_pages']}")
        logger.info(f"📖 Chapters found: {results['chapters_found']}")
        logger.info(f"📑 Sections found: {results['sections_found']}")
        logger.info(f"💾 Files created: {results['files_created']}")
        logger.info(f"📁 Output: {results['output_directory']}")
        logger.info("="*60)
        
        # Console output for quiet mode
        if args.quiet:
            print(f"Extracted {results['files_created']} files to: {results['output_directory']}")
            print(f"Log saved to: {log_file}")
    
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        logger.debug(traceback.format_exc())
        if args.quiet:
            print(f"❌ Error: {e}")
            print(f"Check log file for details: {log_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()