import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Document type classification"""
    UNKNOWN = "unknown"
    ACADEMIC_PAPER = "academic_paper"
    LEGAL_DOCUMENT = "legal_document"
    TECHNICAL_MANUAL = "technical_manual"
    BUSINESS_REPORT = "business_report"
    PRESENTATION = "presentation"
    BOOK = "book"

@dataclass
class Section:
    """Represents a document section with hierarchy"""
    level: int  # 1 for H1, 2 for H2, etc.
    title: str
    content: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    confidence: float = 1.0
    parent_section: Optional['Section'] = None
    subsections: List['Section'] = None
    
    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []

@dataclass
class DocumentStructure:
    """Complete document structure analysis"""
    document_type: DocumentType
    sections: List[Section]
    page_count: int
    metadata: Dict[str, Any]
    confidence_scores: Dict[str, float]

class DocumentStructureAnalyzer:
    """
    Enhanced document structure analysis for improved metadata extraction
    Uses layout analysis and hierarchical section detection
    """
    
    def __init__(self):
        self.section_patterns = {
            'h1': [
                r'^#\s+(.+)$',  # Markdown H1
                r'^[A-Z][A-Z\s]{10,}$',  # ALL CAPS HEADING
                r'^\d+\.\s+[A-Z][^a-z]{20,}$',  # Numbered ALL CAPS
                r'^[IVX]+\.\s+[A-Z][^a-z]{15,}$',  # Roman numeral ALL CAPS
            ],
            'h2': [
                r'^##\s+(.+)$',  # Markdown H2
                r'^\d+\.\d+\s+.+$',  # Numbered subheading
                r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title Case
                r'^[A-Z][^.]{5,50}[.:]$',  # Sentence-like heading
            ],
            'h3': [
                r'^###\s+(.+)$',  # Markdown H3
                r'^\d+\.\d+\.\d+\s+.+$',  # Deep numbered
                r'^[a-z][^.]{5,40}[.:]$',  # Lowercase sentence heading
            ]
        }
        
        self.page_patterns = [
            r'\bpage\s+(\d+)\b',
            r'\bp\.?\s*(\d+)\b',
            r'^\s*(\d+)\s*$',  # Standalone page number
            r'^-\s*(\d+)\s*-$',  # Centered page number
        ]
        
        self.document_type_patterns = {
            DocumentType.ACADEMIC_PAPER: [
                r'\babstract\b', r'\bintroduction\b', r'\bmethodology\b',
                r'\bresults\b', r'\bdiscussion\b', r'\breferences\b',
                r'\bcitation\b', r'\bliterature review\b'
            ],
            DocumentType.LEGAL_DOCUMENT: [
                r'\bsection\b', r'\bclause\b', r'\barticle\b',
                r'\bwhereas\b', r'\bhereby\b', r'\bparty\b',
                r'\bagreement\b', r'\bcontract\b'
            ],
            DocumentType.TECHNICAL_MANUAL: [
                r'\bchapter\b', r'\bprocedure\b', r'\bstep\b',
                r'\bwarning\b', r'\bcaution\b', r'\bnote\b',
                r'\btable\b', r'\bfigure\b'
            ],
            DocumentType.BUSINESS_REPORT: [
                r'\bexecutive summary\b', r'\bmarket analysis\b',
                r'\bfinancials\b', r'\brecommendations\b',
                r'\bconclusion\b', r'\bappendix\b'
            ]
        }
    
    def analyze_document_structure(self, content: str, filename: str = None) -> DocumentStructure:
        """
        Analyze document structure and extract enhanced metadata
        
        Args:
            content: Document text content
            filename: Optional filename for context
            
        Returns:
            DocumentStructure with hierarchical sections and metadata
        """
        try:
            lines = content.split('\n')
            
            # Step 1: Classify document type
            document_type = self._classify_document_type(content, filename)
            
            # Step 2: Extract page numbers and structure
            page_markers = self._extract_page_markers(lines)
            
            # Step 3: Detect hierarchical sections
            sections = self._detect_hierarchical_sections(lines, page_markers)
            
            # Step 4: Build section hierarchy
            structured_sections = self._build_section_hierarchy(sections)
            
            # Step 5: Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                structured_sections, page_markers, document_type
            )
            
            return DocumentStructure(
                document_type=document_type,
                sections=structured_sections,
                page_count=len(page_markers) if page_markers else 1,
                metadata={
                    'filename': filename,
                    'total_sections': len(structured_sections),
                    'max_section_depth': self._get_max_depth(structured_sections),
                    'page_markers_found': len(page_markers),
                },
                confidence_scores=confidence_scores
            )
            
        except Exception as e:
            logger.error(f"Error analyzing document structure: {str(e)}")
            # Return minimal structure on error
            return DocumentStructure(
                document_type=DocumentType.UNKNOWN,
                sections=[],
                page_count=1,
                metadata={'error': str(e)},
                confidence_scores={'overall': 0.0}
            )
    
    def _classify_document_type(self, content: str, filename: str) -> DocumentType:
        """Classify document type based on content patterns"""
        content_lower = content.lower()
        filename_lower = filename.lower() if filename else ""
        
        scores = {}
        
        for doc_type, patterns in self.document_type_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    score += 1
            scores[doc_type] = score
        
        # Check filename extensions
        if filename:
            if any(ext in filename_lower for ext in ['.pdf', '.doc', '.docx']):
                if 'contract' in filename_lower or 'agreement' in filename_lower:
                    scores[DocumentType.LEGAL_DOCUMENT] += 2
                elif 'manual' in filename_lower or 'guide' in filename_lower:
                    scores[DocumentType.TECHNICAL_MANUAL] += 2
                elif 'report' in filename_lower:
                    scores[DocumentType.BUSINESS_REPORT] += 2
        
        # Return type with highest score
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])[0]
            if scores[best_type] > 0:
                return best_type
        
        return DocumentType.UNKNOWN
    
    def _extract_page_markers(self, lines: List[str]) -> List[Tuple[int, int]]:
        """
        Extract page numbers and their line positions
        
        Returns:
            List of (line_number, page_number) tuples
        """
        page_markers = []
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            
            for pattern in self.page_patterns:
                match = re.search(pattern, line_clean, re.IGNORECASE)
                if match:
                    try:
                        page_num = int(match.group(1))
                        # Validate page number (reasonable range)
                        if 1 <= page_num <= 1000:
                            page_markers.append((i, page_num))
                            break
                    except ValueError:
                        continue
        
        return page_markers
    
    def _detect_hierarchical_sections(self, lines: List[str], page_markers: List[Tuple[int, int]]) -> List[Section]:
        """Detect sections with hierarchical levels"""
        sections = []
        current_page = 1
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
            
            # Update current page based on markers
            for marker_line, page_num in page_markers:
                if i >= marker_line:
                    current_page = page_num
            
            # Check for section patterns
            section_level, title = self._classify_section_line(line_clean)
            
            if section_level:
                section = Section(
                    level=section_level,
                    title=title,
                    content="",
                    page_start=current_page,
                    confidence=self._calculate_section_confidence(line_clean, section_level)
                )
                sections.append(section)
        
        return sections
    
    def _classify_section_line(self, line: str) -> Tuple[Optional[int], str]:
        """Classify a line as a section heading and determine its level"""
        line_clean = line.strip()
        
        # Check H1 patterns
        for pattern in self.section_patterns['h1']:
            match = re.match(pattern, line_clean, re.IGNORECASE)
            if match:
                title = match.group(1) if match.groups() else line_clean
                return 1, title.strip()
        
        # Check H2 patterns
        for pattern in self.section_patterns['h2']:
            match = re.match(pattern, line_clean, re.IGNORECASE)
            if match:
                title = match.group(1) if match.groups() else line_clean
                return 2, title.strip()
        
        # Check H3 patterns
        for pattern in self.section_patterns['h3']:
            match = re.match(pattern, line_clean, re.IGNORECASE)
            if match:
                title = match.group(1) if match.groups() else line_clean
                return 3, title.strip()
        
        return None, ""
    
    def _calculate_section_confidence(self, line: str, level: int) -> float:
        """Calculate confidence score for section detection"""
        confidence = 0.5  # Base confidence
        
        # Length-based confidence
        line_len = len(line.strip())
        if 10 <= line_len <= 100:
            confidence += 0.2
        elif line_len > 100:
            confidence -= 0.1
        
        # Formatting clues
        if line.isupper():
            confidence += 0.1
        if line.strip().endswith(':'):
            confidence += 0.1
        
        # Level-specific adjustments
        if level == 1 and line.isupper():
            confidence += 0.2
        
        return min(1.0, max(0.1, confidence))
    
    def _build_section_hierarchy(self, flat_sections: List[Section]) -> List[Section]:
        """Build hierarchical section structure from flat list"""
        if not flat_sections:
            return []
        
        root_sections = []
        section_stack = []
        
        for section in flat_sections:
            # Pop sections from stack until we find appropriate parent
            while section_stack and section_stack[-1].level >= section.level:
                section_stack.pop()
            
            if section_stack:
                # Add as subsection of current parent
                parent = section_stack[-1]
                section.parent_section = parent
                parent.subsections.append(section)
            else:
                # This is a root section
                root_sections.append(section)
            
            section_stack.append(section)
        
        return root_sections
    
    def _calculate_confidence_scores(self, sections: List[Section], page_markers: List[Tuple[int, int]], 
                                   document_type: DocumentType) -> Dict[str, float]:
        """Calculate overall confidence scores for the analysis"""
        scores = {}
        
        # Section detection confidence
        if sections:
            avg_section_confidence = sum(s.confidence for s in sections) / len(sections)
            scores['section_detection'] = avg_section_confidence
        else:
            scores['section_detection'] = 0.1
        
        # Page marker confidence
        if page_markers:
            scores['page_detection'] = min(1.0, len(page_markers) / 10.0)  # Normalize
        else:
            scores['page_detection'] = 0.1
        
        # Document type confidence
        if document_type != DocumentType.UNKNOWN:
            scores['document_classification'] = 0.8
        else:
            scores['document_classification'] = 0.3
        
        # Overall confidence (weighted average)
        weights = {
            'section_detection': 0.4,
            'page_detection': 0.3,
            'document_classification': 0.3
        }
        
        overall = sum(scores[key] * weights[key] for key in weights)
        scores['overall'] = overall
        
        return scores
    
    def _get_max_depth(self, sections: List[Section]) -> int:
        """Get maximum section depth in the hierarchy"""
        if not sections:
            return 0
        
        max_depth = 0
        for section in sections:
            max_depth = max(max_depth, section.level)
            if section.subsections:
                max_depth = max(max_depth, self._get_max_depth(section.subsections))
        
        return max_depth
    
    def extract_enhanced_metadata(self, content: str, filename: str = None) -> Dict[str, Any]:
        """
        Extract enhanced metadata using structure analysis
        
        Args:
            content: Document text content
            filename: Optional filename for context
            
        Returns:
            Enhanced metadata dictionary
        """
        structure = self.analyze_document_structure(content, filename)
        
        return {
            'document_type': structure.document_type.value,
            'page_count': structure.page_count,
            'section_count': len(structure.sections),
            'max_section_depth': structure.metadata['max_section_depth'],
            'confidence_scores': structure.confidence_scores,
            'sections_hierarchy': self._serialize_sections(structure.sections),
            'analysis_timestamp': self._get_timestamp()
        }
    
    def _serialize_sections(self, sections: List[Section]) -> List[Dict]:
        """Serialize sections hierarchy for JSON output"""
        serialized = []
        for section in sections:
            serialized_section = {
                'level': section.level,
                'title': section.title,
                'page_start': section.page_start,
                'page_end': section.page_end,
                'confidence': section.confidence,
                'subsections': self._serialize_sections(section.subsections)
            }
            serialized.append(serialized_section)
        return serialized
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata"""
        from datetime import datetime
        return datetime.utcnow().isoformat()


# Factory function
def create_document_structure_analyzer() -> DocumentStructureAnalyzer:
    """Factory function to create document structure analyzer"""
    return DocumentStructureAnalyzer()