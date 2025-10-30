#!/usr/bin/env python3
"""
CitationValidator Service
Validates and scores citation accuracy for search results
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class CitationValidationStatus(Enum):
    """Status of citation validation"""
    VALID = "valid"
    PARTIAL = "partial"
    INVALID = "invalid"
    UNVERIFIABLE = "unverifiable"

@dataclass
class CitationValidationResult:
    """Result of citation validation"""
    status: CitationValidationStatus
    confidence_score: float
    validation_details: Dict[str, Any]
    suggestions: List[str]
    source_verification: Dict[str, bool]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "status": self.status.value,
            "confidence_score": self.confidence_score,
            "validation_details": self.validation_details,
            "suggestions": self.suggestions,
            "source_verification": self.source_verification
        }

class CitationValidator:
    """
    Validates citation accuracy and provides quality scoring
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Citation patterns for different document types
        self.citation_patterns = {
            'academic': [
                r'\(\s*(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+)?\d{4}(?:\s*,\s*p\.?\s*\d+(?:-\d+)?)?\s*\)',
                r'\[(?:\d+(?:,\s*\d+)*)\]',
                r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\(\d{4}\))',
            ],
            'legal': [
                r'\d+\s+[A-Z]+\s+\d+',
                r'[A-Z]+\s+No\.?\s+\d+',
                r'\d+\s+U\.?S\.?\s+\d+',
                r'\d+\s+F\.?(?:2d|3d|4th)?\s+\d+',
            ],
            'technical': [
                r'Section\s+\d+(?:\.\d+)*',
                r'Chapter\s+\d+',
                r'Table\s+\d+(?:-\d+)?',
                r'Figure\s+\d+(?:-\d+)?',
            ]
        }
        
        # Source verification patterns
        self.source_patterns = {
            'document_name': r'[A-Za-z0-9_\-\.]+\.(?:pdf|docx?|txt|md)',
            'page_number': r'(?:p\.?\s*)?\d+(?:\s*-\s*\d+)?',
            'section_title': r'[A-Z][A-Za-z0-9\s\-_]+',
        }
        
        # Quality scoring weights
        self.quality_weights = {
            'source_presence': 0.3,
            'citation_format': 0.25,
            'context_relevance': 0.2,
            'position_accuracy': 0.15,
            'completeness': 0.1,
        }

    def validate_citation(self, 
                         chunk_text: str, 
                         source_document: str, 
                         page_numbers: List[int],
                         section_title: Optional[str] = None) -> CitationValidationResult:
        """
        Validate a citation for accuracy and completeness
        
        Args:
            chunk_text: The text chunk containing the citation
            source_document: Name of the source document
            page_numbers: List of page numbers where content appears
            section_title: Optional section title for context
            
        Returns:
            CitationValidationResult with validation details
        """
        self.logger.info(f"Validating citation for document: {source_document}")
        
        validation_details = {}
        suggestions = []
        source_verification = {}
        
        # 1. Check source document presence
        source_presence_score = self._validate_source_presence(chunk_text, source_document)
        validation_details['source_presence'] = source_presence_score
        
        # 2. Check citation format
        citation_format_score = self._validate_citation_format(chunk_text)
        validation_details['citation_format'] = citation_format_score
        
        # 3. Check context relevance
        context_relevance_score = self._validate_context_relevance(chunk_text)
        validation_details['context_relevance'] = context_relevance_score
        
        # 4. Check position accuracy
        position_accuracy_score = self._validate_position_accuracy(chunk_text, page_numbers, section_title)
        validation_details['position_accuracy'] = position_accuracy_score
        
        # 5. Check completeness
        completeness_score = self._validate_completeness(chunk_text, source_document, page_numbers, section_title)
        validation_details['completeness'] = completeness_score
        
        # 6. Verify sources
        source_verification = self._verify_sources(chunk_text, source_document, page_numbers, section_title)
        
        # 7. Generate suggestions
        suggestions = self._generate_suggestions(validation_details, source_verification)
        
        # 8. Calculate overall confidence score
        confidence_score = self._calculate_confidence_score(validation_details)
        
        # 9. Determine validation status
        status = self._determine_validation_status(confidence_score, validation_details)
        
        return CitationValidationResult(
            status=status,
            confidence_score=confidence_score,
            validation_details=validation_details,
            suggestions=suggestions,
            source_verification=source_verification
        )

    def _validate_source_presence(self, chunk_text: str, source_document: str) -> float:
        """Validate that source document is properly referenced"""
        score = 0.0
        
        # Check for document name mentions
        doc_name_clean = source_document.lower().replace('.pdf', '').replace('.docx', '')
        doc_name_patterns = [
            re.escape(doc_name_clean),
            re.escape(source_document),
            r'source\s+(?:document|file)',
            r'from\s+(?:the\s+)?document',
        ]
        
        for pattern in doc_name_patterns:
            if re.search(pattern, chunk_text.lower()):
                score += 0.25
        
        # Check for explicit source attribution
        attribution_patterns = [
            r'according\s+to',
            r'as\s+stated\s+in',
            r'per\s+the\s+document',
            r'based\s+on',
            r'source:',
        ]
        
        for pattern in attribution_patterns:
            if re.search(pattern, chunk_text.lower()):
                score += 0.25
        
        return min(score, 1.0)

    def _validate_citation_format(self, chunk_text: str) -> float:
        """Validate citation format and style"""
        score = 0.0
        
        # Check for academic citation patterns
        for pattern in self.citation_patterns['academic']:
            if re.search(pattern, chunk_text):
                score += 0.3
                break
        
        # Check for legal citation patterns
        for pattern in self.citation_patterns['legal']:
            if re.search(pattern, chunk_text):
                score += 0.3
                break
        
        # Check for technical citation patterns
        for pattern in self.citation_patterns['technical']:
            if re.search(pattern, chunk_text):
                score += 0.2
                break
        
        # Check for general citation indicators
        general_patterns = [
            r'\([^)]+\)',  # Parenthetical citations
            r'\[[^\]]+\]',  # Bracket citations
            r'page\s+\d+',
            r'p\.\s*\d+',
        ]
        
        for pattern in general_patterns:
            if re.search(pattern, chunk_text.lower()):
                score += 0.2
                break
        
        return min(score, 1.0)

    def _validate_context_relevance(self, chunk_text: str) -> float:
        """Validate that citation context is relevant"""
        score = 0.0
        
        # Check for contextual indicators
        context_indicators = [
            r'in\s+the\s+context\s+of',
            r'regarding',
            r'with\s+respect\s+to',
            r'in\s+relation\s+to',
            r'as\s+discussed\s+in',
            r'as\s+mentioned\s+in',
        ]
        
        for pattern in context_indicators:
            if re.search(pattern, chunk_text.lower()):
                score += 0.3
        
        # Check for logical connectors
        logical_connectors = [
            r'therefore',
            r'consequently',
            r'accordingly',
            r'thus',
            r'hence',
        ]
        
        for pattern in logical_connectors:
            if re.search(pattern, chunk_text.lower()):
                score += 0.2
        
        # Check for evidence indicators
        evidence_indicators = [
            r'evidence\s+suggests',
            r'studies\s+show',
            r'research\s+indicates',
            r'data\s+demonstrates',
        ]
        
        for pattern in evidence_indicators:
            if re.search(pattern, chunk_text.lower()):
                score += 0.3
        
        return min(score, 1.0)

    def _validate_position_accuracy(self, 
                                  chunk_text: str, 
                                  page_numbers: List[int],
                                  section_title: Optional[str]) -> float:
        """Validate position accuracy (page numbers, sections)"""
        score = 0.0
        
        # Check for page number references
        if page_numbers:
            page_ref_pattern = r'(?:page|p\.?)\s*(\d+)'
            matches = re.findall(page_ref_pattern, chunk_text.lower())
            
            if matches:
                referenced_pages = [int(match) for match in matches if match.isdigit()]
                matching_pages = [page for page in referenced_pages if page in page_numbers]
                
                if matching_pages:
                    score += 0.4
                elif referenced_pages:
                    score += 0.2  # Partial credit for referencing pages
        
        # Check for section title references
        if section_title:
            section_clean = re.escape(section_title.lower())
            if re.search(section_clean, chunk_text.lower()):
                score += 0.3
            
            # Check for section-like patterns
            section_patterns = [
                r'section\s+[A-Za-z0-9\.\-]+',
                r'chapter\s+[A-Za-z0-9\.\-]+',
                r'part\s+[A-Za-z0-9\.\-]+',
            ]
            
            for pattern in section_patterns:
                if re.search(pattern, chunk_text.lower()):
                    score += 0.2
                    break
        
        # Check for position indicators
        position_indicators = [
            r'in\s+the\s+(?:beginning|middle|end)',
            r'earlier\s+in',
            r'later\s+in',
            r'previously\s+mentioned',
        ]
        
        for pattern in position_indicators:
            if re.search(pattern, chunk_text.lower()):
                score += 0.1
        
        return min(score, 1.0)

    def _validate_completeness(self, 
                             chunk_text: str, 
                             source_document: str,
                             page_numbers: List[int],
                             section_title: Optional[str]) -> float:
        """Validate citation completeness"""
        score = 0.0
        completeness_factors = 0
        total_factors = 3  # document, pages, section
        
        # Document reference
        if self._validate_source_presence(chunk_text, source_document) > 0:
            completeness_factors += 1
        
        # Page reference
        if page_numbers and re.search(r'(?:page|p\.?)\s*\d+', chunk_text.lower()):
            completeness_factors += 1
        
        # Section reference
        if section_title and (re.search(re.escape(section_title.lower()), chunk_text.lower()) or 
                             re.search(r'section\s+[A-Za-z0-9\.\-]+', chunk_text.lower())):
            completeness_factors += 1
        
        score = completeness_factors / total_factors
        return score

    def _verify_sources(self, 
                       chunk_text: str, 
                       source_document: str,
                       page_numbers: List[int],
                       section_title: Optional[str]) -> Dict[str, bool]:
        """Verify source information presence"""
        verification = {}
        
        # Verify document name
        verification['document_name'] = bool(
            re.search(re.escape(source_document.lower()), chunk_text.lower()) or
            re.search(r'source\s+(?:document|file)', chunk_text.lower())
        )
        
        # Verify page numbers
        verification['page_numbers'] = bool(
            page_numbers and
            re.search(r'(?:page|p\.?)\s*\d+', chunk_text.lower())
        )
        
        # Verify section title
        verification['section_title'] = bool(
            section_title and 
            re.search(re.escape(section_title.lower()), chunk_text.lower())
        )
        
        # Verify citation format
        verification['citation_format'] = bool(
            self._validate_citation_format(chunk_text) > 0.3
        )
        
        return verification

    def _generate_suggestions(self, 
                            validation_details: Dict[str, float],
                            source_verification: Dict[str, bool]) -> List[str]:
        """Generate improvement suggestions based on validation results"""
        suggestions = []
        
        # Source presence suggestions
        if validation_details['source_presence'] < 0.5:
            suggestions.append("Include explicit reference to the source document")
        
        # Citation format suggestions
        if validation_details['citation_format'] < 0.4:
            suggestions.append("Use standard citation format (e.g., (Author, Year) or [1])")
        
        # Context relevance suggestions
        if validation_details['context_relevance'] < 0.5:
            suggestions.append("Provide more context about how the citation supports the claim")
        
        # Position accuracy suggestions
        if validation_details['position_accuracy'] < 0.5:
            suggestions.append("Include specific page numbers or section references")
        
        # Completeness suggestions
        if validation_details['completeness'] < 0.7:
            missing_sources = [k for k, v in source_verification.items() if not v]
            if missing_sources:
                suggestions.append(f"Add references for: {', '.join(missing_sources)}")
        
        return suggestions

    def _calculate_confidence_score(self, validation_details: Dict[str, float]) -> float:
        """Calculate overall confidence score"""
        total_score = 0.0
        
        for factor, weight in self.quality_weights.items():
            if factor in validation_details:
                total_score += validation_details[factor] * weight
        
        return min(total_score, 1.0)

    def _determine_validation_status(self, 
                                   confidence_score: float,
                                   validation_details: Dict[str, float]) -> CitationValidationStatus:
        """Determine validation status based on confidence score and details"""
        
        if confidence_score >= 0.8:
            return CitationValidationStatus.VALID
        elif confidence_score >= 0.6:
            return CitationValidationStatus.PARTIAL
        elif confidence_score >= 0.3:
            return CitationValidationStatus.UNVERIFIABLE
        else:
            return CitationValidationStatus.INVALID

    def batch_validate_citations(self, 
                               citations_data: List[Dict[str, Any]]) -> List[CitationValidationResult]:
        """
        Validate multiple citations in batch
        
        Args:
            citations_data: List of citation data dictionaries
            
        Returns:
            List of validation results
        """
        results = []
        
        for citation_data in citations_data:
            try:
                result = self.validate_citation(
                    chunk_text=citation_data.get('chunk_text', ''),
                    source_document=citation_data.get('source_document', ''),
                    page_numbers=citation_data.get('page_numbers', []),
                    section_title=citation_data.get('section_title')
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error validating citation: {e}")
                # Create a failed validation result
                results.append(CitationValidationResult(
                    status=CitationValidationStatus.INVALID,
                    confidence_score=0.0,
                    validation_details={'error': str(e)},
                    suggestions=['Citation validation failed due to error'],
                    source_verification={}
                ))
        
        return results

    def get_citation_quality_report(self, 
                                  validation_results: List[CitationValidationResult]) -> Dict[str, Any]:
        """
        Generate a quality report for multiple citations
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Quality report dictionary
        """
        if not validation_results:
            return {}
        
        total_citations = len(validation_results)
        status_counts = {
            status.value: 0 for status in CitationValidationStatus
        }
        
        average_confidence = 0.0
        suggestion_frequency = {}
        
        for result in validation_results:
            status_counts[result.status.value] += 1
            average_confidence += result.confidence_score
            
            # Count suggestion frequency
            for suggestion in result.suggestions:
                suggestion_frequency[suggestion] = suggestion_frequency.get(suggestion, 0) + 1
        
        average_confidence /= total_citations
        
        # Sort suggestions by frequency
        common_suggestions = sorted(
            suggestion_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 suggestions
        
        return {
            'total_citations': total_citations,
            'status_distribution': status_counts,
            'average_confidence': average_confidence,
            'common_suggestions': common_suggestions,
            'quality_score': average_confidence * 100,  # Convert to percentage
            'report_timestamp': datetime.now().isoformat()
        }


def create_citation_validator() -> CitationValidator:
    """Factory function to create CitationValidator instance"""
    return CitationValidator()
