#!/usr/bin/env python3
"""
ChunkQualityAnalyzer Service
Analyzes chunk quality and provides metrics for adaptive chunking
"""

import re
import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class ChunkQualityStatus(Enum):
    """Status of chunk quality assessment"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"

@dataclass
class ChunkQualityResult:
    """Result of chunk quality analysis"""
    status: ChunkQualityStatus
    quality_score: float
    quality_metrics: Dict[str, float]
    improvement_suggestions: List[str]
    adaptive_chunking_recommendations: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "status": self.status.value,
            "quality_score": self.quality_score,
            "quality_metrics": self.quality_metrics,
            "improvement_suggestions": self.improvement_suggestions,
            "adaptive_chunking_recommendations": self.adaptive_chunking_recommendations
        }

class ChunkQualityAnalyzer:
    """
    Analyzes chunk quality and provides metrics for adaptive chunking
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Quality metric weights
        self.quality_weights = {
            'semantic_coherence': 0.25,
            'structural_integrity': 0.20,
            'content_density': 0.15,
            'readability': 0.15,
            'context_preservation': 0.15,
            'boundary_quality': 0.10,
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            ChunkQualityStatus.EXCELLENT: 0.85,
            ChunkQualityStatus.GOOD: 0.70,
            ChunkQualityStatus.FAIR: 0.50,
            ChunkQualityStatus.POOR: 0.30,
            ChunkQualityStatus.UNUSABLE: 0.0,
        }
        
        # Patterns for analysis
        self.sentence_end_pattern = r'[.!?]+'
        self.section_header_patterns = [
            r'^#+\s+',  # Markdown headers
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS headers
            r'^\d+\.\s+',  # Numbered sections
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:',  # Title-like patterns
        ]

    def analyze_chunk_quality(self, 
                            chunk_text: str, 
                            chunk_size: int,
                            document_type: Optional[str] = None,
                            section_context: Optional[str] = None) -> ChunkQualityResult:
        """
        Analyze the quality of a text chunk
        
        Args:
            chunk_text: The text chunk to analyze
            chunk_size: Size of the chunk in characters
            document_type: Optional document type for context-aware analysis
            section_context: Optional section context for boundary analysis
            
        Returns:
            ChunkQualityResult with quality assessment
        """
        self.logger.info(f"Analyzing chunk quality for {len(chunk_text)} characters")
        
        quality_metrics = {}
        improvement_suggestions = []
        
        # Calculate quality metrics
        quality_metrics['semantic_coherence'] = self._calculate_semantic_coherence(chunk_text)
        quality_metrics['structural_integrity'] = self._calculate_structural_integrity(chunk_text)
        quality_metrics['content_density'] = self._calculate_content_density(chunk_text)
        quality_metrics['readability'] = self._calculate_readability(chunk_text)
        quality_metrics['context_preservation'] = self._calculate_context_preservation(chunk_text, section_context)
        quality_metrics['boundary_quality'] = self._calculate_boundary_quality(chunk_text, document_type)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(quality_metrics)
        
        # Determine quality status
        status = self._determine_quality_status(quality_score)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(quality_metrics, chunk_size)
        
        # Generate adaptive chunking recommendations
        adaptive_recommendations = self._generate_adaptive_chunking_recommendations(
            quality_metrics, chunk_size, document_type
        )
        
        return ChunkQualityResult(
            status=status,
            quality_score=quality_score,
            quality_metrics=quality_metrics,
            improvement_suggestions=improvement_suggestions,
            adaptive_chunking_recommendations=adaptive_recommendations
        )

    def _calculate_semantic_coherence(self, chunk_text: str) -> float:
        """Calculate semantic coherence score"""
        score = 0.0
        
        # Check for topic consistency
        sentences = re.split(self.sentence_end_pattern, chunk_text)
        if len(sentences) > 1:
            # Simple topic consistency check (first and last sentence similarity)
            first_sentence = sentences[0].strip().lower()
            last_sentence = sentences[-1].strip().lower()
            
            # Count common meaningful words
            first_words = set(re.findall(r'\b[a-z]{4,}\b', first_sentence))
            last_words = set(re.findall(r'\b[a-z]{4,}\b', last_sentence))
            
            if first_words and last_words:
                common_words = first_words.intersection(last_words)
                similarity = len(common_words) / min(len(first_words), len(last_words))
                score += similarity * 0.4
        
        # Check for logical connectors
        logical_connectors = [
            r'therefore', r'consequently', r'however', r'moreover', 
            r'furthermore', r'additionally', r'in contrast', r'similarly'
        ]
        
        connector_count = sum(1 for connector in logical_connectors 
                            if re.search(connector, chunk_text.lower()))
        score += min(connector_count * 0.1, 0.3)
        
        # Check for paragraph structure
        paragraph_count = chunk_text.count('\n\n') + 1
        if paragraph_count > 1:
            score += 0.2
        
        return min(score, 1.0)

    def _calculate_structural_integrity(self, chunk_text: str) -> float:
        """Calculate structural integrity score"""
        score = 0.0
        
        # Check for complete sentences
        sentences = re.split(self.sentence_end_pattern, chunk_text)
        complete_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if complete_sentences:
            sentence_completeness = len(complete_sentences) / len(sentences)
            score += sentence_completeness * 0.4
        
        # Check for proper punctuation
        punctuation_ratio = len(re.findall(r'[.!?]', chunk_text)) / max(1, len(chunk_text.split()))
        score += min(punctuation_ratio * 2, 0.3)
        
        # Check for paragraph breaks
        if '\n\n' in chunk_text:
            score += 0.2
        
        # Check for list structures
        list_indicators = re.findall(r'^\s*[\-\*•]\s+', chunk_text, re.MULTILINE)
        if list_indicators:
            score += 0.1
        
        return min(score, 1.0)

    def _calculate_content_density(self, chunk_text: str) -> float:
        """Calculate content density score"""
        # Remove common stop words and count meaningful content
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        words = re.findall(r'\b[a-z]+\b', chunk_text.lower())
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        if not words:
            return 0.0
        
        content_density = len(meaningful_words) / len(words)
        
        # Adjust for very short chunks
        if len(words) < 20:
            content_density *= 0.8
        
        return min(content_density, 1.0)

    def _calculate_readability(self, chunk_text: str) -> float:
        """Calculate readability score using simple metrics"""
        score = 0.0
        
        # Average sentence length (optimal range: 15-25 words)
        sentences = re.split(self.sentence_end_pattern, chunk_text)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / len(sentences)
            if 15 <= avg_sentence_length <= 25:
                score += 0.4
            elif 10 <= avg_sentence_length <= 30:
                score += 0.2
        
        # Word length distribution
        words = re.findall(r'\b[a-zA-Z]+\b', chunk_text)
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length <= 6:
                score += 0.3
            elif avg_word_length <= 8:
                score += 0.2
        
        # Paragraph length (optimal: 3-5 sentences)
        paragraphs = chunk_text.split('\n\n')
        if paragraphs:
            valid_paragraphs = sum(1 for p in paragraphs if len(p.strip().split('.')) >= 2)
            paragraph_quality = valid_paragraphs / len(paragraphs)
            score += paragraph_quality * 0.3
        
        return min(score, 1.0)

    def _calculate_context_preservation(self, chunk_text: str, section_context: Optional[str]) -> float:
        """Calculate context preservation score"""
        score = 0.0
        
        if section_context:
            # Check if section context is preserved in chunk
            context_words = set(re.findall(r'\b[a-z]{4,}\b', section_context.lower()))
            chunk_words = set(re.findall(r'\b[a-z]{4,}\b', chunk_text.lower()))
            
            if context_words and chunk_words:
                overlap = len(context_words.intersection(chunk_words))
                context_preservation = overlap / len(context_words)
                score += min(context_preservation, 0.6)
        
        # Check for contextual references
        context_indicators = [
            r'as mentioned', r'previously', r'earlier', r'later', 
            r'in this section', r'as discussed', r'according to'
        ]
        
        indicator_count = sum(1 for indicator in context_indicators 
                            if re.search(indicator, chunk_text.lower()))
        score += min(indicator_count * 0.1, 0.4)
        
        return min(score, 1.0)

    def _calculate_boundary_quality(self, chunk_text: str, document_type: Optional[str]) -> float:
        """Calculate boundary quality score"""
        score = 0.0
        
        # Check for natural boundaries at start
        starts_with_capital = chunk_text.strip()[0].isupper() if chunk_text.strip() else False
        if starts_with_capital:
            score += 0.2
        
        # Check for section headers at boundaries
        header_at_start = any(re.search(pattern, chunk_text.strip()) 
                            for pattern in self.section_header_patterns)
        if header_at_start:
            score += 0.3
        
        # Check for complete sentences at boundaries
        sentences = re.split(self.sentence_end_pattern, chunk_text)
        if sentences and len(sentences[0].strip().split()) >= 3:
            score += 0.2
        
        # Document-type specific boundary checks
        if document_type == 'academic':
            # Academic papers often have structured sections
            academic_patterns = [r'abstract', r'introduction', r'methodology', r'results', r'discussion']
            if any(re.search(pattern, chunk_text.lower()) for pattern in academic_patterns):
                score += 0.3
        
        return min(score, 1.0)

    def _calculate_quality_score(self, quality_metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from metrics"""
        total_score = 0.0
        
        for metric, weight in self.quality_weights.items():
            if metric in quality_metrics:
                total_score += quality_metrics[metric] * weight
        
        return min(total_score, 1.0)

    def _determine_quality_status(self, quality_score: float) -> ChunkQualityStatus:
        """Determine quality status based on score"""
        for status, threshold in self.quality_thresholds.items():
            if quality_score >= threshold:
                return status
        return ChunkQualityStatus.UNUSABLE

    def _generate_improvement_suggestions(self, 
                                        quality_metrics: Dict[str, float],
                                        chunk_size: int) -> List[str]:
        """Generate improvement suggestions based on quality metrics"""
        suggestions = []
        
        # Semantic coherence suggestions
        if quality_metrics['semantic_coherence'] < 0.6:
            suggestions.append("Improve topic consistency throughout the chunk")
            suggestions.append("Add logical connectors between sentences")
        
        # Structural integrity suggestions
        if quality_metrics['structural_integrity'] < 0.5:
            suggestions.append("Ensure complete sentences with proper punctuation")
            suggestions.append("Add paragraph breaks for better structure")
        
        # Content density suggestions
        if quality_metrics['content_density'] < 0.4:
            suggestions.append("Increase meaningful content relative to common words")
            suggestions.append("Remove redundant or repetitive phrases")
        
        # Readability suggestions
        if quality_metrics['readability'] < 0.5:
            suggestions.append("Adjust sentence lengths for better readability")
            suggestions.append("Simplify complex sentence structures")
        
        # Context preservation suggestions
        if quality_metrics['context_preservation'] < 0.4:
            suggestions.append("Include more contextual references")
            suggestions.append("Maintain connection to section themes")
        
        # Boundary quality suggestions
        if quality_metrics['boundary_quality'] < 0.4:
            suggestions.append("Start chunks at natural boundaries like section headers")
            suggestions.append("Ensure chunks begin with complete sentences")
        
        # Chunk size suggestions
        if chunk_size < 200:
            suggestions.append("Consider merging with adjacent chunks for better context")
        elif chunk_size > 1500:
            suggestions.append("Consider splitting into smaller, more focused chunks")
        
        return suggestions

    def _generate_adaptive_chunking_recommendations(self,
                                                  quality_metrics: Dict[str, float],
                                                  chunk_size: int,
                                                  document_type: Optional[str]) -> Dict[str, Any]:
        """Generate adaptive chunking recommendations"""
        recommendations = {
            'optimal_chunk_size': chunk_size,
            'chunking_strategy': 'fixed',
            'boundary_preferences': [],
            'merge_suggestions': [],
            'split_suggestions': []
        }
        
        # Adjust chunk size based on quality metrics
        if quality_metrics['semantic_coherence'] < 0.5:
            recommendations['optimal_chunk_size'] = max(300, chunk_size - 200)
            recommendations['chunking_strategy'] = 'semantic'
        
        if quality_metrics['content_density'] > 0.7:
            # High density content can handle larger chunks
            recommendations['optimal_chunk_size'] = min(2000, chunk_size + 300)
        
        # Boundary preferences
        if quality_metrics['boundary_quality'] < 0.6:
            recommendations['boundary_preferences'].extend([
                'section_headers',
                'paragraph_breaks', 
                'complete_sentences'
            ])
        
        # Merge suggestions for small, low-quality chunks
        if chunk_size < 300 and quality_metrics['semantic_coherence'] < 0.4:
            recommendations['merge_suggestions'].append(
                "Merge with adjacent chunks for better context"
            )
        
        # Split suggestions for large, complex chunks
        if chunk_size > 1200 and quality_metrics['readability'] < 0.5:
            recommendations['split_suggestions'].append(
                "Split at natural boundaries for better focus"
            )
        
        # Document-type specific recommendations
        if document_type == 'academic':
            recommendations['boundary_preferences'].extend([
                'section_titles',
                'subsection_headers',
                'theorem_statements'
            ])
        elif document_type == 'legal':
            recommendations['boundary_preferences'].extend([
                'clause_boundaries',
                'section_numbers',
                'legal_definitions'
            ])
        
        return recommendations

    def batch_analyze_chunk_quality(self, 
                                  chunks_data: List[Dict[str, Any]]) -> List[ChunkQualityResult]:
        """
        Analyze quality of multiple chunks in batch
        
        Args:
            chunks_data: List of chunk data dictionaries
            
        Returns:
            List of quality analysis results
        """
        results = []
        
        for chunk_data in chunks_data:
            try:
                result = self.analyze_chunk_quality(
                    chunk_text=chunk_data.get('chunk_text', ''),
                    chunk_size=chunk_data.get('chunk_size', 0),
                    document_type=chunk_data.get('document_type'),
                    section_context=chunk_data.get('section_context')
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error analyzing chunk quality: {e}")
                # Create a failed analysis result
                results.append(ChunkQualityResult(
                    status=ChunkQualityStatus.UNUSABLE,
                    quality_score=0.0,
                    quality_metrics={},
                    improvement_suggestions=['Chunk quality analysis failed'],
                    adaptive_chunking_recommendations={}
                ))
        
        return results

    def get_chunk_quality_report(self,
                               quality_results: List[ChunkQualityResult]) -> Dict[str, Any]:
        """
        Generate a quality report for multiple chunks
        
        Args:
            quality_results: List of quality analysis results
            
        Returns:
            Quality report dictionary
        """
        if not quality_results:
            return {}
        
        total_chunks = len(quality_results)
        status_counts = {
            status.value: 0 for status in ChunkQualityStatus
        }
        
        average_quality = 0.0
        metric_averages = {}
        
        for result in quality_results:
            status_counts[result.status.value] += 1
            average_quality += result.quality_score
            
            # Calculate metric averages
            for metric, value in result.quality_metrics.items():
                if metric not in metric_averages:
                    metric_averages[metric] = 0.0
                metric_averages[metric] += value
        
        average_quality /= total_chunks
        
        # Calculate final metric averages
        for metric in metric_averages:
            metric_averages[metric] /= total_chunks
        
        # Generate improvement summary
        all_suggestions = []
        for result in quality_results:
            all_suggestions.extend(result.improvement_suggestions)
        
        common_suggestions = []
        suggestion_counts = {}
        for suggestion in all_suggestions:
            suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1
        
        # Get top 5 most common suggestions
        common_suggestions = sorted(
            suggestion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'total_chunks': total_chunks,
            'status_distribution': status_counts,
            'average_quality_score': average_quality,
            'metric_averages': metric_averages,
            'common_improvement_suggestions': common_suggestions,
            'overall_quality_grade': self._calculate_quality_grade(average_quality),
            'report_timestamp': datetime.now().isoformat()
        }

    def _calculate_quality_grade(self, quality_score: float) -> str:
        """Calculate quality grade from score"""
        if quality_score >= 0.85:
            return "A"
        elif quality_score >= 0.70:
            return "B"
        elif quality_score >= 0.50:
            return "C"
        elif quality_score >= 0.30:
            return "D"
        else:
            return "F"


def create_chunk_quality_analyzer() -> ChunkQualityAnalyzer:
    """Factory function to create ChunkQualityAnalyzer instance"""
    return ChunkQualityAnalyzer()