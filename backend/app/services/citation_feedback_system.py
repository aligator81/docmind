#!/usr/bin/env python3
"""
CitationFeedbackSystem Service
Collects and analyzes user feedback on citation quality
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Type of user feedback"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class FeedbackCategory(Enum):
    """Category of feedback"""
    CITATION_ACCURACY = "citation_accuracy"
    SOURCE_VERIFICATION = "source_verification"
    CONTEXT_RELEVANCE = "context_relevance"
    POSITION_ACCURACY = "position_accuracy"
    COMPLETENESS = "completeness"
    FORMAT_QUALITY = "format_quality"

@dataclass
class CitationFeedback:
    """Individual citation feedback entry"""
    feedback_id: str
    citation_id: str
    user_id: str
    feedback_type: FeedbackType
    category: FeedbackCategory
    rating: int  # 1-5 scale
    comment: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "feedback_id": self.feedback_id,
            "citation_id": self.citation_id,
            "user_id": self.user_id,
            "feedback_type": self.feedback_type.value,
            "category": self.category.value,
            "rating": self.rating,
            "comment": self.comment,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class FeedbackAnalysis:
    """Analysis of feedback data"""
    total_feedback: int
    average_rating: float
    feedback_distribution: Dict[FeedbackType, int]
    category_analysis: Dict[FeedbackCategory, Dict[str, Any]]
    improvement_priorities: List[Tuple[FeedbackCategory, float]]
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "total_feedback": self.total_feedback,
            "average_rating": self.average_rating,
            "feedback_distribution": {k.value: v for k, v in self.feedback_distribution.items()},
            "category_analysis": {k.value: v for k, v in self.category_analysis.items()},
            "improvement_priorities": [(cat.value, score) for cat, score in self.improvement_priorities],
            "confidence_score": self.confidence_score
        }

class CitationFeedbackSystem:
    """
    Collects and analyzes user feedback on citation quality
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feedback_storage: Dict[str, CitationFeedback] = {}
        
        # Feedback analysis parameters
        self.min_feedback_threshold = 5
        self.rating_weights = {
            1: -2.0,  # Strongly negative
            2: -1.0,  # Negative
            3: 0.0,   # Neutral
            4: 1.0,   # Positive
            5: 2.0    # Strongly positive
        }
        
        # Category importance weights
        self.category_weights = {
            FeedbackCategory.CITATION_ACCURACY: 0.25,
            FeedbackCategory.SOURCE_VERIFICATION: 0.20,
            FeedbackCategory.CONTEXT_RELEVANCE: 0.20,
            FeedbackCategory.POSITION_ACCURACY: 0.15,
            FeedbackCategory.COMPLETENESS: 0.10,
            FeedbackCategory.FORMAT_QUALITY: 0.10,
        }

    def submit_feedback(self,
                       citation_id: str,
                       user_id: str,
                       feedback_type: FeedbackType,
                       category: FeedbackCategory,
                       rating: int,
                       comment: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit user feedback for a citation
        
        Args:
            citation_id: ID of the citation being rated
            user_id: ID of the user providing feedback
            feedback_type: Type of feedback (positive/negative/neutral)
            category: Category of feedback
            rating: Rating on 1-5 scale
            comment: Optional comment from user
            metadata: Optional additional metadata
            
        Returns:
            Feedback ID for tracking
        """
        feedback_id = f"feedback_{citation_id}_{user_id}_{datetime.now().timestamp()}"
        
        feedback = CitationFeedback(
            feedback_id=feedback_id,
            citation_id=citation_id,
            user_id=user_id,
            feedback_type=feedback_type,
            category=category,
            rating=rating,
            comment=comment,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.feedback_storage[feedback_id] = feedback
        self.logger.info(f"Feedback submitted: {feedback_id} for citation {citation_id}")
        
        return feedback_id

    def get_feedback_for_citation(self, citation_id: str) -> List[CitationFeedback]:
        """
        Get all feedback for a specific citation
        
        Args:
            citation_id: ID of the citation
            
        Returns:
            List of feedback entries
        """
        feedback_list = [
            feedback for feedback in self.feedback_storage.values()
            if feedback.citation_id == citation_id
        ]
        
        return sorted(feedback_list, key=lambda x: x.timestamp, reverse=True)

    def get_feedback_by_user(self, user_id: str) -> List[CitationFeedback]:
        """
        Get all feedback from a specific user
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of feedback entries
        """
        feedback_list = [
            feedback for feedback in self.feedback_storage.values()
            if feedback.user_id == user_id
        ]
        
        return sorted(feedback_list, key=lambda x: x.timestamp, reverse=True)

    def analyze_feedback(self, 
                        citation_id: Optional[str] = None,
                        time_period: Optional[timedelta] = None) -> FeedbackAnalysis:
        """
        Analyze feedback data for citations
        
        Args:
            citation_id: Optional specific citation to analyze
            time_period: Optional time period to analyze
            
        Returns:
            FeedbackAnalysis with insights
        """
        # Filter feedback based on parameters
        feedback_list = list(self.feedback_storage.values())
        
        if citation_id:
            feedback_list = [f for f in feedback_list if f.citation_id == citation_id]
        
        if time_period:
            cutoff_time = datetime.now() - time_period
            feedback_list = [f for f in feedback_list if f.timestamp >= cutoff_time]
        
        if not feedback_list:
            return self._create_empty_analysis()
        
        # Calculate basic statistics
        total_feedback = len(feedback_list)
        ratings = [f.rating for f in feedback_list]
        average_rating = statistics.mean(ratings) if ratings else 0.0
        
        # Feedback distribution by type
        feedback_distribution = {
            FeedbackType.POSITIVE: 0,
            FeedbackType.NEGATIVE: 0,
            FeedbackType.NEUTRAL: 0
        }
        
        for feedback in feedback_list:
            feedback_distribution[feedback.feedback_type] += 1
        
        # Category analysis
        category_analysis = self._analyze_categories(feedback_list)
        
        # Improvement priorities
        improvement_priorities = self._calculate_improvement_priorities(category_analysis)
        
        # Confidence score
        confidence_score = self._calculate_confidence_score(total_feedback, average_rating, feedback_distribution)
        
        return FeedbackAnalysis(
            total_feedback=total_feedback,
            average_rating=average_rating,
            feedback_distribution=feedback_distribution,
            category_analysis=category_analysis,
            improvement_priorities=improvement_priorities,
            confidence_score=confidence_score
        )

    def _analyze_categories(self, feedback_list: List[CitationFeedback]) -> Dict[FeedbackCategory, Dict[str, Any]]:
        """Analyze feedback by category"""
        category_analysis = {}
        
        for category in FeedbackCategory:
            category_feedback = [f for f in feedback_list if f.category == category]
            
            if not category_feedback:
                category_analysis[category] = {
                    'count': 0,
                    'average_rating': 0.0,
                    'weighted_score': 0.0,
                    'feedback_types': {ft.value: 0 for ft in FeedbackType}
                }
                continue
            
            ratings = [f.rating for f in category_feedback]
            average_rating = statistics.mean(ratings)
            
            # Calculate weighted score
            weighted_scores = [self.rating_weights[f.rating] for f in category_feedback]
            weighted_score = statistics.mean(weighted_scores) if weighted_scores else 0.0
            
            # Feedback type distribution
            feedback_types = {ft.value: 0 for ft in FeedbackType}
            for feedback in category_feedback:
                feedback_types[feedback.feedback_type.value] += 1
            
            category_analysis[category] = {
                'count': len(category_feedback),
                'average_rating': average_rating,
                'weighted_score': weighted_score,
                'feedback_types': feedback_types
            }
        
        return category_analysis

    def _calculate_improvement_priorities(self, 
                                        category_analysis: Dict[FeedbackCategory, Dict[str, Any]]) -> List[Tuple[FeedbackCategory, float]]:
        """Calculate improvement priorities based on feedback"""
        priorities = []
        
        for category, analysis in category_analysis.items():
            if analysis['count'] == 0:
                continue
            
            # Calculate priority score (lower weighted score = higher priority)
            weighted_score = analysis['weighted_score']
            category_weight = self.category_weights[category]
            
            # Invert score so lower scores become higher priorities
            priority_score = (1.0 - (weighted_score + 2.0) / 4.0) * category_weight
            
            priorities.append((category, priority_score))
        
        # Sort by priority score (descending)
        priorities.sort(key=lambda x: x[1], reverse=True)
        
        return priorities

    def _calculate_confidence_score(self, 
                                  total_feedback: int,
                                  average_rating: float,
                                  feedback_distribution: Dict[FeedbackType, int]) -> float:
        """Calculate confidence score for feedback analysis"""
        if total_feedback == 0:
            return 0.0
        
        # Base confidence from feedback volume
        volume_confidence = min(total_feedback / self.min_feedback_threshold, 1.0)
        
        # Rating confidence (higher average rating = higher confidence)
        rating_confidence = (average_rating - 1.0) / 4.0  # Normalize to 0-1
        
        # Distribution confidence (balanced feedback = higher confidence)
        positive_ratio = feedback_distribution[FeedbackType.POSITIVE] / total_feedback
        negative_ratio = feedback_distribution[FeedbackType.NEGATIVE] / total_feedback
        
        # Balanced feedback is more reliable than extreme polarization
        distribution_confidence = 1.0 - abs(positive_ratio - negative_ratio)
        
        # Combine confidence factors
        confidence_score = (
            volume_confidence * 0.4 +
            rating_confidence * 0.3 +
            distribution_confidence * 0.3
        )
        
        return min(confidence_score, 1.0)

    def _create_empty_analysis(self) -> FeedbackAnalysis:
        """Create empty analysis when no feedback exists"""
        return FeedbackAnalysis(
            total_feedback=0,
            average_rating=0.0,
            feedback_distribution={ft: 0 for ft in FeedbackType},
            category_analysis={cat: {
                'count': 0,
                'average_rating': 0.0,
                'weighted_score': 0.0,
                'feedback_types': {ft.value: 0 for ft in FeedbackType}
            } for cat in FeedbackCategory},
            improvement_priorities=[],
            confidence_score=0.0
        )

    def get_improvement_recommendations(self, 
                                      feedback_analysis: FeedbackAnalysis) -> List[Dict[str, Any]]:
        """
        Generate improvement recommendations based on feedback analysis
        
        Args:
            feedback_analysis: Analysis of feedback data
            
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        
        if feedback_analysis.total_feedback == 0:
            recommendations.append({
                'priority': 'high',
                'category': 'general',
                'recommendation': 'Collect more user feedback to improve citation quality',
                'reason': 'Insufficient feedback data available'
            })
            return recommendations
        
        # Generate recommendations based on improvement priorities
        for category, priority_score in feedback_analysis.improvement_priorities[:3]:  # Top 3 priorities
            if priority_score > 0.1:  # Only include significant priorities
                category_analysis = feedback_analysis.category_analysis[category]
                
                recommendation = self._generate_category_recommendation(category, category_analysis, priority_score)
                recommendations.append(recommendation)
        
        # Add general recommendations based on overall feedback
        if feedback_analysis.average_rating < 3.0:
            recommendations.append({
                'priority': 'medium',
                'category': 'general',
                'recommendation': 'Focus on improving overall citation quality',
                'reason': f'Average rating is low ({feedback_analysis.average_rating:.1f}/5.0)'
            })
        
        return recommendations

    def _generate_category_recommendation(self, 
                                        category: FeedbackCategory,
                                        category_analysis: Dict[str, Any],
                                        priority_score: float) -> Dict[str, Any]:
        """Generate specific recommendation for a category"""
        priority_level = 'high' if priority_score > 0.3 else 'medium' if priority_score > 0.15 else 'low'
        
        recommendations_map = {
            FeedbackCategory.CITATION_ACCURACY: {
                'recommendation': 'Improve citation accuracy and verification',
                'reason': f'Low weighted score ({category_analysis["weighted_score"]:.2f}) in citation accuracy'
            },
            FeedbackCategory.SOURCE_VERIFICATION: {
                'recommendation': 'Enhance source document verification',
                'reason': f'Poor performance in source verification (rating: {category_analysis["average_rating"]:.1f})'
            },
            FeedbackCategory.CONTEXT_RELEVANCE: {
                'recommendation': 'Improve context relevance of citations',
                'reason': f'Context relevance needs improvement (rating: {category_analysis["average_rating"]:.1f})'
            },
            FeedbackCategory.POSITION_ACCURACY: {
                'recommendation': 'Enhance position accuracy (page numbers, sections)',
                'reason': f'Position accuracy feedback indicates issues'
            },
            FeedbackCategory.COMPLETENESS: {
                'recommendation': 'Ensure complete citation information',
                'reason': f'Completeness is a concern based on user feedback'
            },
            FeedbackCategory.FORMAT_QUALITY: {
                'recommendation': 'Improve citation formatting and presentation',
                'reason': f'Format quality needs attention'
            }
        }
        
        base_recommendation = recommendations_map[category]
        
        return {
            'priority': priority_level,
            'category': category.value,
            'recommendation': base_recommendation['recommendation'],
            'reason': base_recommendation['reason'],
            'feedback_count': category_analysis['count'],
            'average_rating': category_analysis['average_rating']
        }

    def export_feedback_data(self, 
                           format_type: str = 'json',
                           citation_id: Optional[str] = None) -> str:
        """
        Export feedback data in specified format
        
        Args:
            format_type: Export format ('json' or 'csv')
            citation_id: Optional specific citation to export
            
        Returns:
            Exported data as string
        """
        feedback_list = list(self.feedback_storage.values())
        
        if citation_id:
            feedback_list = [f for f in feedback_list if f.citation_id == citation_id]
        
        if format_type.lower() == 'json':
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_feedback': len(feedback_list),
                'feedback_entries': [f.to_dict() for f in feedback_list]
            }
            return json.dumps(export_data, indent=2)
        
        elif format_type.lower() == 'csv':
            # Simple CSV format
            csv_lines = ['feedback_id,citation_id,user_id,feedback_type,category,rating,comment,timestamp']
            for feedback in feedback_list:
                comment_clean = feedback.comment.replace('"', '""') if feedback.comment else ''
                csv_line = f'"{feedback.feedback_id}","{feedback.citation_id}","{feedback.user_id}",'
                csv_line += f'"{feedback.feedback_type.value}","{feedback.category.value}",'
                csv_line += f'{feedback.rating},"{comment_clean}","{feedback.timestamp.isoformat()}"'
                csv_lines.append(csv_line)
            
            return '\n'.join(csv_lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")


def create_citation_feedback_system() -> CitationFeedbackSystem:
    """Factory function to create CitationFeedbackSystem instance"""
    return CitationFeedbackSystem()