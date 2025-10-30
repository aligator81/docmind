#!/usr/bin/env python3
"""
Test script for CitationFeedbackSystem service
"""

import sys
import os
import json
from datetime import datetime, timedelta

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.citation_feedback_system import (
    CitationFeedbackSystem, 
    FeedbackType,
    FeedbackCategory,
    create_citation_feedback_system
)

def test_feedback_submission():
    """Test feedback submission functionality"""
    print("🧪 Testing feedback submission...")
    
    feedback_system = create_citation_feedback_system()
    
    # Submit various types of feedback
    feedback_id1 = feedback_system.submit_feedback(
        citation_id="citation_123",
        user_id="user_456",
        feedback_type=FeedbackType.POSITIVE,
        category=FeedbackCategory.CITATION_ACCURACY,
        rating=5,
        comment="Excellent citation with perfect accuracy",
        metadata={"document_type": "academic"}
    )
    
    feedback_id2 = feedback_system.submit_feedback(
        citation_id="citation_123",
        user_id="user_789",
        feedback_type=FeedbackType.NEGATIVE,
        category=FeedbackCategory.SOURCE_VERIFICATION,
        rating=2,
        comment="Source document reference is unclear",
        metadata={"document_type": "legal"}
    )
    
    feedback_id3 = feedback_system.submit_feedback(
        citation_id="citation_456",
        user_id="user_456",
        feedback_type=FeedbackType.NEUTRAL,
        category=FeedbackCategory.CONTEXT_RELEVANCE,
        rating=3,
        comment="Context could be more relevant",
        metadata={"document_type": "technical"}
    )
    
    assert feedback_id1 is not None
    assert feedback_id2 is not None
    assert feedback_id3 is not None
    print(f"✅ Submitted {len(feedback_system.feedback_storage)} feedback entries")

def test_feedback_retrieval():
    """Test feedback retrieval functionality"""
    print("🧪 Testing feedback retrieval...")
    
    feedback_system = create_citation_feedback_system()
    
    # Submit test feedback
    feedback_system.submit_feedback(
        citation_id="test_citation_1",
        user_id="test_user_1",
        feedback_type=FeedbackType.POSITIVE,
        category=FeedbackCategory.POSITION_ACCURACY,
        rating=4,
        comment="Good position accuracy"
    )
    
    feedback_system.submit_feedback(
        citation_id="test_citation_1",
        user_id="test_user_2",
        feedback_type=FeedbackType.NEGATIVE,
        category=FeedbackCategory.COMPLETENESS,
        rating=2,
        comment="Missing some information"
    )
    
    # Test retrieval by citation
    citation_feedback = feedback_system.get_feedback_for_citation("test_citation_1")
    assert len(citation_feedback) == 2
    print(f"✅ Retrieved {len(citation_feedback)} feedback entries for citation")
    
    # Test retrieval by user
    user_feedback = feedback_system.get_feedback_by_user("test_user_1")
    assert len(user_feedback) == 1
    print(f"✅ Retrieved {len(user_feedback)} feedback entries for user")

def test_feedback_analysis():
    """Test feedback analysis functionality"""
    print("🧪 Testing feedback analysis...")
    
    feedback_system = create_citation_feedback_system()
    
    # Submit diverse feedback for analysis
    feedback_data = [
        # Positive feedback
        ("citation_analyze_1", "user_1", FeedbackType.POSITIVE, FeedbackCategory.CITATION_ACCURACY, 5),
        ("citation_analyze_1", "user_2", FeedbackType.POSITIVE, FeedbackCategory.SOURCE_VERIFICATION, 4),
        ("citation_analyze_1", "user_3", FeedbackType.POSITIVE, FeedbackCategory.CONTEXT_RELEVANCE, 4),
        
        # Mixed feedback
        ("citation_analyze_1", "user_4", FeedbackType.NEUTRAL, FeedbackCategory.POSITION_ACCURACY, 3),
        ("citation_analyze_1", "user_5", FeedbackType.NEGATIVE, FeedbackCategory.COMPLETENESS, 2),
        ("citation_analyze_1", "user_6", FeedbackType.NEGATIVE, FeedbackCategory.FORMAT_QUALITY, 1),
    ]
    
    for citation_id, user_id, feedback_type, category, rating in feedback_data:
        feedback_system.submit_feedback(
            citation_id=citation_id,
            user_id=user_id,
            feedback_type=feedback_type,
            category=category,
            rating=rating
        )
    
    # Analyze feedback
    analysis = feedback_system.analyze_feedback(citation_id="citation_analyze_1")
    
    assert analysis.total_feedback == 6
    assert 2.0 <= analysis.average_rating <= 4.0
    assert analysis.confidence_score > 0.5
    
    print(f"✅ Feedback analysis completed:")
    print(f"   - Total feedback: {analysis.total_feedback}")
    print(f"   - Average rating: {analysis.average_rating:.2f}")
    print(f"   - Confidence score: {analysis.confidence_score:.2f}")
    print(f"   - Feedback distribution: {analysis.feedback_distribution}")

def test_category_analysis():
    """Test category-specific analysis"""
    print("🧪 Testing category analysis...")
    
    feedback_system = create_citation_feedback_system()
    
    # Submit feedback with specific category patterns
    categories_to_test = [
        (FeedbackCategory.CITATION_ACCURACY, [5, 4, 3]),  # Good accuracy
        (FeedbackCategory.SOURCE_VERIFICATION, [2, 1, 2]),  # Poor verification
        (FeedbackCategory.CONTEXT_RELEVANCE, [4, 3, 4]),  # Moderate relevance
    ]
    
    citation_counter = 0
    for category, ratings in categories_to_test:
        for rating in ratings:
            feedback_system.submit_feedback(
                citation_id=f"category_test_{citation_counter}",
                user_id=f"user_{citation_counter}",
                feedback_type=FeedbackType.POSITIVE if rating >= 4 else FeedbackType.NEGATIVE if rating <= 2 else FeedbackType.NEUTRAL,
                category=category,
                rating=rating
            )
            citation_counter += 1
    
    analysis = feedback_system.analyze_feedback()
    
    # Check category analysis
    category_analysis = analysis.category_analysis
    assert FeedbackCategory.CITATION_ACCURACY in category_analysis
    assert FeedbackCategory.SOURCE_VERIFICATION in category_analysis
    assert FeedbackCategory.CONTEXT_RELEVANCE in category_analysis
    
    # Verify accuracy category has good scores
    accuracy_analysis = category_analysis[FeedbackCategory.CITATION_ACCURACY]
    assert accuracy_analysis['average_rating'] > 3.5
    print(f"✅ Citation accuracy: {accuracy_analysis['average_rating']:.2f}")
    
    # Verify source verification has poor scores
    verification_analysis = category_analysis[FeedbackCategory.SOURCE_VERIFICATION]
    assert verification_analysis['average_rating'] < 2.5
    print(f"✅ Source verification: {verification_analysis['average_rating']:.2f}")

def test_improvement_priorities():
    """Test improvement priority calculation"""
    print("🧪 Testing improvement priorities...")
    
    feedback_system = create_citation_feedback_system()
    
    # Submit feedback that should create clear priorities
    feedback_data = [
        (FeedbackCategory.SOURCE_VERIFICATION, 1),  # Very poor
        (FeedbackCategory.SOURCE_VERIFICATION, 2),  # Poor
        (FeedbackCategory.CITATION_ACCURACY, 5),    # Excellent
        (FeedbackCategory.CITATION_ACCURACY, 4),    # Good
        (FeedbackCategory.CONTEXT_RELEVANCE, 3),    # Neutral
        (FeedbackCategory.CONTEXT_RELEVANCE, 2),    # Poor
    ]
    
    for i, (category, rating) in enumerate(feedback_data):
        feedback_system.submit_feedback(
            citation_id=f"priority_test_{i}",
            user_id=f"user_{i}",
            feedback_type=FeedbackType.POSITIVE if rating >= 4 else FeedbackType.NEGATIVE if rating <= 2 else FeedbackType.NEUTRAL,
            category=category,
            rating=rating
        )
    
    analysis = feedback_system.analyze_feedback()
    
    # Check improvement priorities
    priorities = analysis.improvement_priorities
    assert len(priorities) > 0
    
    # Source verification should be highest priority (lowest scores)
    top_priority = priorities[0][0]
    assert top_priority == FeedbackCategory.SOURCE_VERIFICATION
    
    print(f"✅ Improvement priorities calculated:")
    for category, score in priorities:
        print(f"   - {category.value}: {score:.3f}")

def test_improvement_recommendations():
    """Test improvement recommendation generation"""
    print("🧪 Testing improvement recommendations...")
    
    feedback_system = create_citation_feedback_system()
    
    # Submit feedback that should generate specific recommendations
    feedback_system.submit_feedback(
        citation_id="recommendation_test",
        user_id="user_1",
        feedback_type=FeedbackType.NEGATIVE,
        category=FeedbackCategory.SOURCE_VERIFICATION,
        rating=1,
        comment="Cannot verify the source document"
    )
    
    feedback_system.submit_feedback(
        citation_id="recommendation_test",
        user_id="user_2",
        feedback_type=FeedbackType.NEGATIVE,
        category=FeedbackCategory.POSITION_ACCURACY,
        rating=2,
        comment="Page numbers are incorrect"
    )
    
    analysis = feedback_system.analyze_feedback()
    recommendations = feedback_system.get_improvement_recommendations(analysis)
    
    assert len(recommendations) > 0
    
    print(f"✅ Generated {len(recommendations)} improvement recommendations:")
    for rec in recommendations:
        print(f"   - [{rec['priority']}] {rec['category']}: {rec['recommendation']}")

def test_empty_feedback_analysis():
    """Test analysis with no feedback data"""
    print("🧪 Testing empty feedback analysis...")
    
    feedback_system = create_citation_feedback_system()
    
    # Analyze with no feedback
    analysis = feedback_system.analyze_feedback()
    
    assert analysis.total_feedback == 0
    assert analysis.average_rating == 0.0
    assert analysis.confidence_score == 0.0
    
    # Get recommendations for empty analysis
    recommendations = feedback_system.get_improvement_recommendations(analysis)
    assert len(recommendations) == 1
    assert recommendations[0]['priority'] == 'high'
    assert 'Collect more user feedback' in recommendations[0]['recommendation']
    
    print("✅ Empty feedback analysis handled correctly")

def test_time_period_analysis():
    """Test analysis with time period filtering"""
    print("🧪 Testing time period analysis...")
    
    feedback_system = create_citation_feedback_system()
    
    # This would normally require manipulating timestamps, but we'll test the interface
    analysis = feedback_system.analyze_feedback(time_period=timedelta(days=7))
    
    # Just verify the method works without errors
    assert analysis is not None
    print("✅ Time period analysis completed")

def test_data_export():
    """Test feedback data export functionality"""
    print("🧪 Testing data export...")
    
    feedback_system = create_citation_feedback_system()
    
    # Submit test feedback
    feedback_system.submit_feedback(
        citation_id="export_test",
        user_id="export_user",
        feedback_type=FeedbackType.POSITIVE,
        category=FeedbackCategory.FORMAT_QUALITY,
        rating=5,
        comment="Well-formatted citation"
    )
    
    # Test JSON export
    json_export = feedback_system.export_feedback_data(format_type='json')
    export_data = json.loads(json_export)
    
    assert 'export_timestamp' in export_data
    assert 'total_feedback' in export_data
    assert 'feedback_entries' in export_data
    assert len(export_data['feedback_entries']) == 1
    
    print("✅ JSON export completed successfully")
    
    # Test CSV export
    csv_export = feedback_system.export_feedback_data(format_type='csv')
    csv_lines = csv_export.split('\n')
    
    assert len(csv_lines) >= 2  # Header + at least one data row
    assert 'feedback_id' in csv_lines[0]
    
    print("✅ CSV export completed successfully")

def test_confidence_score_calculation():
    """Test confidence score calculation"""
    print("🧪 Testing confidence score calculation...")
    
    feedback_system = create_citation_feedback_system()
    
    # Test with minimal feedback (low confidence)
    feedback_system.submit_feedback(
        citation_id="confidence_test_1",
        user_id="user_1",
        feedback_type=FeedbackType.POSITIVE,
        category=FeedbackCategory.CITATION_ACCURACY,
        rating=5
    )
    
    analysis_minimal = feedback_system.analyze_feedback()
    assert analysis_minimal.confidence_score < 0.5
    print(f"✅ Minimal feedback confidence: {analysis_minimal.confidence_score:.2f}")
    
    # Add more diverse feedback (higher confidence)
    for i in range(10):
        rating = 4 if i % 2 == 0 else 3  # Mix of ratings
        feedback_type = FeedbackType.POSITIVE if rating >= 4 else FeedbackType.NEUTRAL
        
        feedback_system.submit_feedback(
            citation_id=f"confidence_test_2",
            user_id=f"user_{i+2}",
            feedback_type=feedback_type,
            category=FeedbackCategory.CITATION_ACCURACY,
            rating=rating
        )
    
    analysis_diverse = feedback_system.analyze_feedback(citation_id="confidence_test_2")
    assert analysis_diverse.confidence_score > analysis_minimal.confidence_score
    print(f"✅ Diverse feedback confidence: {analysis_diverse.confidence_score:.2f}")

def main():
    """Run all CitationFeedbackSystem tests"""
    print("🚀 Starting Citation Feedback System Tests")
    print("=" * 50)
    
    try:
        test_feedback_submission()
        test_feedback_retrieval()
        test_feedback_analysis()
        test_category_analysis()
        test_improvement_priorities()
        test_improvement_recommendations()
        test_empty_feedback_analysis()
        test_time_period_analysis()
        test_data_export()
        test_confidence_score_calculation()
        
        print("\n🎉 All CitationFeedbackSystem tests completed successfully!")
        print("📊 The feedback system can now:")
        print("   - Collect and store user feedback on citations")
        print("   - Analyze feedback by category and type")
        print("   - Calculate improvement priorities")
        print("   - Generate specific recommendations")
        print("   - Export feedback data in multiple formats")
        print("   - Calculate confidence scores for analysis")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())