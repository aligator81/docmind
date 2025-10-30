"""
Integration Test for Phase 2 Enhancements
Tests all Phase 2 services working together in the document processing pipeline
"""

import sys
import os
import json
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.document_structure_analyzer import DocumentStructureAnalyzer
from app.services.citation_validator import CitationValidator
from app.services.chunk_quality_analyzer import ChunkQualityAnalyzer
from app.services.citation_feedback_system import CitationFeedbackSystem, FeedbackType, FeedbackCategory

def test_phase2_integration():
    """Test the complete Phase 2 enhancement pipeline"""
    print("🚀 Starting Phase 2 Integration Test")
    print("=" * 60)
    
    # Initialize all services
    structure_analyzer = DocumentStructureAnalyzer()
    citation_validator = CitationValidator()
    chunk_analyzer = ChunkQualityAnalyzer()
    feedback_system = CitationFeedbackSystem()
    
    # Test document content
    test_document = """
    # Research Paper on Machine Learning
    
    ## Abstract
    This paper explores the application of machine learning algorithms in healthcare diagnostics.
    
    ## Introduction
    Machine learning has revolutionized many industries, particularly healthcare. According to recent studies, ML algorithms can achieve up to 95% accuracy in medical image analysis.
    
    ## Methodology
    We conducted experiments using three different algorithms: Random Forest, SVM, and Neural Networks. The dataset consisted of 10,000 medical images from various sources.
    
    ## Results
    Our results show that Neural Networks outperformed other algorithms with 94.7% accuracy on the test set. Random Forest achieved 92.1% and SVM achieved 89.3%.
    
    ## Conclusion
    Machine learning shows great promise in healthcare diagnostics, with neural networks providing the best performance in our study.
    
    ## References
    1. Smith, J. et al. "ML in Healthcare", Journal of Medical AI, 2023, pages 45-52.
    2. Johnson, A. "Advanced ML Techniques", AI Review, 2022, pages 123-130.
    """
    
    print("📄 Testing Document Structure Analysis...")
    structure_analysis = structure_analyzer.analyze_document_structure(test_document)
    print(f"✅ Document Type: {structure_analysis.document_type.value}")
    print(f"✅ Sections Found: {len(structure_analysis.sections)}")
    print(f"✅ Page Count: {structure_analysis.page_count}")
    
    # Test chunk quality analysis
    print("\n🔍 Testing Chunk Quality Analysis...")
    test_chunks = [
        {
            "text": "Machine learning has revolutionized many industries, particularly healthcare. According to recent studies, ML algorithms can achieve up to 95% accuracy in medical image analysis.",
            "metadata": {
                "section": "Introduction",
                "page_numbers": [2],
                "document_name": "research_paper.pdf"
            }
        },
        {
            "text": "Our results show that Neural Networks outperformed other algorithms with 94.7% accuracy on the test set. Random Forest achieved 92.1% and SVM achieved 89.3%.",
            "metadata": {
                "section": "Results", 
                "page_numbers": [4],
                "document_name": "research_paper.pdf"
            }
        }
    ]
    
    chunk_quality_results = []
    for i, chunk in enumerate(test_chunks):
        quality = chunk_analyzer.analyze_chunk_quality(
            chunk_text=chunk["text"],
            chunk_size=len(chunk["text"]),
            document_type="academic_paper",
            section_context=chunk["metadata"]["section"]
        )
        quality_dict = quality.to_dict()
        chunk_quality_results.append(quality_dict)
        print(f"✅ Chunk {i+1} Quality Score: {quality_dict['quality_score']:.2f}")
        print(f"   - Semantic Coherence: {quality_dict['quality_metrics']['semantic_coherence']:.2f}")
        print(f"   - Structural Integrity: {quality_dict['quality_metrics']['structural_integrity']:.2f}")
    
    # Test citation validation
    print("\n📚 Testing Citation Validation...")
    test_citations = [
        {
            "text": "According to recent studies, ML algorithms can achieve up to 95% accuracy in medical image analysis.",
            "source_document": "research_paper.pdf",
            "page_numbers": [2],
            "section": "Introduction"
        },
        {
            "text": "Smith, J. et al. 'ML in Healthcare', Journal of Medical AI, 2023, pages 45-52.",
            "source_document": "research_paper.pdf", 
            "page_numbers": [6],
            "section": "References"
        }
    ]
    
    citation_validation_results = []
    for i, citation in enumerate(test_citations):
        validation = citation_validator.validate_citation(
            chunk_text=citation["text"],
            source_document=citation["source_document"],
            page_numbers=citation["page_numbers"],
            section_title=citation["section"]
        )
        validation_dict = validation.to_dict()
        citation_validation_results.append(validation_dict)
        print(f"✅ Citation {i+1} Validation Score: {validation_dict['confidence_score']:.2f}")
        print(f"   - Source Verified: {validation_dict['source_verification']['document_name']}")
        print(f"   - Position Accuracy: {validation_dict['validation_details']['position_accuracy']:.2f}")
    
    # Test feedback system integration
    print("\n💬 Testing Feedback System Integration...")
    
    # Submit feedback for citations
    for i, citation in enumerate(test_citations):
        feedback_id = feedback_system.submit_feedback(
            citation_id=f"test_citation_{i}",
            user_id=f"test_user_{i}",
            feedback_type=FeedbackType.POSITIVE if i == 0 else FeedbackType.NEUTRAL,
            category=FeedbackCategory.CITATION_ACCURACY,
            rating=4 if i == 0 else 3,  # Higher rating for first citation
            comment="Good citation with proper source attribution" if i == 0 else "Average citation quality",
            metadata={
                "citation_accuracy": 4 if i == 0 else 3,
                "source_verification": 5 if i == 0 else 3,
                "context_relevance": 4 if i == 0 else 3
            }
        )
        print(f"✅ Submitted feedback for citation {i+1}: {feedback_id}")
    
    # Analyze feedback
    feedback_analysis = feedback_system.analyze_feedback("test_citation_0")
    feedback_analysis_dict = feedback_analysis.to_dict()
    print(f"✅ Feedback Analysis - Average Rating: {feedback_analysis_dict['average_rating']:.2f}")
    print(f"✅ Feedback Analysis - Confidence: {feedback_analysis_dict['confidence_score']:.2f}")
    
    # Get improvement priorities
    priorities = feedback_analysis_dict['improvement_priorities']
    print(f"✅ Improvement Priorities: {priorities}")
    
    # Generate recommendations
    recommendations = feedback_system.get_improvement_recommendations(feedback_analysis)
    print(f"✅ Generated {len(recommendations)} improvement recommendations")
    
    # Test adaptive chunking recommendations
    print("\n🔄 Testing Adaptive Chunking Recommendations...")
    # Get recommendations from the first chunk result
    if chunk_quality_results:
        chunking_recommendations = chunk_quality_results[0]['adaptive_chunking_recommendations']
        print(f"✅ Chunking Strategy: {chunking_recommendations['chunking_strategy']}")
        print(f"✅ Target Chunk Size: {chunking_recommendations['optimal_chunk_size']}")
    
    # Generate comprehensive report
    print("\n📊 Generating Phase 2 Integration Report...")
    integration_report = {
        "timestamp": datetime.now().isoformat(),
        "document_analysis": {
            "document_type": structure_analysis.document_type.value,
            "sections_count": len(structure_analysis.sections),
            "page_count": structure_analysis.page_count,
            "confidence_scores": structure_analysis.confidence_scores
        },
        "chunk_quality": {
            "average_score": sum(r["quality_score"] for r in chunk_quality_results) / len(chunk_quality_results),
            "individual_scores": [r["quality_score"] for r in chunk_quality_results]
        },
        "citation_validation": {
            "average_confidence": sum(r["confidence_score"] for r in citation_validation_results) / len(citation_validation_results),
            "individual_scores": [r["confidence_score"] for r in citation_validation_results]
        },
        "feedback_analysis": feedback_analysis_dict,
        "improvement_priorities": priorities,
        "chunking_recommendations": chunking_recommendations
    }
    
    print("🎉 Phase 2 Integration Test Completed Successfully!")
    print("=" * 60)
    print("📋 Integration Report Summary:")
    print(f"   - Document Type: {integration_report['document_analysis']['document_type']}")
    print(f"   - Average Chunk Quality: {integration_report['chunk_quality']['average_score']:.2f}")
    print(f"   - Average Citation Confidence: {integration_report['citation_validation']['average_confidence']:.2f}")
    print(f"   - Feedback Confidence: {integration_report['feedback_analysis']['confidence_score']:.2f}")
    print(f"   - Top Improvement Priority: {integration_report['improvement_priorities'][0][0] if integration_report['improvement_priorities'] else 'None'}")
    
    return integration_report

if __name__ == "__main__":
    report = test_phase2_integration()
    
    # Save detailed report
    with open("phase2_integration_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n💾 Detailed report saved to: phase2_integration_report.json")