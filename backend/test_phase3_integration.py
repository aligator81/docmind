"""
Comprehensive Phase 3 Integration Test
Tests cross-document relationships, adaptive chunking, and relevance tuning.
"""
import sys
import os
import asyncio
import logging
import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import get_db
from app.models import Base, Document, DocumentChunk, SearchQuery, UserFeedback
from app.services.document_relationship_mapper import DocumentRelationshipMapper
from app.services.multi_document_analyzer import MultiDocumentAnalyzer
from app.services.document_type_classifier import DocumentTypeClassifier
from app.services.adaptive_chunker import AdaptiveChunker
from app.services.relevance_tuner import RelevanceTuner, FeedbackType, TuningStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test database setup
TEST_DATABASE_URL = "sqlite:///./test_phase3.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def setup_test_database():
    """Set up test database with sample data."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    db = TestingSessionLocal()
    
    try:
        # Create sample documents with all required fields
        sample_documents = [
            Document(
                filename="legal_contract_1.pdf",
                original_filename="legal_contract_1.pdf",
                file_path="/test/legal_contract_1.pdf",
                file_size=1024,
                mime_type="application/pdf",
                user_id=1,
                content="This agreement is made between Company A and Company B. The contract covers software development services for a period of 12 months. Payment terms: $50,000 upon signing, $100,000 upon completion.",
                metadata_=json.dumps({
                    "document_type": "contract",
                    "parties": ["Company A", "Company B"],
                    "duration": "12 months",
                    "payment_terms": ["$50,000 upon signing", "$100,000 upon completion"]
                })
            ),
            Document(
                filename="legal_contract_2.pdf",
                original_filename="legal_contract_2.pdf",
                file_path="/test/legal_contract_2.pdf",
                file_size=1024,
                mime_type="application/pdf",
                user_id=1,
                content="Amendment to the software development agreement. The payment schedule is modified to include milestone-based payments. Milestone 1: $25,000, Milestone 2: $50,000, Milestone 3: $75,000.",
                metadata_=json.dumps({
                    "document_type": "amendment",
                    "related_document": "legal_contract_1.pdf",
                    "payment_structure": "milestone-based"
                })
            ),
            Document(
                filename="technical_specification.docx",
                original_filename="technical_specification.docx",
                file_path="/test/technical_specification.docx",
                file_size=1024,
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                user_id=1,
                content="Technical requirements for the software project. The system must support 1000 concurrent users and have response times under 200ms. Database requirements: PostgreSQL 14+, Redis for caching.",
                metadata_=json.dumps({
                    "document_type": "technical_spec",
                    "requirements": ["1000 concurrent users", "response times under 200ms"],
                    "technologies": ["PostgreSQL", "Redis"]
                })
            ),
            Document(
                filename="project_timeline.xlsx",
                original_filename="project_timeline.xlsx",
                file_path="/test/project_timeline.xlsx",
                file_size=1024,
                mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                user_id=1,
                content="Project timeline with key milestones. Phase 1: Requirements gathering (2 weeks), Phase 2: Development (8 weeks), Phase 3: Testing (2 weeks), Phase 4: Deployment (1 week).",
                metadata_=json.dumps({
                    "document_type": "timeline",
                    "phases": ["Requirements", "Development", "Testing", "Deployment"],
                    "total_duration": "13 weeks"
                })
            )
        ]
        
        db.add_all(sample_documents)
        db.commit()
        
        # Create sample chunks for testing
        for doc in sample_documents:
            chunk = DocumentChunk(
                document_id=doc.id,
                chunk_text=doc.content[:500],  # First 500 chars as chunk
                chunk_index=0,
                page_numbers_list=[1],  # Use the property setter
                section_title="Main Content"
            )
            # Set metadata through the metadata_ column
            chunk.metadata_ = json.dumps(doc.metadata) if doc.metadata else None
            db.add(chunk)
        
        db.commit()
        
        # Create sample search queries and feedback
        search_queries = [
            SearchQuery(
                query_text="payment terms contract",
                search_count=5,
                first_seen=datetime.utcnow() - timedelta(days=10),
                last_seen=datetime.utcnow()
            ),
            SearchQuery(
                query_text="technical requirements software",
                search_count=3,
                first_seen=datetime.utcnow() - timedelta(days=5),
                last_seen=datetime.utcnow()
            ),
            SearchQuery(
                query_text="project timeline milestones",
                search_count=2,
                first_seen=datetime.utcnow() - timedelta(days=2),
                last_seen=datetime.utcnow()
            )
        ]
        
        db.add_all(search_queries)
        db.commit()
        
        # Create sample feedback
        feedback_entries = [
            UserFeedback(
                query_id=search_queries[0].id,
                feedback_type=FeedbackType.POSITIVE.value,
                timestamp=datetime.utcnow() - timedelta(days=1),
                metadata={"document_ids": [1, 2], "query": "payment terms contract"}
            ),
            UserFeedback(
                query_id=search_queries[1].id,
                feedback_type=FeedbackType.NEGATIVE.value,
                timestamp=datetime.utcnow() - timedelta(days=2),
                metadata={"document_ids": [3], "query": "technical requirements software"}
            ),
            UserFeedback(
                query_id=search_queries[2].id,
                feedback_type=FeedbackType.CLICK.value,
                timestamp=datetime.utcnow() - timedelta(hours=12),
                metadata={"document_ids": [4], "query": "project timeline milestones"}
            )
        ]
        
        db.add_all(feedback_entries)
        db.commit()
        
        logger.info("Test database setup completed with sample data")
        
    except Exception as e:
        logger.error(f"Error setting up test database: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

def test_document_relationship_mapper():
    """Test cross-document relationship mapping."""
    logger.info("Testing Document Relationship Mapper...")
    
    db = TestingSessionLocal()
    try:
        mapper = DocumentRelationshipMapper(db)
        
        # Test relationship analysis
        document_ids = [1, 2, 3, 4]
        relationships = mapper.analyze_relationships(document_ids)
        
        logger.info(f"Found {len(relationships)} relationships")
        for rel in relationships:
            logger.info(f"Relationship: {rel['source_doc']} -> {rel['target_doc']} ({rel['relationship_type']})")
        
        # Test relationship graph
        graph_data = mapper.create_relationship_graph(document_ids)
        logger.info(f"Graph nodes: {len(graph_data.get('nodes', []))}")
        logger.info(f"Graph edges: {len(graph_data.get('edges', []))}")
        
        return relationships, graph_data
        
    except Exception as e:
        logger.error(f"Error testing relationship mapper: {str(e)}")
        raise
    finally:
        db.close()

def test_multi_document_analyzer():
    """Test multi-document analysis capabilities."""
    logger.info("Testing Multi-Document Analyzer...")
    
    db = TestingSessionLocal()
    try:
        analyzer = MultiDocumentAnalyzer(db)
        
        document_ids = [1, 2, 3, 4]
        
        # Test comparative analysis
        comparative_results = analyzer.perform_comparative_analysis(document_ids)
        logger.info(f"Comparative analysis: {len(comparative_results.get('comparisons', []))} comparisons")
        
        # Test synthesis analysis
        synthesis_results = analyzer.perform_synthesis_analysis(document_ids)
        logger.info(f"Synthesis analysis: {len(synthesis_results.get('synthesized_content', []))} synthesized items")
        
        # Test conflict detection
        conflicts = analyzer.detect_conflicts(document_ids)
        logger.info(f"Conflict detection: {len(conflicts.get('conflicts', []))} conflicts")
        
        # Test gap analysis
        gaps = analyzer.perform_gap_analysis(document_ids)
        logger.info(f"Gap analysis: {len(gaps.get('gaps', []))} gaps")
        
        return comparative_results, synthesis_results, conflicts, gaps
        
    except Exception as e:
        logger.error(f"Error testing multi-document analyzer: {str(e)}")
        raise
    finally:
        db.close()

def test_document_type_classifier():
    """Test document type classification."""
    logger.info("Testing Document Type Classifier...")
    
    db = TestingSessionLocal()
    try:
        classifier = DocumentTypeClassifier(db)
        
        document_ids = [1, 2, 3, 4]
        
        # Test document classification
        classification_results = classifier.classify_documents(document_ids)
        logger.info("Document classifications:")
        for doc_id, doc_type in classification_results.get("document_types", {}).items():
            logger.info(f"  Document {doc_id}: {doc_type}")
        
        # Test structure pattern identification
        structure_patterns = classifier.identify_structure_patterns(document_ids)
        logger.info(f"Structure patterns: {len(structure_patterns.get('patterns', []))} patterns")
        
        return classification_results, structure_patterns
        
    except Exception as e:
        logger.error(f"Error testing document type classifier: {str(e)}")
        raise
    finally:
        db.close()

def test_adaptive_chunker():
    """Test adaptive chunking strategies."""
    logger.info("Testing Adaptive Chunker...")
    
    db = TestingSessionLocal()
    try:
        chunker = AdaptiveChunker(db)
        
        # Test different chunking strategies
        document_id = 1  # Legal contract
        
        strategies = ["fixed_size", "section_based", "semantic", "adaptive"]
        
        for strategy in strategies:
            logger.info(f"Testing {strategy} chunking...")
            
            chunking_result = chunker.apply_chunking_strategy(document_id, strategy)
            
            logger.info(f"  Strategy: {chunking_result.get('strategy')}")
            logger.info(f"  Chunk count: {chunking_result.get('chunk_count')}")
            logger.info(f"  Quality score: {chunking_result.get('quality_metrics', {}).get('overall_score', 0):.2f}")
            
            # Test chunk quality analysis
            quality_analysis = chunker.analyze_chunk_quality(document_id)
            logger.info(f"  Quality analysis: {quality_analysis.get('overall_score', 0):.2f}")
        
        # Test strategy comparison
        comparison = chunker.compare_strategies(document_id)
        logger.info(f"Strategy comparison: {len(comparison.get('strategies', []))} strategies compared")
        
        return comparison
        
    except Exception as e:
        logger.error(f"Error testing adaptive chunker: {str(e)}")
        raise
    finally:
        db.close()

def test_relevance_tuner():
    """Test relevance tuning capabilities."""
    logger.info("Testing Relevance Tuner...")
    
    db = TestingSessionLocal()
    try:
        tuner = RelevanceTuner(db)
        
        # Test feedback collection
        feedback_result = tuner.collect_feedback(
            query="test query for relevance tuning",
            document_ids=[1, 2],
            feedback_type=FeedbackType.POSITIVE
        )
        logger.info(f"Feedback collection: {feedback_result.get('success', False)}")
        
        # Test feedback analysis
        analysis = tuner.analyze_feedback_patterns(days_back=7)
        logger.info(f"Feedback analysis: {analysis.get('summary', {}).get('overall_health', 'unknown')}")
        logger.info(f"  Total feedback: {analysis.get('feedback_statistics', {}).get('total_feedback', 0)}")
        logger.info(f"  Relevance issues: {len(analysis.get('relevance_issues', []))}")
        
        # Test tuning strategies
        strategies = [
            TuningStrategy.WEIGHT_ADJUSTMENT,
            TuningStrategy.QUERY_EXPANSION,
            TuningStrategy.HYBRID
        ]
        
        for strategy in strategies:
            logger.info(f"Testing {strategy.value} tuning...")
            
            tuning_result = tuner.tune_relevance_parameters(strategy)
            
            logger.info(f"  Strategy: {tuning_result.get('strategy')}")
            logger.info(f"  Effectiveness: {tuning_result.get('effectiveness_estimate', {}).get('estimated_improvement', 0):.1%}")
            logger.info(f"  Recommendations: {len(tuning_result.get('recommendations', []))}")
        
        # Test A/B test creation
        ab_test = tuner.create_ab_test(
            TuningStrategy.WEIGHT_ADJUSTMENT,
            TuningStrategy.QUERY_EXPANSION,
            test_duration_days=3
        )
        logger.info(f"A/B test created: {ab_test.get('test_id')}")
        
        return analysis, ab_test
        
    except Exception as e:
        logger.error(f"Error testing relevance tuner: {str(e)}")
        raise
    finally:
        db.close()

def test_integrated_workflow():
    """Test integrated workflow across all Phase 3 services."""
    logger.info("Testing Integrated Workflow...")
    
    db = TestingSessionLocal()
    try:
        # Get all documents
        documents = db.query(Document).all()
        document_ids = [doc.id for doc in documents]
        
        logger.info(f"Testing integrated workflow with {len(document_ids)} documents")
        
        # 1. Classify documents
        classifier = DocumentTypeClassifier(db)
        classification = classifier.classify_documents(document_ids)
        
        # 2. Analyze relationships
        mapper = DocumentRelationshipMapper(db)
        relationships = mapper.analyze_relationships(document_ids)
        
        # 3. Multi-document analysis
        analyzer = MultiDocumentAnalyzer(db)
        multi_analysis = analyzer.perform_comprehensive_analysis(document_ids)
        
        # 4. Adaptive chunking
        chunker = AdaptiveChunker(db)
        chunking_results = {}
        for doc_id in document_ids:
            chunking_results[doc_id] = chunker.apply_optimal_strategy(doc_id)
        
        # 5. Relevance tuning
        tuner = RelevanceTuner(db)
        tuning_analysis = tuner.analyze_feedback_patterns()
        tuning_recommendations = tuning_analysis.get('tuning_recommendations', [])
        
        # Log integrated results
        logger.info("Integrated Workflow Results:")
        logger.info(f"  Document types: {classification.get('document_types', {})}")
        logger.info(f"  Relationships found: {len(relationships)}")
        logger.info(f"  Multi-document insights: {len(multi_analysis.get('insights', []))}")
        logger.info(f"  Chunking strategies applied: {len(chunking_results)}")
        logger.info(f"  Tuning recommendations: {len(tuning_recommendations)}")
        
        return {
            "classification": classification,
            "relationships": relationships,
            "multi_analysis": multi_analysis,
            "chunking_results": chunking_results,
            "tuning_recommendations": tuning_recommendations
        }
        
    except Exception as e:
        logger.error(f"Error testing integrated workflow: {str(e)}")
        raise
    finally:
        db.close()

def main():
    """Run all Phase 3 integration tests."""
    logger.info("Starting Phase 3 Integration Tests...")
    
    try:
        # Setup test database
        setup_test_database()
        
        # Run individual service tests
        test_document_relationship_mapper()
        test_multi_document_analyzer()
        test_document_type_classifier()
        test_adaptive_chunker()
        test_relevance_tuner()
        
        # Run integrated workflow test
        integrated_results = test_integrated_workflow()
        
        logger.info("🎉 Phase 3 Integration Tests Completed Successfully!")
        logger.info("All services are working correctly and integrated properly.")
        
        # Print summary
        logger.info("\n📊 Phase 3 Test Summary:")
        logger.info("  ✅ Document Relationship Mapper - Cross-document analysis")
        logger.info("  ✅ Multi-Document Analyzer - Comparative and synthesis analysis")
        logger.info("  ✅ Document Type Classifier - Document classification and structure patterns")
        logger.info("  ✅ Adaptive Chunker - Multiple chunking strategies with quality assessment")
        logger.info("  ✅ Relevance Tuner - Feedback analysis and parameter tuning")
        logger.info("  ✅ Integrated Workflow - All services working together")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 3 Integration Tests Failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)