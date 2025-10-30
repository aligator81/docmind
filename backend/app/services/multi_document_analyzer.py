
"""
Multi-Document Analyzer Service
Provides cross-document analysis, synthesis, and insights.
"""
import logging
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Document, DocumentChunk, Embedding
from .document_relationship_mapper import DocumentRelationshipMapper, RelationshipType

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of multi-document analysis."""
    COMPARATIVE = "comparative"
    SYNTHESIS = "synthesis"
    CONFLICT_DETECTION = "conflict_detection"
    GAP_ANALYSIS = "gap_analysis"
    TIMELINE_ANALYSIS = "timeline_analysis"


class MultiDocumentAnalyzer:
    """
    Service for analyzing multiple documents together to provide insights.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.relationship_mapper = DocumentRelationshipMapper(db_session)
    
    def analyze_document_set(self, document_ids: List[int], analysis_type: AnalysisType) -> Dict[str, Any]:
        """
        Analyze a set of documents based on specified analysis type.
        
        Args:
            document_ids: List of document IDs to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis results
        """
        try:
            # Get documents
            documents = self.db.query(Document).filter(Document.id.in_(document_ids)).all()
            if len(documents) != len(document_ids):
                missing_ids = set(document_ids) - {doc.id for doc in documents}
                raise ValueError(f"Documents not found: {missing_ids}")
            
            # Perform analysis based on type
            if analysis_type == AnalysisType.COMPARATIVE:
                return self._perform_comparative_analysis(documents)
            elif analysis_type == AnalysisType.SYNTHESIS:
                return self._perform_synthesis_analysis(documents)
            elif analysis_type == AnalysisType.CONFLICT_DETECTION:
                return self._perform_conflict_detection(documents)
            elif analysis_type == AnalysisType.GAP_ANALYSIS:
                return self._perform_gap_analysis(documents)
            elif analysis_type == AnalysisType.TIMELINE_ANALYSIS:
                return self._perform_timeline_analysis(documents)
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
                
        except Exception as e:
            logger.error(f"Error analyzing document set {document_ids}: {str(e)}")
            raise
    
    def _perform_comparative_analysis(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Perform comparative analysis between documents.
        
        Args:
            documents: List of documents to compare
            
        Returns:
            Comparative analysis results
        """
        try:
            analysis = {
                "analysis_type": AnalysisType.COMPARATIVE.value,
                "documents_analyzed": len(documents),
                "comparisons": [],
                "key_differences": [],
                "key_similarities": []
            }
            
            # Compare each pair of documents
            for i, doc1 in enumerate(documents):
                for doc2 in documents[i+1:]:
                    comparison = self._compare_documents(doc1, doc2)
                    analysis["comparisons"].append(comparison)
                    
                    # Extract key differences and similarities
                    if comparison["similarity_score"] < 0.3:
                        analysis["key_differences"].append({
                            "documents": [doc1.title, doc2.title],
                            "difference_type": "content_focus",
                            "description": f"Different content focus: {doc1.title} vs {doc2.title}"
                        })
                    elif comparison["similarity_score"] > 0.7:
                        analysis["key_similarities"].append({
                            "documents": [doc1.title, doc2.title],
                            "similarity_type": "content_overlap",
                            "description": f"High content overlap: {doc1.title} and {doc2.title}"
                        })
            
            # Generate overall insights
            analysis["overall_insights"] = self._generate_comparative_insights(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error performing comparative analysis: {str(e)}")
            raise
    
    def _perform_synthesis_analysis(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Perform synthesis analysis to combine insights from multiple documents.
        
        Args:
            documents: List of documents to synthesize
            
        Returns:
            Synthesis analysis results
        """
        try:
            analysis = {
                "analysis_type": AnalysisType.SYNTHESIS.value,
                "documents_analyzed": len(documents),
                "synthesized_topics": [],
                "complementary_insights": [],
                "knowledge_gaps": []
            }
            
            # Analyze topic coverage across documents
            all_topics = self._extract_topics_from_documents(documents)
            analysis["synthesized_topics"] = self._synthesize_topics(all_topics)
            
            # Identify complementary information
            analysis["complementary_insights"] = self._identify_complementary_insights(documents)
            
            # Identify knowledge gaps
            analysis["knowledge_gaps"] = self._identify_knowledge_gaps(documents, all_topics)
            
            # Generate unified summary
            analysis["unified_summary"] = self._generate_unified_summary(documents)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error performing synthesis analysis: {str(e)}")
            raise
    
    def _perform_conflict_detection(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Detect conflicts and contradictions between documents.
        
        Args:
            documents: List of documents to analyze for conflicts
            
        Returns:
            Conflict detection results
        """
        try:
            analysis = {
                "analysis_type": AnalysisType.CONFLICT_DETECTION.value,
                "documents_analyzed": len(documents),
                "potential_conflicts": [],
                "contradiction_areas": [],
                "confidence_scores": {}
            }
            
            # Analyze relationships for conflicts
            for doc in documents:
                relationships = self.relationship_mapper.analyze_document_relationships(doc.id)
                conflicting_rels = [r for r in relationships.get("relationships", []) 
                                  if r["relationship_type"] == RelationshipType.CONFLICTING.value]
                
                for conflict in conflicting_rels:
                    if conflict["related_document_id"] in [d.id for d in documents]:
                        analysis["potential_conflicts"].append({
                            "document_pair": [doc.title, conflict["related_document_title"]],
                            "confidence": conflict["confidence"],
                            "evidence": conflict["evidence"],
                            "metrics": conflict["metrics"]
                        })
            
            # Analyze content for contradictions
            analysis["contradiction_areas"] = self._analyze_content_contradictions(documents)
            
            # Calculate confidence scores for conflict detection
            analysis["confidence_scores"] = self._calculate_conflict_confidence(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error performing conflict detection: {str(e)}")
            raise
    
    def _perform_gap_analysis(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Identify knowledge gaps in the document set.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            Gap analysis results
        """
        try:
            analysis = {
                "analysis_type": AnalysisType.GAP_ANALYSIS.value,
                "documents_analyzed": len(documents),
                "identified_gaps": [],
                "coverage_analysis": {},
                "recommendations": []
            }
            
            # Analyze topic coverage
            all_topics = self._extract_topics_from_documents(documents)
            expected_topics = self._get_expected_topic_coverage(documents)
            
            # Identify missing topics
            missing_topics = expected_topics - set(all_topics.keys())
            for topic in missing_topics:
                analysis["identified_gaps"].append({
                    "gap_type": "topic_coverage",
                    "topic": topic,
                    "severity": "medium",
                    "description": f"Topic '{topic}' is not covered in the document set"
                })
            
            # Analyze depth of coverage
            analysis["coverage_analysis"] = self._analyze_coverage_depth(documents, all_topics)
            
            # Generate recommendations
            analysis["recommendations"] = self._generate_gap_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error performing gap analysis: {str(e)}")
            raise
    
    def _perform_timeline_analysis(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Analyze temporal relationships and evolution of information.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            Timeline analysis results
        """
        try:
            analysis = {
                "analysis_type": AnalysisType.TIMELINE_ANALYSIS.value,
                "documents_analyzed": len(documents),
                "temporal_sequence": [],
                "evolution_patterns": [],
                "key_milestones": []
            }
            
            # Sort documents by creation date
            sorted_docs = sorted(documents, key=lambda x: x.created_at)
            
            # Build temporal sequence
            for i, doc in enumerate(sorted_docs):
                analysis["temporal_sequence"].append({
                    "document_id": doc.id,
                    "document_title": doc.title,
                    "timestamp": doc.created_at.isoformat(),
                    "position": i + 1,
                    "key_contributions": self._extract_key_contributions(doc)
                })
            
            # Analyze evolution patterns
            analysis["evolution_patterns"] = self._analyze_evolution_patterns(sorted_docs)
            
            # Identify key milestones
            analysis["key_milestones"] = self._identify_milestones(sorted_docs)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error performing timeline analysis: {str(e)}")
            raise
    
    def _compare_documents(self, doc1: Document, doc2: Document) -> Dict[str, Any]:
        """
        Compare two documents across multiple dimensions.
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            Comparison results
        """
        try:
            # Use relationship mapper for basic comparison
            relationship = self.relationship_mapper._analyze_pair_relationship(doc1, doc2)
            
            comparison = {
                "document_pair": [doc1.title, doc2.title],
                "similarity_score": relationship["confidence"] if relationship else 0.0,
                "relationship_type": relationship["relationship_type"] if relationship else "unrelated",
                "comparison_metrics": relationship["metrics"] if relationship else {}
            }
            
            # Add additional comparison dimensions
            comparison["content_overlap"] = self._calculate_content_overlap(doc1, doc2)
            comparison["structural_comparison"] = self._compare_document_structure(doc1, doc2)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing documents {doc1.id} and {doc2.id}: {str(e)}")
            return {
                "document_pair": [doc1.title, doc2.title],
                "similarity_score": 0.0,
                "relationship_type": "error",
                "comparison_metrics": {}
            }
    
    def _extract_topics_from_documents(self, documents: List[Document]) -> Dict[str, int]:
        """
        Extract topics and their frequency from documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary of topics and their frequency
        """
        topics = {}
        
        for doc in documents:
            # Simple keyword extraction (can be enhanced with NLP)
            words = set()
            if doc.title:
                words.update(doc.title.lower().split())
            if doc.summary:
                words.update(doc.summary.lower().split())
            
            # Get chunks for more content
            chunks = self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == doc.id
            ).all()
            
            for chunk in chunks:
                if chunk.content:
                    words.update(chunk.content.lower().split()[:50])  # Limit to first 50 words
            
            # Filter and count meaningful words (simple approach)
            for word in words:
                if len(word) > 3 and word.isalpha():  # Basic filtering
                    topics[word] = topics.get(word, 0) + 1
        
        return topics
    
    def _synthesize_topics(self, topics: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        Synthesize topics from frequency analysis.
        
        Args:
            topics: Dictionary of topics and frequencies
            
        Returns:
            List of synthesized topics
        """
        synthesized = []
        
        # Group related topics (simplified approach)
        for topic, frequency in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:20]:
            synthesized.append({
                "topic": topic,
                "frequency": frequency,
                "importance": "high" if frequency > 2 else "medium",
                "coverage": "comprehensive" if frequency > 3 else "partial"
            })
        
        return synthesized
    
    def _identify_complementary_insights(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Identify complementary information across documents.
        
        Args:
            documents: List of documents
            
        Returns:
            List of complementary insights
        """
        insights = []
        
        # Analyze relationships for complementary patterns
        for doc in documents:
            relationships = self.relationship_mapper.analyze_document_relationships(doc.id)
            complementary_rels = [r for r in relationships.get("relationships", []) 
                               if r["relationship_type"] == RelationshipType.COMPLEMENTARY.value]
            
            for comp_rel in complementary_rels:
                if comp_rel["related_document_id"] in [d.id for d in documents]:
                    insights.append({
                        "documents": [doc.title, comp_rel["related_document_title"]],
                        "complementary_aspect": "information_coverage",
                        "description": f"{doc.title} and {comp_rel['related_document_title']} provide complementary information",
                        "confidence": comp_rel["confidence"]
                    })
        
        return insights
    
    def _identify_knowledge_gaps(self, documents: List[Document], topics: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        Identify knowledge gaps in the document set.
        
        Args:
            documents: List of documents
            topics: Extracted topics
            
        Returns:
            List of identified gaps
        """
        gaps = []
        
        # Simple gap detection based on topic coverage
        high_freq_topics = [t for t, f in topics.items() if f > 2]
        
        if len(high_freq_topics) < 5:
            gaps.append({
                "gap_type": "breadth",
                "description": "Limited topic breadth in document set",
                "severity": "medium"
            })
        
        # Check for document relationships coverage
        relationship_graph = self.relationship_mapper.create_relationship_graph([d.id for d in documents])
        if len(relationship_graph["edges"]) < len(documents) - 1:
            gaps.append({
                "gap_type": "connectivity",
                "description": "Poor connectivity between documents",
                "severity": "low"
            })
        
        return gaps
    
    def _generate_unified_summary(self, documents: List[Document]) -> str:
        """
        Generate a unified summary of the document set.
        
        Args:
            documents: List of documents
            
        Returns:
            Unified summary text
        """
        # Simple concatenation of document summaries
        summaries = [doc.summary for doc in documents if doc.summary]
        
        if not summaries:
            # Fallback to titles
            summaries = [doc.title for doc in documents]
        
        return " ".join(summaries)[:500]  # Limit length
    
    def _analyze_content_contradictions(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Analyze content for potential contradictions.
        
        Args:
            documents: List of documents
            
        Returns:
            List of potential contradictions
        """
        contradictions = []
        
        # Simple contradiction detection based on relationship analysis
        for doc in documents:
            relationships = self.relationship_mapper.analyze_document_relationships(doc.id)
            conflicting_rels = [r for r in relationships.get("relationships", []) 
                              if r["relationship_type"] == RelationshipType.CONFLICTING.value]
            
            for conflict in conflicting_rels:
                if conflict["related_document_id"] in [d.id for d in documents]:
                    contradictions.append({
                        "documents": [doc.title, conflict["related_document_title"]],
                        "contradiction_type": "content_conflict",
                        "confidence": conflict["confidence"],
                        "evidence": "High topic overlap with low content similarity suggests potential contradiction"
                    })
        
        return contradictions
    
    def _calculate_conflict_confidence(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate confidence scores for conflict detection.
        
        Args:
            analysis: Conflict analysis results
            
        Returns:
            Confidence scores
        """
        conflicts = analysis.get("potential_conflicts", [])
        contradictions = analysis.get("contradiction_areas", [])
        
        total_checks = len(conflicts) + len(contradictions)
        if total_checks == 0:
            return {"overall_confidence": 0.0}
        
        avg_confidence = sum([c.get("confidence", 0.0) for c in conflicts]) / max(len(conflicts), 1)
        
        return {
            "overall_confidence": avg_confidence,
            "conflict_count": len(conflicts),
            "contradiction_count": len(contradictions)
        }
    
    def _get_expected_topic_coverage(self, documents: List[Document]) -> set:
        """
        Get expected topic coverage based on document titles and summaries.
        
        Args:
            documents: List of documents
            
        Returns:
            Set of expected topics
        """
        expected_topics = set()
        
        for doc in documents:
            if doc.title:
                expected_topics.update(doc.title.lower().split())
            if doc.summary:
                expected_topics.update(doc.summary.lower().split()[:10])
        
        return {t for t in expected_topics if len(t) > 3}
    
    def _analyze_coverage_depth(self, documents: List[Document], topics: Dict[str, int]) -> Dict[str, Any]:
        """
        Analyze depth of topic coverage across documents.
        
        Args:
            documents: List of documents
            topics: Extracted topics
            
        Returns:
            Coverage depth analysis
        """
        coverage = {
            "total_topics": len(topics),
            "well_covered_topics": len([t for t, f in topics.items() if f > 2]),
            "partially_covered_topics": len([t for t, f in topics.items() if f == 2]),
            "lightly_covered_topics": len([t for t, f in topics.items() if f == 1]),
            "coverage_score": len([t for t, f in topics.items() if f > 1]) / max(len(topics), 1)
        }
        
        return coverage
    
    def _generate_gap_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations to address identified gaps.
        
        Args:
            analysis: Gap analysis results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        gaps = analysis.get("identified_gaps", [])
        coverage = analysis.get("coverage_analysis", {})
        
        for gap in gaps:
            if gap["gap_type"] == "topic_coverage":
                recommendations.append({
                    "type": "content_addition",
                    "priority": "medium",
                    "description": f"Add content covering topic: {gap['topic']}",
                    "impact": "Improves topic coverage breadth"
                })
        
        if coverage.get("coverage_score", 0) < 0.5:
            recommendations.append({
                "type": "content_depth",
                "priority": "high",
                "description": "Add more detailed content on existing topics",
                "impact": "Improves topic coverage depth"
            })
        
        return recommendations
    
    def _extract_key_contributions(self, doc: Document) -> List[str]:
        """
        Extract key contributions from a document.
        
        Args:
            doc: Document to analyze
            
        Returns:
            List of key contributions
        """
        contributions = []
        
        if doc.summary:
            # Simple extraction from summary (first few sentences)
            sentences = doc.summary.split('.')[:3]
            contributions.extend([s.strip() for s in sentences if s.strip()])
        
        if not contributions:
            contributions.append(f"Document: {doc.title}")
        
        return contributions
    
    def _analyze_evolution_patterns(self, sorted_docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Analyze evolution patterns across documents.
        
        Args:
            sorted_docs: Documents sorted by creation date
            
        Returns:
            List of evolution patterns
        """
        patterns = []
        
        if len(sorted_docs) < 2:
            return patterns
        
        # Simple pattern detection based on title and summary changes
        for i in range(1, len(sorted_docs)):
            prev_doc = sorted_docs[i-1]
            curr_doc = sorted_docs[i]
            
            pattern = {
                "from_document": prev_doc.title,
                "to_document": curr_doc.title,
                "pattern_type": "content_evolution",
                "description": f"Evolution from {prev_doc.title} to {curr_doc.title}",
                "confidence": 0.7  # Placeholder
            }
            
            patterns.append(pattern)
        
        return patterns
    
    def _identify_milestones(self, sorted_docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Identify key milestones in the document timeline.
        
        Args:
            sorted_docs: Documents sorted by creation date
            
        Returns:
            List of milestones
        """
        milestones = []
        
        if not sorted_docs:
            return milestones
        
        # First document is a milestone
        milestones.append({
            "document": sorted_docs[0].title,
            "milestone_type": "foundation",
            "description": "First document in the series",
            "timestamp": sorted_docs[0].created_at.isoformat()
        })
        
        # Last document is a milestone
        if len(sorted_docs) > 1:
            milestones.append({
                "document": sorted_docs[-1].title,
                "milestone_type": "latest",
                "description": "Most recent document",
                "timestamp": sorted_docs[-1].created_at.isoformat()
            })
        
        return milestones
    
    def _calculate_content_overlap(self, doc1: Document, doc2: Document) -> float:
        """
        Calculate content overlap between two documents.
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            Content overlap score (0-1)
        """
        try:
            # Simple word overlap calculation
            words1 = set()
            words2 = set()
            
            if doc1.title:
                words1.update(doc1.title.lower().split())
            if doc1.summary:
                words1.update(doc1.summary.lower().split()[:20])
            
            if doc2.title:
                words2.update(doc2.title.lower().split())
            if doc2.summary:
                words2.update(doc2.summary.lower().split()[:20])
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating content overlap: {str(e)}")
            return 0.0
    
    def _compare_document_structure(self, doc1: Document, doc2: Document) -> Dict[str, Any]:
        """
        Compare document structure between two documents.
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            Structure comparison results
        """
        comparison = {
            "similarity_score": 0.0,
            "structural_elements": {}
        }
        
        try:
            # Compare chunk counts
            chunks1 = self.db.query(DocumentChunk).filter(DocumentChunk.document_id == doc1.id).count()
            chunks2 = self.db.query(DocumentChunk).filter(DocumentChunk.document_id == doc2.id).count()
            
            chunk_similarity = 1 - abs(chunks1 - chunks2) / max(chunks1, chunks2, 1)
            
            # Compare section structure
            sections1 = self.db.query(DocumentChunk.section_title).filter(
                DocumentChunk.document_id == doc1.id,
                DocumentChunk.section_title.isnot(None)
            ).distinct().count()
            
            sections2 = self.db.query(DocumentChunk.section_title).filter(
                DocumentChunk.document_id == doc2.id,
                DocumentChunk.section_title.isnot(None)
            ).distinct().count()
            
            section_similarity = 1 - abs(sections1 - sections2) / max(sections1, sections2, 1)
            
            comparison["similarity_score"] = (chunk_similarity + section_similarity) / 2
            comparison["structural_elements"] = {
                "chunk_count_similarity": chunk_similarity,
                "section_count_similarity": section_similarity,
                "doc1_chunks": chunks1,
                "doc2_chunks": chunks2,
                "doc1_sections": sections1,
                "doc2_sections": sections2
            }
            
        except Exception as e:
            logger.error(f"Error comparing document structure: {str(e)}")
        
        return comparison
    
    def _generate_comparative_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate comparative insights from analysis results.
        
        Args:
            analysis: Comparative analysis results
            
        Returns:
            List of insights
        """
        insights = []
        
        comparisons = analysis.get("comparisons", [])
        if not comparisons:
            insights.append("No significant comparisons available")
            return insights
        
        # Calculate average similarity
        avg_similarity = sum(c["similarity_score"] for c in comparisons) / len(comparisons)
        
        if avg_similarity > 0.7:
            insights.append("Documents show high similarity and likely cover overlapping content")
        elif avg_similarity < 0.3:
            insights.append("Documents show low similarity and cover distinct topics")
        else:
            insights.append("Documents show moderate similarity with some overlapping content")
        
        # Check for relationship patterns
        rel_types = [c["relationship_type"] for c in comparisons]
        if "conflicting" in rel_types:
            insights.append("Some documents may contain conflicting information")
        
        if "complementary" in rel_types:
            insights.append("Some documents provide complementary information")
        
        return insights