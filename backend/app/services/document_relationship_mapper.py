"""
Document Relationship Mapper Service
Identifies and maps relationships between documents for enhanced search and analysis.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..database import get_db
from ..models import Document, DocumentChunk, Embedding

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Types of relationships between documents."""
    SIMILAR_CONTENT = "similar_content"
    COMPLEMENTARY = "complementary"
    CONFLICTING = "conflicting"
    REFERENTIAL = "referential"
    HIERARCHICAL = "hierarchical"
    TEMPORAL = "temporal"


class DocumentRelationshipMapper:
    """
    Service for identifying and mapping relationships between documents.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.similarity_threshold = 0.7
        self.complementary_threshold = 0.5
        self.conflict_threshold = 0.3
    
    def analyze_document_relationships(self, document_id: int) -> Dict[str, Any]:
        """
        Analyze relationships for a specific document with all other documents.
        
        Args:
            document_id: ID of the document to analyze
            
        Returns:
            Dictionary containing relationship analysis results
        """
        try:
            # Get the target document
            target_doc = self.db.query(Document).filter(Document.id == document_id).first()
            if not target_doc:
                raise ValueError(f"Document {document_id} not found")
            
            # Get all other documents
            other_docs = self.db.query(Document).filter(Document.id != document_id).all()
            
            relationships = {
                "document_id": document_id,
                "document_title": target_doc.title,
                "relationships": [],
                "summary": {
                    "total_relationships": 0,
                    "similar_content": 0,
                    "complementary": 0,
                    "conflicting": 0,
                    "referential": 0
                }
            }
            
            for other_doc in other_docs:
                relationship_analysis = self._analyze_pair_relationship(target_doc, other_doc)
                if relationship_analysis:
                    relationships["relationships"].append(relationship_analysis)
                    relationships["summary"]["total_relationships"] += 1
                    
                    # Update summary counts
                    rel_type = relationship_analysis["relationship_type"]
                    if rel_type == RelationshipType.SIMILAR_CONTENT.value:
                        relationships["summary"]["similar_content"] += 1
                    elif rel_type == RelationshipType.COMPLEMENTARY.value:
                        relationships["summary"]["complementary"] += 1
                    elif rel_type == RelationshipType.CONFLICTING.value:
                        relationships["summary"]["conflicting"] += 1
                    elif rel_type == RelationshipType.REFERENTIAL.value:
                        relationships["summary"]["referential"] += 1
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error analyzing document relationships for {document_id}: {str(e)}")
            raise
    
    def _analyze_pair_relationship(self, doc1: Document, doc2: Document) -> Optional[Dict[str, Any]]:
        """
        Analyze relationship between two documents.
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            Relationship analysis or None if no significant relationship
        """
        try:
            # Get embeddings for both documents
            doc1_embeddings = self._get_document_embeddings(doc1.id)
            doc2_embeddings = self._get_document_embeddings(doc2.id)
            
            if not doc1_embeddings or not doc2_embeddings:
                return None
            
            # Calculate similarity metrics
            content_similarity = self._calculate_content_similarity(doc1_embeddings, doc2_embeddings)
            topic_overlap = self._calculate_topic_overlap(doc1, doc2)
            structural_similarity = self._calculate_structural_similarity(doc1, doc2)
            
            # Determine relationship type
            relationship_type, confidence = self._determine_relationship_type(
                content_similarity, topic_overlap, structural_similarity
            )
            
            if confidence < 0.3:  # Minimum confidence threshold
                return None
            
            return {
                "related_document_id": doc2.id,
                "related_document_title": doc2.title,
                "relationship_type": relationship_type.value,
                "confidence": confidence,
                "metrics": {
                    "content_similarity": content_similarity,
                    "topic_overlap": topic_overlap,
                    "structural_similarity": structural_similarity
                },
                "evidence": self._gather_relationship_evidence(doc1, doc2, relationship_type)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing relationship between {doc1.id} and {doc2.id}: {str(e)}")
            return None
    
    def _get_document_embeddings(self, document_id: int) -> List[np.ndarray]:
        """
        Get all embeddings for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of embedding vectors
        """
        try:
            chunks = self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).all()
            
            if not chunks:
                return []
            
            embeddings = []
            for chunk in chunks:
                embedding = self.db.query(Embedding).filter(
                    Embedding.chunk_id == chunk.id
                ).first()
                
                if embedding and embedding.embedding_vector:
                    # Convert embedding vector to numpy array
                    if isinstance(embedding.embedding_vector, list):
                        emb_array = np.array(embedding.embedding_vector)
                    else:
                        emb_array = np.array(embedding.embedding_vector)
                    embeddings.append(emb_array)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings for document {document_id}: {str(e)}")
            return []
    
    def _calculate_content_similarity(self, emb1: List[np.ndarray], emb2: List[np.ndarray]) -> float:
        """
        Calculate content similarity between two sets of embeddings.
        
        Args:
            emb1: First set of embeddings
            emb2: Second set of embeddings
            
        Returns:
            Similarity score (0-1)
        """
        if not emb1 or not emb2:
            return 0.0
        
        try:
            # Use maximum similarity between any pair of embeddings
            max_similarity = 0.0
            for e1 in emb1:
                for e2 in emb2:
                    similarity = self._cosine_similarity(e1, e2)
                    max_similarity = max(max_similarity, similarity)
            
            return max_similarity
            
        except Exception as e:
            logger.error(f"Error calculating content similarity: {str(e)}")
            return 0.0
    
    def _calculate_topic_overlap(self, doc1: Document, doc2: Document) -> float:
        """
        Calculate topic overlap between two documents.
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            Topic overlap score (0-1)
        """
        try:
            # Simple keyword-based topic overlap
            words1 = set(doc1.title.lower().split() + (doc1.summary or "").lower().split())
            words2 = set(doc2.title.lower().split() + (doc2.summary or "").lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating topic overlap: {str(e)}")
            return 0.0
    
    def _calculate_structural_similarity(self, doc1: Document, doc2: Document) -> float:
        """
        Calculate structural similarity between two documents.
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            Structural similarity score (0-1)
        """
        try:
            # Compare document structure based on chunk patterns
            chunks1 = self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == doc1.id
            ).all()
            
            chunks2 = self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == doc2.id
            ).all()
            
            if not chunks1 or not chunks2:
                return 0.0
            
            # Compare chunk length distributions
            lengths1 = [len(chunk.content) for chunk in chunks1]
            lengths2 = [len(chunk.content) for chunk in chunks2]
            
            avg_len1 = sum(lengths1) / len(lengths1)
            avg_len2 = sum(lengths2) / len(lengths2)
            
            length_similarity = 1 - abs(avg_len1 - avg_len2) / max(avg_len1, avg_len2)
            
            # Compare section structure (if available)
            sections1 = [chunk.section_title for chunk in chunks1 if chunk.section_title]
            sections2 = [chunk.section_title for chunk in chunks2 if chunk.section_title]
            
            section_similarity = len(set(sections1).intersection(set(sections2))) / max(len(set(sections1)), len(set(sections2)), 1)
            
            return (length_similarity + section_similarity) / 2
            
        except Exception as e:
            logger.error(f"Error calculating structural similarity: {str(e)}")
            return 0.0
    
    def _determine_relationship_type(self, content_similarity: float, topic_overlap: float, 
                                   structural_similarity: float) -> Tuple[RelationshipType, float]:
        """
        Determine the type of relationship between documents.
        
        Args:
            content_similarity: Content similarity score
            topic_overlap: Topic overlap score
            structural_similarity: Structural similarity score
            
        Returns:
            Tuple of (relationship_type, confidence)
        """
        # Calculate overall similarity score
        overall_similarity = (content_similarity + topic_overlap + structural_similarity) / 3
        
        if content_similarity > self.similarity_threshold:
            return RelationshipType.SIMILAR_CONTENT, overall_similarity
        elif topic_overlap > self.complementary_threshold and content_similarity < 0.3:
            return RelationshipType.COMPLEMENTARY, overall_similarity
        elif topic_overlap > 0.7 and content_similarity < self.conflict_threshold:
            return RelationshipType.CONFLICTING, overall_similarity
        elif structural_similarity > 0.6:
            return RelationshipType.REFERENTIAL, overall_similarity
        else:
            return RelationshipType.SIMILAR_CONTENT, overall_similarity
    
    def _gather_relationship_evidence(self, doc1: Document, doc2: Document, 
                                    relationship_type: RelationshipType) -> List[str]:
        """
        Gather evidence for the identified relationship.
        
        Args:
            doc1: First document
            doc2: Second document
            relationship_type: Type of relationship
            
        Returns:
            List of evidence strings
        """
        evidence = []
        
        if relationship_type == RelationshipType.SIMILAR_CONTENT:
            evidence.append(f"Both documents cover similar topics: '{doc1.title}' and '{doc2.title}'")
        
        elif relationship_type == RelationshipType.COMPLEMENTARY:
            evidence.append(f"Documents provide complementary information on related topics")
        
        elif relationship_type == RelationshipType.CONFLICTING:
            evidence.append(f"Documents may contain conflicting information on overlapping topics")
        
        elif relationship_type == RelationshipType.REFERENTIAL:
            evidence.append(f"Documents share similar structural patterns")
        
        return evidence
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score
        """
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    def get_related_documents(self, document_id: int, relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get documents related to a specific document.
        
        Args:
            document_id: Document ID
            relationship_type: Optional filter for relationship type
            
        Returns:
            List of related documents with relationship details
        """
        try:
            analysis = self.analyze_document_relationships(document_id)
            relationships = analysis.get("relationships", [])
            
            if relationship_type:
                relationships = [r for r in relationships if r["relationship_type"] == relationship_type]
            
            # Sort by confidence (descending)
            relationships.sort(key=lambda x: x["confidence"], reverse=True)
            
            return relationships[:10]  # Return top 10 related documents
            
        except Exception as e:
            logger.error(f"Error getting related documents for {document_id}: {str(e)}")
            return []
    
    def create_relationship_graph(self, document_ids: List[int]) -> Dict[str, Any]:
        """
        Create a relationship graph for multiple documents.
        
        Args:
            document_ids: List of document IDs to include in graph
            
        Returns:
            Graph structure with nodes and edges
        """
        try:
            graph = {
                "nodes": [],
                "edges": []
            }
            
            # Add nodes
            for doc_id in document_ids:
                doc = self.db.query(Document).filter(Document.id == doc_id).first()
                if doc:
                    graph["nodes"].append({
                        "id": doc.id,
                        "label": doc.title,
                        "type": "document"
                    })
            
            # Add edges (relationships)
            for i, doc_id1 in enumerate(document_ids):
                for doc_id2 in document_ids[i+1:]:
                    doc1 = self.db.query(Document).filter(Document.id == doc_id1).first()
                    doc2 = self.db.query(Document).filter(Document.id == doc_id2).first()
                    
                    if doc1 and doc2:
                        relationship = self._analyze_pair_relationship(doc1, doc2)
                        if relationship and relationship["confidence"] > 0.5:
                            graph["edges"].append({
                                "source": doc_id1,
                                "target": doc_id2,
                                "type": relationship["relationship_type"],
                                "weight": relationship["confidence"],
                                "label": relationship["relationship_type"]
                            })
            
            return graph
            
        except Exception as e:
            logger.error(f"Error creating relationship graph: {str(e)}")
            return {"nodes": [], "edges": []}