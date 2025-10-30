
"""
Adaptive Chunker Service
Implements specialized chunking strategies based on document type and structure patterns.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Document, DocumentChunk
from .document_type_classifier import DocumentType, DocumentStructurePattern, DocumentTypeClassifier

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SECTION_BASED = "section_based"
    CLAUSE_BASED = "clause_based"
    PROCEDURE_BASED = "procedure_based"
    SEMANTIC_BASED = "semantic_based"
    HYBRID = "hybrid"


class ChunkBoundary(Enum):
    """Types of chunk boundaries."""
    SECTION_HEADER = "section_header"
    CLAUSE_BOUNDARY = "clause_boundary"
    PROCEDURAL_STEP = "procedural_step"
    SEMANTIC_BREAK = "semantic_break"
    FIXED_LENGTH = "fixed_length"


class AdaptiveChunker:
    """
    Service for adaptive document chunking based on document type and structure.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.type_classifier = DocumentTypeClassifier(db_session)
        
        # Default chunking parameters
        self.default_chunk_size = 1000
        self.default_chunk_overlap = 200
        
        # Strategy-specific parameters
        self.strategy_parameters = {
            ChunkingStrategy.FIXED_SIZE: {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "description": "Fixed-size chunks for general documents"
            },
            ChunkingStrategy.SECTION_BASED: {
                "chunk_size": 1500,
                "chunk_overlap": 100,
                "description": "Section-based chunks preserving document structure"
            },
            ChunkingStrategy.CLAUSE_BASED: {
                "chunk_size": 800,
                "chunk_overlap": 50,
                "description": "Clause-based chunks for legal documents"
            },
            ChunkingStrategy.PROCEDURE_BASED: {
                "chunk_size": 600,
                "chunk_overlap": 100,
                "description": "Procedure-based chunks for technical manuals"
            },
            ChunkingStrategy.SEMANTIC_BASED: {
                "chunk_size": 1200,
                "chunk_overlap": 150,
                "description": "Semantic-based chunks for academic content"
            },
            ChunkingStrategy.HYBRID: {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "description": "Hybrid approach combining multiple strategies"
            }
        }
        
        # Document type to strategy mapping
        self.type_strategy_mapping = {
            DocumentType.LEGAL: ChunkingStrategy.CLAUSE_BASED,
            DocumentType.TECHNICAL: ChunkingStrategy.PROCEDURE_BASED,
            DocumentType.ACADEMIC: ChunkingStrategy.SECTION_BASED,
            DocumentType.BUSINESS: ChunkingStrategy.SEMANTIC_BASED,
            DocumentType.MEDICAL: ChunkingStrategy.PROCEDURE_BASED,
            DocumentType.SCIENTIFIC: ChunkingStrategy.SECTION_BASED,
            DocumentType.GOVERNMENT: ChunkingStrategy.SECTION_BASED,
            DocumentType.GENERAL: ChunkingStrategy.FIXED_SIZE
        }
    
    def chunk_document_adaptively(self, document_id: int, strategy: Optional[ChunkingStrategy] = None) -> Dict[str, Any]:
        """
        Chunk a document using adaptive strategy based on document type and structure.
        
        Args:
            document_id: ID of the document to chunk
            strategy: Optional specific strategy to use (auto-detected if None)
            
        Returns:
            Chunking results with strategy details and chunks
        """
        try:
            # Get the document
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Classify document if strategy not specified
            if strategy is None:
                classification = self.type_classifier.classify_document(document_id)
                strategy = self._select_optimal_strategy(classification)
            
            # Get strategy parameters
            strategy_params = self.strategy_parameters.get(strategy, self.strategy_parameters[ChunkingStrategy.FIXED_SIZE])
            
            # Perform chunking based on strategy
            if strategy == ChunkingStrategy.FIXED_SIZE:
                chunks = self._chunk_fixed_size(document, strategy_params)
            elif strategy == ChunkingStrategy.SECTION_BASED:
                chunks = self._chunk_section_based(document, strategy_params)
            elif strategy == ChunkingStrategy.CLAUSE_BASED:
                chunks = self._chunk_clause_based(document, strategy_params)
            elif strategy == ChunkingStrategy.PROCEDURE_BASED:
                chunks = self._chunk_procedure_based(document, strategy_params)
            elif strategy == ChunkingStrategy.SEMANTIC_BASED:
                chunks = self._chunk_semantic_based(document, strategy_params)
            elif strategy == ChunkingStrategy.HYBRID:
                chunks = self._chunk_hybrid(document, strategy_params)
            else:
                raise ValueError(f"Unsupported chunking strategy: {strategy}")
            
            # Analyze chunk quality
            quality_metrics = self._analyze_chunk_quality(chunks, strategy)
            
            return {
                "document_id": document_id,
                "document_title": document.title,
                "strategy_used": strategy.value,
                "strategy_parameters": strategy_params,
                "chunks_generated": len(chunks),
                "chunks": chunks,
                "quality_metrics": quality_metrics,
                "recommendations": self._generate_chunking_recommendations(quality_metrics, strategy)
            }
            
        except Exception as e:
            logger.error(f"Error chunking document {document_id}: {str(e)}")
            raise
    
    def _select_optimal_strategy(self, classification: Dict[str, Any]) -> ChunkingStrategy:
        """
        Select optimal chunking strategy based on document classification.
        
        Args:
            classification: Document classification results
            
        Returns:
            Optimal chunking strategy
        """
        # Get primary document type
        type_classification = classification.get("type_classification", [])
        if not type_classification:
            return ChunkingStrategy.FIXED_SIZE
        
        primary_type = type_classification[0]
        doc_type = DocumentType(primary_type["document_type"])
        
        # Get structure patterns
        structure_patterns = classification.get("structure_patterns", [])
        
        # Select strategy based on type and patterns
        base_strategy = self.type_strategy_mapping.get(doc_type, ChunkingStrategy.FIXED_SIZE)
        
        # Adjust strategy based on structure patterns
        for pattern in structure_patterns:
            if pattern["confidence"] > 0.7:
                pattern_type = DocumentStructurePattern(pattern["pattern_type"])
                
                if pattern_type == DocumentStructurePattern.PROCEDURAL:
                    if base_strategy != ChunkingStrategy.PROCEDURE_BASED:
                        base_strategy = ChunkingStrategy.HYBRID
                
                elif pattern_type == DocumentStructurePattern.SECTION_HIERARCHY:
                    if base_strategy != ChunkingStrategy.SECTION_BASED:
                        base_strategy = ChunkingStrategy.SECTION_BASED
        
        return base_strategy
    
    def _chunk_fixed_size(self, document: Document, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk document using fixed-size strategy.
        
        Args:
            document: Document to chunk
            params: Strategy parameters
            
        Returns:
            List of chunks
        """
        chunks = []
        chunk_size = params.get("chunk_size", self.default_chunk_size)
        overlap = params.get("chunk_overlap", self.default_chunk_overlap)
        
        # Simple fixed-size chunking
        content = self._get_document_content(document)
        words = content.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "content": chunk_text,
                "boundary_type": ChunkBoundary.FIXED_LENGTH.value,
                "position": i,
                "word_count": len(chunk_words),
                "section_title": None,
                "metadata": {
                    "strategy": "fixed_size",
                    "chunk_index": len(chunks),
                    "overlap_size": overlap if i > 0 else 0
                }
            })
        
        return chunks
    
    def _chunk_section_based(self, document: Document, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk document using section-based strategy.
        
        Args:
            document: Document to chunk
            params: Strategy parameters
            
        Returns:
            List of chunks
        """
        chunks = []
        
        # Get existing chunks with section information
        existing_chunks = self.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document.id
        ).order_by(DocumentChunk.chunk_index).all()
        
        if not existing_chunks:
            # Fallback to fixed-size if no section information
            return self._chunk_fixed_size(document, params)
        
        # Group chunks by section
        current_section = None
        current_content = []
        
        for chunk in existing_chunks:
            section_title = chunk.section_title
            
            if section_title != current_section:
                # Save previous section
                if current_content:
                    chunks.append({
                        "content": " ".join(current_content),
                        "boundary_type": ChunkBoundary.SECTION_HEADER.value,
                        "position": len(chunks),
                        "word_count": len(" ".join(current_content).split()),
                        "section_title": current_section,
                        "metadata": {
                            "strategy": "section_based",
                            "section_boundary": True
                        }
                    })
                
                # Start new section
                current_section = section_title
                current_content = []
            
            if chunk.content:
                current_content.append(chunk.content)
        
        # Add final section
        if current_content:
            chunks.append({
                "content": " ".join(current_content),
                "boundary_type": ChunkBoundary.SECTION_HEADER.value,
                "position": len(chunks),
                "word_count": len(" ".join(current_content).split()),
                "section_title": current_section,
                "metadata": {
                    "strategy": "section_based",
                    "section_boundary": True
                }
            })
        
        return chunks
    
    def _chunk_clause_based(self, document: Document, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk document using clause-based strategy for legal documents.
        
        Args:
            document: Document to chunk
            params: Strategy parameters
            
        Returns:
            List of chunks
        """
        chunks = []
        content = self._get_document_content(document)
        
        # Legal clause patterns
        clause_patterns = [
            r'\b(?:SECTION|ARTICLE)\s+\d+[.:]',
            r'\bCLAUSE\s+\d+[.:]',
            r'\b\d+\.\s+',  # Numbered clauses
            r'\b\([a-z]\)\s+',  # Lettered subclauses
        ]
        
        # Split content by clause boundaries
        clauses = []
        current_clause = ""
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts a new clause
            is_clause_start = any(re.match(pattern, line, re.IGNORECASE) for pattern in clause_patterns)
            
            if is_clause_start and current_clause:
                clauses.append(current_clause)
                current_clause = line
            else:
                if current_clause:
                    current_clause += " " + line
                else:
                    current_clause = line
        
        if current_clause:
            clauses.append(current_clause)
        
        # Create chunks from clauses
        for i, clause in enumerate(clauses):
            chunks.append({
                "content": clause,
                "boundary_type": ChunkBoundary.CLAUSE_BOUNDARY.value,
                "position": i,
                "word_count": len(clause.split()),
                "section_title": f"Clause {i + 1}",
                "metadata": {
                    "strategy": "clause_based",
                    "clause_index": i,
                    "clause_type": self._identify_clause_type(clause)
                }
            })
        
        return chunks
    
    def _chunk_procedure_based(self, document: Document, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk document using procedure-based strategy for technical documents.
        
        Args:
            document: Document to chunk
            params: Strategy parameters
            
        Returns:
            List of chunks
        """
        chunks = []
        content = self._get_document_content(document)
        
        # Procedure step patterns
        step_patterns = [
            r'\b(?:Step|Procedure|Method)\s+\d+[.:]',
            r'\b\d+\.\s+',  # Numbered steps
            r'\b•\s+',  # Bullet points
            r'\b-\s+',  # Dash points
        ]
        
        # Split content by procedure steps
        steps = []
        current_step = ""
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts a new step
            is_step_start = any(re.match(pattern, line) for pattern in step_patterns)
            
            if is_step_start and current_step:
                steps.append(current_step)
                current_step = line
            else:
                if current_step:
                    current_step += " " + line
                else:
                    current_step = line
        
        if current_step:
            steps.append(current_step)
        
        # Create chunks from steps
        for i, step in enumerate(steps):
            chunks.append({
                "content": step,
                "boundary_type": ChunkBoundary.PROCEDURAL_STEP.value,
                "position": i,
                "word_count": len(step.split()),
                "section_title": f"Step {i + 1}",
                "metadata": {
                    "strategy": "procedure_based",
                    "step_index": i,
                    "step_type": self._identify_step_type(step)
                }
            })
        
        return chunks
    
    def _chunk_semantic_based(self, document: Document, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk document using semantic-based strategy.
        
        Args:
            document: Document to chunk
            params: Strategy parameters
            
        Returns:
            List of chunks
        """
        chunks = []
        content = self._get_document_content(document)
        
        # Semantic break patterns (paragraphs, topic shifts)
        paragraphs = content.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Further split long paragraphs
            if len(paragraph.split()) > params.get("chunk_size", self.default_chunk_size):
                sub_chunks = self._split_long_paragraph(paragraph, params)
                for j, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        "content": sub_chunk,
                        "boundary_type": ChunkBoundary.SEMANTIC_BREAK.value,
                        "position": len(chunks),
                        "word_count": len(sub_chunk.split()),
                        "section_title": f"Paragraph {i + 1}.{j + 1}",
                        "metadata": {
                            "strategy": "semantic_based",
                            "paragraph_index": i,
                            "sub_paragraph_index": j,
                            "semantic_unit": True
                        }
                    })
            else:
                chunks.append({
                    "content": paragraph,
                    "boundary_type": ChunkBoundary.SEMANTIC_BREAK.value,
                    "position": len(chunks),
                    "word_count": len(paragraph.split()),
                    "section_title": f"Paragraph {i + 1}",
                    "metadata": {
                        "strategy": "semantic_based",
                        "paragraph_index": i,
                        "semantic_unit": True
                    }
                })
        
        return chunks
    
    def _chunk_hybrid(self, document: Document, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk document using hybrid strategy combining multiple approaches.
        
        Args:
            document: Document to chunk
            params: Strategy parameters
            
        Returns:
            List of chunks
        """
        # Try section-based first
        section_chunks = self._chunk_section_based(document, params)
        
        if len(section_chunks) > 1:
            return section_chunks
        
        # Fall back to semantic-based
        semantic_chunks = self._chunk_semantic_based(document, params)
        
        if len(semantic_chunks) > 1:
            return semantic_chunks
        
        # Final fallback to fixed-size
        return self._chunk_fixed_size(document, params)
    
    def _get_document_content(self, document: Document) -> str:
        """
        Get complete document content from chunks.
        
        Args:
            document: Document to get content from
            
        Returns:
            Combined document content
        """
        chunks = self.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document.id
        ).order_by(DocumentChunk.chunk_index).all()
        
        content_parts = []
        for chunk in chunks:
            if chunk.content:
                content_parts.append(chunk.content)
        
        return " ".join(content_parts)
    
    def _split_long_paragraph(self, paragraph: str, params: Dict[str, Any]) -> List[str]:
        """
        Split a long paragraph into smaller chunks.
        
        Args:
            paragraph: Paragraph to split
            params: Strategy parameters
            
        Returns:
            List of smaller chunks
        """
        chunk_size = params.get("chunk_size", self.default_chunk_size)
        overlap = params.get("chunk_overlap", self.default_chunk_overlap)
        
        words = paragraph.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunks.append(" ".join(chunk_words))
        
        return chunks
    
    def _identify_clause_type(self, clause: str) -> str:
        """
        Identify the type of legal clause.
        
        Args:
            clause: Clause text
            
        Returns:
            Clause type
        """
        clause_lower = clause.lower()
        
        if any(word in clause_lower for word in ["whereas", "recital"]):
            return "recital"
        elif any(word in clause_lower for word in ["party", "parties"]):
            return "party_definition"
        elif any(word in clause_lower for word in ["term", "duration", "effective date"]):
            return "term"
        elif any(word in clause_lower for word in ["confidential", "proprietary"]):
            return "confidentiality"
        elif any(word in clause_lower for word in ["indemnify", "liability", "warranty"]):
            return "liability"
        elif any(word in clause_lower for word in ["terminate", "breach", "default"]):
            return "termination"
        else:
            return "general"
    
    def _identify_step_type(self, step: str) -> str:
        """
        Identify the type of procedural step.
        
        Args:
            step: Step text
            
        Returns:
            Step type
        """
        step_lower = step.lower()
        
        if any(word in step_lower for word in ["install", "setup", "configure"]):
            return "installation"
        elif any(word in step_lower for word in ["connect", "attach", "plug"]):
            return "connection"
        elif any(word in step_lower for word in ["test", "verify", "check"]):
            return "verification"
        elif any(word in step_lower for word in ["troubleshoot", "debug", "fix"]):
            return "troubleshooting"
        elif any(word in step_lower for word in ["warning", "caution", "danger"]):
            return "safety"
        else:
            return "instruction"
    
    def _analyze_chunk_quality(self, chunks: List[Dict[str, Any]], strategy: ChunkingStrategy) -> Dict[str, Any]:
        """
        Analyze the quality of generated chunks.
        
        Args:
            chunks: Generated chunks
            strategy: Chunking strategy used
            
        Returns:
            Quality metrics
        """
        if not chunks:
            return {
                "average_word_count": 0,
                "chunk_variability": 0,
                "boundary_effectiveness": 0,
                "semantic_coherence": 0,
                "overall_quality": 0
            }
        
        word_counts = [chunk["word_count"] for chunk in chunks]
        avg_word_count = sum(word_counts) / len(word_counts)
        
        # Calculate variability (standard deviation)
        if len(word_counts) > 1:
            mean = avg_word_count
            variance = sum((x - mean) ** 2 for x in word_counts) / len(word_counts)
            chunk_variability = variance ** 0.5 / mean
        else:
            chunk_variability = 0
        
        # Calculate boundary effectiveness
        boundary_types = [chunk["boundary_type"] for chunk in chunks]
        semantic_boundaries = sum(1 for bt in boundary_types 
                                if bt in [ChunkBoundary.SECTION_HEADER.value, 
                                         ChunkBoundary.CLAUSE_BOUNDARY.value,
                                         ChunkBoundary.PROCEDURAL_STEP.value,
                                         ChunkBoundary.SEMANTIC_BREAK.value])
        boundary_effectiveness = semantic_boundaries / len(chunks)
        
        # Estimate semantic coherence (simplified)
        semantic_coherence = 0.7  # Placeholder - could be enhanced with NLP
        
        # Overall quality score
        overall_quality = (
            (min(avg_word_count / 800, 1.0) * 0.3) +  # Word count optimality
            (max(0, 1 - chunk_variability) * 0.2) +   # Consistency
            (boundary_effectiveness * 0.3) +           # Boundary quality
            (semantic_coherence * 0.2)                 # Semantic quality
        )
        
        return {
            "average_word_count": avg_word_count,
            "chunk_variability": chunk_variability,
            "boundary_effectiveness": boundary_effectiveness,
            "semantic_coherence": semantic_coherence,
            "overall_quality": overall_quality,
            "strategy_suitability": self._calculate_strategy_suitability(strategy, chunks)
        }
    
    def _calculate_strategy_suitability(self, strategy: ChunkingStrategy, chunks: List[Dict[str, Any]]) -> float:
        """
        Calculate how suitable the strategy was for the document.
        
        Args:
            strategy: Chunking strategy used
            chunks: Generated chunks
            
        Returns:
            Suitability score (0-1)
        """
        if not chunks:
            return 0.0
        
        # Strategy-specific suitability metrics
        boundary_types = [chunk["boundary_type"] for chunk in chunks]
        
        if strategy == ChunkingStrategy.SECTION_BASED:
            section_boundaries = sum(1 for bt in boundary_types 
                                   if bt == ChunkBoundary.SECTION_HEADER.value)
            return section_boundaries / len(chunks)
        
        elif strategy == ChunkingStrategy.CLAUSE_BASED:
            clause_boundaries = sum(1 for bt in boundary_types 
                                  if bt == ChunkBoundary.CLAUSE_BOUNDARY.value)
            return clause_boundaries / len(chunks)
        
        elif strategy == ChunkingStrategy.PROCEDURE_BASED:
            step_boundaries = sum(1 for bt in boundary_types 
                                if bt == ChunkBoundary.PROCEDURAL_STEP.value)
            return step_boundaries / len(chunks)
        
        elif strategy == ChunkingStrategy.SEMANTIC_BASED:
            semantic_boundaries = sum(1 for bt in boundary_types 
                                    if bt == ChunkBoundary.SEMANTIC_BREAK.value)
            return semantic_boundaries / len(chunks)
        
        else:
            # For fixed-size and hybrid, use general metrics
            word_counts = [chunk["word_count"] for chunk in chunks]
            avg_word_count = sum(word_counts) / len(word_counts)
            return min(avg_word_count / 800, 1.0)  # Optimal around 800 words
    
    def _generate_chunking_recommendations(self, quality_metrics: Dict[str, Any], 
                                         strategy: ChunkingStrategy) -> List[Dict[str, Any]]:
        """
        Generate recommendations for chunking improvement.
        
        Args:
            quality_metrics: Chunk quality analysis
            strategy: Chunking strategy used
            
        Returns:
            List of recommendations
        """
        recommendations = []
        overall_quality = quality_metrics.get("overall_quality", 0)
        boundary_effectiveness = quality_metrics.get("boundary_effectiveness", 0)
        avg_word_count = quality_metrics.get("average_word_count", 0)
        
        if overall_quality < 0.6:
            recommendations.append({
                "type": "strategy_adjustment",
                "priority": "high",
                "description": f"Consider switching from {strategy.value} to alternative strategy",
                "rationale": f"Current strategy shows low effectiveness (score: {overall_quality:.2f})"
            })
        
        if boundary_effectiveness < 0.5:
            recommendations.append({
                "type": "boundary_detection",
                "priority": "medium",
                "description": "Improve boundary detection for better chunk coherence",
                "rationale": f"Only {boundary_effectiveness:.1%} of chunks use semantic boundaries"
            })
        
        if avg_word_count > 1200:
            recommendations.append({
                "type": "chunk_size",
                "priority": "medium",
                "description": "Reduce chunk size for better processing",
                "rationale": f"Average chunk size ({avg_word_count:.0f} words) exceeds optimal range"
            })
        elif avg_word_count < 400:
            recommendations.append({
                "type": "chunk_size",
                "priority": "medium",
                "description": "Increase chunk size for better context",
                "rationale": f"Average chunk size ({avg_word_count:.0f} words) is below optimal range"
            })
        
        return recommendations
    
    def compare_strategies(self, document_id: int) -> Dict[str, Any]:
        """
        Compare different chunking strategies for a document.
        
        Args:
            document_id: ID of the document to compare
            
        Returns:
            Strategy comparison results
        """
        comparison = {
            "document_id": document_id,
            "strategies_tested": [],
            "best_strategy": None,
            "recommendation": None
        }
        
        # Test all strategies
        for strategy in ChunkingStrategy:
            try:
                result = self.chunk_document_adaptively(document_id, strategy)
                comparison["strategies_tested"].append({
                    "strategy": strategy.value,
                    "chunks_generated": result["chunks_generated"],
                    "quality_metrics": result["quality_metrics"],
                    "recommendations": result["recommendations"]
                })
            except Exception as e:
                logger.error(f"Error testing strategy {strategy.value} for document {document_id}: {str(e)}")
                comparison["strategies_tested"].append({
                    "strategy": strategy.value,
                    "error": str(e)
                })
        
        # Find best strategy
        successful_tests = [s for s in comparison["strategies_tested"] if "error" not in s]
        if successful_tests:
            best_strategy = max(successful_tests, 
                              key=lambda x: x["quality_metrics"]["overall_quality"])
            comparison["best_strategy"] = best_strategy["strategy"]
            comparison["recommendation"] = {
                "strategy": best_strategy["strategy"],
                "quality_score": best_strategy["quality_metrics"]["overall_quality"],
                "reason": f"Highest overall quality score ({best_strategy['quality_metrics']['overall_quality']:.2f})"
            }
        
        return comparison