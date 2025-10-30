
"""
Document Type Classifier Service
Classifies documents by type and identifies document structure patterns.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Document, DocumentChunk

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Types of documents that can be classified."""
    LEGAL = "legal"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    BUSINESS = "business"
    MEDICAL = "medical"
    SCIENTIFIC = "scientific"
    GOVERNMENT = "government"
    GENERAL = "general"


class DocumentStructurePattern(Enum):
    """Common document structure patterns."""
    SECTION_HIERARCHY = "section_hierarchy"
    PROCEDURAL = "procedural"
    ARGUMENTATIVE = "argumentative"
    DESCRIPTIVE = "descriptive"
    COMPARATIVE = "comparative"
    NARRATIVE = "narrative"


class DocumentTypeClassifier:
    """
    Service for classifying documents by type and identifying structure patterns.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        
        # Keywords and patterns for document type classification
        self.type_keywords = {
            DocumentType.LEGAL: [
                "law", "legal", "contract", "agreement", "clause", "section", "article",
                "party", "whereas", "hereinafter", "witnesseth", "jurisdiction",
                "indemnification", "confidentiality", "termination", "breach"
            ],
            DocumentType.TECHNICAL: [
                "technical", "specification", "manual", "guide", "installation",
                "configuration", "troubleshooting", "api", "interface", "protocol",
                "system", "component", "module", "function", "parameter"
            ],
            DocumentType.ACADEMIC: [
                "abstract", "introduction", "methodology", "results", "discussion",
                "conclusion", "references", "citation", "hypothesis", "theory",
                "experiment", "analysis", "literature", "review", "thesis"
            ],
            DocumentType.BUSINESS: [
                "business", "strategy", "marketing", "financial", "revenue",
                "profit", "market", "customer", "product", "service", "growth",
                "plan", "report", "analysis", "forecast", "budget"
            ],
            DocumentType.MEDICAL: [
                "medical", "patient", "treatment", "diagnosis", "symptom",
                "therapy", "medication", "clinical", "health", "disease",
                "condition", "procedure", "examination", "prescription"
            ],
            DocumentType.SCIENTIFIC: [
                "scientific", "research", "experiment", "hypothesis", "method",
                "results", "conclusion", "data", "analysis", "statistical",
                "significant", "correlation", "variable", "control"
            ],
            DocumentType.GOVERNMENT: [
                "government", "regulation", "policy", "compliance", "standard",
                "requirement", "guideline", "procedure", "directive", "mandate",
                "authority", "jurisdiction", "legislation", "statute"
            ]
        }
        
        # Structure patterns for different document types
        self.structure_patterns = {
            DocumentType.LEGAL: [
                DocumentStructurePattern.SECTION_HIERARCHY,
                DocumentStructurePattern.ARGUMENTATIVE
            ],
            DocumentType.TECHNICAL: [
                DocumentStructurePattern.PROCEDURAL,
                DocumentStructurePattern.DESCRIPTIVE
            ],
            DocumentType.ACADEMIC: [
                DocumentStructurePattern.ARGUMENTATIVE,
                DocumentStructurePattern.SECTION_HIERARCHY
            ],
            DocumentType.BUSINESS: [
                DocumentStructurePattern.DESCRIPTIVE,
                DocumentStructurePattern.COMPARATIVE
            ],
            DocumentType.MEDICAL: [
                DocumentStructurePattern.PROCEDURAL,
                DocumentStructurePattern.DESCRIPTIVE
            ],
            DocumentType.SCIENTIFIC: [
                DocumentStructurePattern.ARGUMENTATIVE,
                DocumentStructurePattern.SECTION_HIERARCHY
            ],
            DocumentType.GOVERNMENT: [
                DocumentStructurePattern.SECTION_HIERARCHY,
                DocumentStructurePattern.PROCEDURAL
            ]
        }
    
    def classify_document(self, document_id: int) -> Dict[str, Any]:
        """
        Classify a document by type and identify its structure patterns.
        
        Args:
            document_id: ID of the document to classify
            
        Returns:
            Classification results including type and structure patterns
        """
        try:
            # Get the document
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Get document chunks for analysis
            chunks = self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).all()
            
            # Analyze document content
            content_analysis = self._analyze_document_content(document, chunks)
            
            # Classify document type
            type_classification = self._classify_document_type(document, chunks, content_analysis)
            
            # Identify structure patterns
            structure_patterns = self._identify_structure_patterns(document, chunks, content_analysis)
            
            # Generate recommendations
            recommendations = self._generate_classification_recommendations(
                type_classification, structure_patterns
            )
            
            return {
                "document_id": document_id,
                "document_title": document.title,
                "type_classification": type_classification,
                "structure_patterns": structure_patterns,
                "content_analysis": content_analysis,
                "recommendations": recommendations,
                "confidence": self._calculate_classification_confidence(
                    type_classification, structure_patterns
                )
            }
            
        except Exception as e:
            logger.error(f"Error classifying document {document_id}: {str(e)}")
            raise
    
    def _analyze_document_content(self, document: Document, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Analyze document content for classification features.
        
        Args:
            document: Document to analyze
            chunks: Document chunks
            
        Returns:
            Content analysis results
        """
        analysis = {
            "word_count": 0,
            "avg_chunk_length": 0,
            "section_count": 0,
            "keyword_distribution": {},
            "structural_features": {},
            "language_patterns": {}
        }
        
        try:
            # Calculate word count and chunk statistics
            total_words = 0
            chunk_lengths = []
            sections = set()
            
            for chunk in chunks:
                if chunk.content:
                    words = len(chunk.content.split())
                    total_words += words
                    chunk_lengths.append(words)
                
                if chunk.section_title:
                    sections.add(chunk.section_title)
            
            analysis["word_count"] = total_words
            analysis["avg_chunk_length"] = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
            analysis["section_count"] = len(sections)
            
            # Analyze keyword distribution
            analysis["keyword_distribution"] = self._analyze_keyword_distribution(document, chunks)
            
            # Analyze structural features
            analysis["structural_features"] = self._analyze_structural_features(chunks)
            
            # Analyze language patterns
            analysis["language_patterns"] = self._analyze_language_patterns(document, chunks)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing document content: {str(e)}")
            return analysis
    
    def _analyze_keyword_distribution(self, document: Document, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """
        Analyze keyword distribution across document types.
        
        Args:
            document: Document to analyze
            chunks: Document chunks
            
        Returns:
            Keyword distribution by document type
        """
        keyword_distribution = {}
        all_text = ""
        
        # Combine all text for analysis
        if document.title:
            all_text += document.title.lower() + " "
        if document.summary:
            all_text += document.summary.lower() + " "
        
        for chunk in chunks:
            if chunk.content:
                all_text += chunk.content.lower() + " "
        
        # Count keyword occurrences by document type
        for doc_type, keywords in self.type_keywords.items():
            count = 0
            for keyword in keywords:
                count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', all_text))
            keyword_distribution[doc_type.value] = count
        
        return keyword_distribution
    
    def _analyze_structural_features(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Analyze structural features of the document.
        
        Args:
            chunks: Document chunks
            
        Returns:
            Structural features analysis
        """
        features = {
            "hierarchy_depth": 0,
            "procedural_elements": 0,
            "argumentative_elements": 0,
            "descriptive_elements": 0,
            "section_variety": 0
        }
        
        try:
            # Analyze section hierarchy
            section_titles = [chunk.section_title for chunk in chunks if chunk.section_title]
            features["section_variety"] = len(set(section_titles))
            
            # Analyze hierarchy depth based on section title patterns
            max_depth = 0
            for title in section_titles:
                if title:
                    # Simple depth estimation based on numbering patterns
                    depth = self._estimate_section_depth(title)
                    max_depth = max(max_depth, depth)
            
            features["hierarchy_depth"] = max_depth
            
            # Analyze content patterns
            for chunk in chunks:
                if chunk.content:
                    content_lower = chunk.content.lower()
                    
                    # Procedural elements (steps, instructions)
                    if any(word in content_lower for word in ["step", "instruction", "procedure", "method"]):
                        features["procedural_elements"] += 1
                    
                    # Argumentative elements (claims, evidence, conclusions)
                    if any(word in content_lower for word in ["therefore", "consequently", "thus", "evidence"]):
                        features["argumentative_elements"] += 1
                    
                    # Descriptive elements (descriptions, definitions)
                    if any(word in content_lower for word in ["definition", "description", "characteristic", "feature"]):
                        features["descriptive_elements"] += 1
            
            return features
            
        except Exception as e:
            logger.error(f"Error analyzing structural features: {str(e)}")
            return features
    
    def _analyze_language_patterns(self, document: Document, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Analyze language patterns for classification.
        
        Args:
            document: Document to analyze
            chunks: Document chunks
            
        Returns:
            Language pattern analysis
        """
        patterns = {
            "formality_level": 0,
            "technical_terms": 0,
            "legal_terms": 0,
            "academic_terms": 0,
            "sentence_complexity": 0
        }
        
        try:
            all_text = ""
            if document.title:
                all_text += document.title + " "
            if document.summary:
                all_text += document.summary + " "
            
            for chunk in chunks:
                if chunk.content:
                    all_text += chunk.content + " "
            
            # Analyze formality level
            formal_words = ["shall", "must", "required", "obligated", "pursuant"]
            informal_words = ["can", "maybe", "probably", "might", "could"]
            
            formal_count = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', all_text.lower())) 
                              for word in formal_words)
            informal_count = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', all_text.lower())) 
                                for word in informal_words)
            
            patterns["formality_level"] = formal_count / max(formal_count + informal_count, 1)
            
            # Count specialized terms
            for doc_type, keywords in self.type_keywords.items():
                term_count = sum(len(re.findall(r'\b' + re.escape(keyword) + r'\b', all_text.lower())) 
                                for keyword in keywords[:10])  # Use top 10 keywords
                patterns[f"{doc_type.value}_terms"] = term_count
            
            # Simple sentence complexity (average words per sentence)
            sentences = re.split(r'[.!?]+', all_text)
            if sentences:
                avg_words = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
                patterns["sentence_complexity"] = min(avg_words / 20, 1.0)  # Normalize
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing language patterns: {str(e)}")
            return patterns
    
    def _estimate_section_depth(self, section_title: str) -> int:
        """
        Estimate section hierarchy depth based on title patterns.
        
        Args:
            section_title: Section title to analyze
            
        Returns:
            Estimated depth level
        """
        # Check for numbered patterns (1.1, 1.1.1, etc.)
        numbered_pattern = re.match(r'^(\d+(\.\d+)*)', section_title)
        if numbered_pattern:
            numbers = numbered_pattern.group(1).split('.')
            return len(numbers)
        
        # Check for letter patterns (A, B, C or I, II, III)
        roman_pattern = re.match(r'^([IVXLCDM]+)', section_title, re.IGNORECASE)
        if roman_pattern:
            return 1
        
        letter_pattern = re.match(r'^([A-Z])', section_title)
        if letter_pattern:
            return 1
        
        return 0
    
    def _classify_document_type(self, document: Document, chunks: List[DocumentChunk], 
                               content_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Classify document type based on multiple features.
        
        Args:
            document: Document to classify
            chunks: Document chunks
            content_analysis: Content analysis results
            
        Returns:
            List of classified types with confidence scores
        """
        classifications = []
        keyword_distribution = content_analysis.get("keyword_distribution", {})
        language_patterns = content_analysis.get("language_patterns", {})
        structural_features = content_analysis.get("structural_features", {})
        
        for doc_type in DocumentType:
            # Calculate confidence score based on multiple factors
            confidence = 0.0
            factors = []
            
            # Keyword matching (40% weight)
            keyword_score = keyword_distribution.get(doc_type.value, 0) / max(sum(keyword_distribution.values()), 1)
            confidence += keyword_score * 0.4
            factors.append(f"keywords: {keyword_score:.2f}")
            
            # Language patterns (30% weight)
            language_score = language_patterns.get(f"{doc_type.value}_terms", 0) / 10.0  # Normalize
            confidence += min(language_score, 1.0) * 0.3
            factors.append(f"language: {language_score:.2f}")
            
            # Structural features (20% weight)
            structure_score = self._calculate_structure_score(doc_type, structural_features)
            confidence += structure_score * 0.2
            factors.append(f"structure: {structure_score:.2f}")
            
            # Title analysis (10% weight)
            title_score = self._analyze_title_for_type(document.title, doc_type)
            confidence += title_score * 0.1
            factors.append(f"title: {title_score:.2f}")
            
            classifications.append({
                "document_type": doc_type.value,
                "confidence": min(confidence, 1.0),
                "factors": factors
            })
        
        # Sort by confidence (descending)
        classifications.sort(key=lambda x: x["confidence"], reverse=True)
        
        return classifications
    
    def _calculate_structure_score(self, doc_type: DocumentType, structural_features: Dict[str, Any]) -> float:
        """
        Calculate structure score for document type classification.
        
        Args:
            doc_type: Document type to score
            structural_features: Structural features analysis
            
        Returns:
            Structure score (0-1)
        """
        score = 0.0
        
        # Expected structure patterns for this document type
        expected_patterns = self.structure_patterns.get(doc_type, [])
        
        # Score based on structural features
        hierarchy_depth = structural_features.get("hierarchy_depth", 0)
        procedural_elements = structural_features.get("procedural_elements", 0)
        argumentative_elements = structural_features.get("argumentative_elements", 0)
        descriptive_elements = structural_features.get("descriptive_elements", 0)
        
        if DocumentStructurePattern.SECTION_HIERARCHY in expected_patterns:
            score += min(hierarchy_depth / 3.0, 1.0) * 0.25
        
        if DocumentStructurePattern.PROCEDURAL in expected_patterns:
            score += min(procedural_elements / 5.0, 1.0) * 0.25
        
        if DocumentStructurePattern.ARGUMENTATIVE in expected_patterns:
            score += min(argumentative_elements / 5.0, 1.0) * 0.25
        
        if DocumentStructurePattern.DESCRIPTIVE in expected_patterns:
            score += min(descriptive_elements / 5.0, 1.0) * 0.25
        
        return score
    
    def _analyze_title_for_type(self, title: Optional[str], doc_type: DocumentType) -> float:
        """
        Analyze document title for type classification.
        
        Args:
            title: Document title
            doc_type: Document type to check
            
        Returns:
            Title analysis score (0-1)
        """
        if not title:
            return 0.0
        
        title_lower = title.lower()
        keywords = self.type_keywords.get(doc_type, [])
        
        # Check if any keywords appear in title
        for keyword in keywords[:5]:  # Check top 5 keywords
            if keyword in title_lower:
                return 0.5  # Moderate confidence from title match
        
        return 0.0
    
    def _identify_structure_patterns(self, document: Document, chunks: List[DocumentChunk],
                                   content_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify structure patterns in the document.
        
        Args:
            document: Document to analyze
            chunks: Document chunks
            content_analysis: Content analysis results
            
        Returns:
            List of identified structure patterns with confidence scores
        """
        patterns = []
        structural_features = content_analysis.get("structural_features", {})
        
        for pattern in DocumentStructurePattern:
            confidence = self._calculate_pattern_confidence(pattern, structural_features, chunks)
            
            if confidence > 0.3:  # Minimum confidence threshold
                patterns.append({
                    "pattern_type": pattern.value,
                    "confidence": confidence,
                    "evidence": self._gather_pattern_evidence(pattern, structural_features, chunks)
                })
        
        # Sort by confidence (descending)
        patterns.sort(key=lambda x: x["confidence"], reverse=True)
        
        return patterns
    
    def _calculate_pattern_confidence(self, pattern: DocumentStructurePattern, 
                                    structural_features: Dict[str, Any], 
                                    chunks: List[DocumentChunk]) -> float:
        """
        Calculate confidence for a specific structure pattern.
        
        Args:
            pattern: Structure pattern to evaluate
            structural_features: Structural features analysis
            chunks: Document chunks
            
        Returns:
            Pattern confidence score (0-1)
        """
        confidence = 0.0
        
        if pattern == DocumentStructurePattern.SECTION_HIERARCHY:
            hierarchy_depth = structural_features.get("hierarchy_depth", 0)
            section_variety = structural_features.get("section_variety", 0)
            confidence = min((hierarchy_depth * 0.3 + section_variety * 0.1), 1.0)
        
        elif pattern == DocumentStructurePattern.PROCEDURAL:
            procedural_elements = structural_features.get("procedural_elements", 0)
            # Check for step-by-step content
            step_patterns = sum(1 for chunk in chunks if any(word in chunk.content.lower() 
                                                           for word in ["step", "first", "next", "then", "finally"]))
            confidence = min((procedural_elements * 0.2 + step_patterns * 0.1), 1.0)
        
        elif pattern == DocumentStructurePattern.ARGUMENTATIVE:
            argumentative_elements = structural_features.get("argumentative_elements", 0)
            # Check for argument structure
            argument_words = sum(1 for chunk in chunks if any(word in chunk.content.lower() 
                                                            for word in ["therefore", "thus", "consequently", "evidence"]))
            confidence = min((argumentative_elements * 0.2 + argument_words * 0.1), 1.0)
        
        elif pattern == DocumentStructurePattern.DESCRIPTIVE:
            descriptive_elements = structural_features.get("descriptive_elements", 0)
            # Check for descriptive language
            descriptive_words = sum(1 for chunk in chunks if any(word in chunk.content.lower() 
                                                               for word in ["description", "characteristic", "feature", "property"]))
            confidence = min((descriptive_elements * 0.2 + descriptive_words * 0.1), 1.0)
        
        elif pattern == DocumentStructurePattern.COMPARATIVE:
            # Check for comparative language
            comparative_words = sum(1 for chunk in chunks if any(word in chunk.content.lower() 
                                                               for word in ["compared", "versus", "difference", "similar", "contrast"]))
            confidence = min(comparative_words * 0.2, 1.0)
        
        elif pattern == DocumentStructurePattern.NARRATIVE:
            # Check for narrative elements
            narrative_words = sum(1 for chunk in chunks if any(word in chunk.content.lower() 
                                                             for word in ["story", "narrative", "event", "occurred", "experience"]))
            confidence = min(narrative_words * 0.2, 1.0)
        
        return confidence
    
    def _gather_pattern_evidence(self, pattern: DocumentStructurePattern, 
                               structural_features: Dict[str, Any], 
                               chunks: List[DocumentChunk]) -> List[str]:
        """
        Gather evidence for identified structure patterns.
        
        Args:
            pattern: Structure pattern
            structural_features: Structural features analysis
            chunks: Document chunks
            
        Returns:
            List of evidence strings
        """
        evidence = []
        
        if pattern == DocumentStructurePattern.SECTION_HIERARCHY:
            depth = structural_features.get("hierarchy_depth", 0)
            sections = structural_features.get("section_variety", 0)
            evidence.append(f"Document has {depth}-level section hierarchy")
            evidence.append(f"Contains {sections} distinct sections")
        
        elif pattern == DocumentStructurePattern.PROCEDURAL:
            elements = structural_features.get("procedural_elements", 0)
            evidence.append(f"Contains {elements} procedural elements")
            evidence.append("Includes step-by-step instructions")
        
        elif pattern == DocumentStructurePattern.ARGUMENTATIVE:
            elements = structural_features.get("argumentative_elements", 0)
            evidence.append(f"Contains {elements} argumentative elements")
            evidence.append("Uses logical reasoning and evidence")
        
        elif pattern == DocumentStructurePattern.DESCRIPTIVE:
            elements = structural_features.get("descriptive_elements", 0)
            evidence.append(f"Contains {elements} descriptive elements")
            evidence.append("Focuses on descriptions and characteristics")
        
        elif pattern == DocumentStructurePattern.COMPARATIVE:
            evidence.append("Uses comparative language")
            evidence.append("Focuses on differences and similarities")
        
        elif pattern == DocumentStructurePattern.NARRATIVE:
            evidence.append("Contains narrative elements")
            evidence.append("Follows chronological or experiential structure")
        
        return evidence
    
    def _generate_classification_recommendations(self, type_classification: List[Dict[str, Any]], 
                                               structure_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on classification results.
        
        Args:
            type_classification: Document type classification results
            structure_patterns: Identified structure patterns
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Get primary document type
        primary_type = type_classification[0] if type_classification else None
        
        if primary_type and primary_type["confidence"] > 0.7:
            doc_type = primary_type["document_type"]
            
            # Type-specific recommendations
            if doc_type == "legal":
                recommendations.append({
                    "type": "chunking_strategy",
                    "priority": "high",
                    "description": "Use clause-based chunking for legal documents",
                    "rationale": "Preserves legal clause boundaries and relationships"
                })
            
            elif doc_type == "technical":
                recommendations.append({
                    "type": "chunking_strategy",
                    "priority": "high",
                    "description": "Use procedure-based chunking for technical documents",
                    "rationale": "Maintains procedural flow and step sequences"
                })
            
            elif doc_type == "academic":
                recommendations.append({
                    "type": "chunking_strategy",
                    "priority": "high",
                    "description": "Use section-based chunking for academic papers",
                    "rationale": "Preserves argument structure and citation context"
                })
        
        # Structure pattern recommendations
        for pattern in structure_patterns:
            if pattern["confidence"] > 0.6:
                if pattern["pattern_type"] == "section_hierarchy":
                    recommendations.append({
                        "type": "metadata_extraction",
                        "priority": "medium",
                        "description": "Extract and preserve section hierarchy metadata",
                        "rationale": "Enables hierarchical navigation and context preservation"
                    })
                
                elif pattern["pattern_type"] == "procedural":
                    recommendations.append({
                        "type": "content_processing",
                        "priority": "medium",
                        "description": "Process procedural steps as atomic units",
                        "rationale": "Maintains procedural integrity and sequence"
                    })
        
        return recommendations
    
    def _calculate_classification_confidence(self, type_classification: List[Dict[str, Any]], 
                                           structure_patterns: List[Dict[str, Any]]) -> float:
        """
        Calculate overall classification confidence.
        
        Args:
            type_classification: Document type classification results
            structure_patterns: Identified structure patterns
            
        Returns:
            Overall confidence score (0-1)
        """
        if not type_classification:
            return 0.0
        
        # Base confidence on primary type classification
        primary_confidence = type_classification[0]["confidence"]
        
        # Boost confidence if structure patterns align with document type
        if structure_patterns:
            pattern_confidence = max(p["confidence"] for p in structure_patterns)
            alignment_boost = pattern_confidence * 0.2
        else:
            alignment_boost = 0.0
        
        return min(primary_confidence + alignment_boost, 1.0)
    
    def batch_classify_documents(self, document_ids: List[int]) -> Dict[str, Any]:
        """
        Classify multiple documents in batch.
        
        Args:
            document_ids: List of document IDs to classify
            
        Returns:
            Batch classification results
        """
        results = {
            "total_documents": len(document_ids),
            "classifications": [],
            "type_distribution": {},
            "pattern_distribution": {},
            "summary": {}
        }
        
        for doc_id in document_ids:
            try:
                classification = self.classify_document(doc_id)
                results["classifications"].append(classification)
                
                # Update type distribution
                primary_type = classification["type_classification"][0]["document_type"]
                results["type_distribution"][primary_type] = results["type_distribution"].get(primary_type, 0) + 1
                
                # Update pattern distribution
                for pattern in classification["structure_patterns"]:
                    pattern_type = pattern["pattern_type"]
                    results["pattern_distribution"][pattern_type] = results["pattern_distribution"].get(pattern_type, 0) + 1
                    
            except Exception as e:
                logger.error(f"Error classifying document {doc_id}: {str(e)}")
                results["classifications"].append({
                    "document_id": doc_id,
                    "error": str(e)
                })
        
        # Generate summary
        results["summary"] = {
            "successful_classifications": len([c for c in results["classifications"] if "error" not in c]),
            "failed_classifications": len([c for c in results["classifications"] if "error" in c]),
            "most_common_type": max(results["type_distribution"].items(), key=lambda x: x[1])[0] if results["type_distribution"] else "unknown",
            "most_common_pattern": max(results["pattern_distribution"].items(), key=lambda x: x[1])[0] if results["pattern_distribution"] else "unknown"
        }
        
        return results