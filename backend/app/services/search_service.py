from sqlalchemy import or_, and_, func, desc
from sqlalchemy.orm import Session
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

from ..models import Document, DocumentChunk, Embedding, User
from ..schemas import HybridSearchRequest
from .hybrid_search_service import create_hybrid_search_service

logger = logging.getLogger(__name__)

class AdvancedSearch:
    def __init__(self, db: Session):
        self.db = db

    def search_documents(self, query: str = None, filters: Dict = None, user_id: int = None, user_role: str = "user") -> Dict:
        """Advanced document search with filters and pagination"""
        try:
            # Check if semantic search is requested
            use_semantic_search = filters and filters.get('semantic_search', False) if filters else False
            
            if use_semantic_search and query and query.strip():
                # Use hybrid semantic search for content
                semantic_results = self.hybrid_content_search(
                    query=query,
                    user_id=user_id,
                    user_role=user_role,
                    limit=filters.get('per_page', 20) if filters else 20,
                    vector_weight=filters.get('vector_weight', 0.7) if filters else 0.7,
                    text_weight=filters.get('text_weight', 0.3) if filters else 0.3,
                    similarity_threshold=filters.get('similarity_threshold', 0.5) if filters else 0.5
                )
                
                if semantic_results["success"]:
                    # Convert hybrid search results to document format
                    document_ids = set()
                    documents = []
                    
                    for result in semantic_results["results"]:
                        doc_id = result["document_id"]
                        if doc_id not in document_ids:
                            document = self.db.query(Document).filter(Document.id == doc_id).first()
                            if document:
                                documents.append(document)
                                document_ids.add(doc_id)
                    
                    return {
                        "success": True,
                        "documents": documents,
                        "total_count": semantic_results["total_count"],
                        "page": filters.get('page', 1) if filters else 1,
                        "per_page": filters.get('per_page', 20) if filters else 20,
                        "total_pages": (semantic_results["total_count"] + (filters.get('per_page', 20) if filters else 20) - 1) // (filters.get('per_page', 20) if filters else 20),
                        "has_next": False,  # Simplified for semantic search
                        "has_prev": False,  # Simplified for semantic search
                        "search_metadata": semantic_results.get("search_metadata", {}),
                        "response_time_ms": semantic_results.get("response_time_ms", 0)
                    }
                else:
                    # Fall back to regular search if semantic search fails
                    logger.warning(f"Semantic search failed, falling back to regular search: {semantic_results.get('error')}")

            # Start with base query
            base_query = self.db.query(Document)

            # Apply user-based filtering
            if user_role != "admin":
                base_query = base_query.filter(Document.user_id == user_id)

            # Apply text search if query provided
            if query and query.strip():
                search_term = f"%{query.strip()}%"
                base_query = base_query.filter(
                    or_(
                        Document.original_filename.ilike(search_term),
                        Document.filename.ilike(search_term),
                        Document.content.ilike(search_term),
                        Document.mime_type.ilike(search_term)
                    )
                )

            # Apply filters
            if filters:
                base_query = self._apply_filters(base_query, filters)

            # Get total count before pagination
            total_count = base_query.count()

            # Apply sorting
            sort_by = filters.get('sort_by', 'created_at') if filters else 'created_at'
            sort_order = filters.get('sort_order', 'desc') if filters else 'desc'

            if sort_by == 'created_at':
                base_query = base_query.order_by(desc(Document.created_at) if sort_order == 'desc' else Document.created_at)
            elif sort_by == 'file_size':
                base_query = base_query.order_by(desc(Document.file_size) if sort_order == 'desc' else Document.file_size)
            elif sort_by == 'filename':
                base_query = base_query.order_by(desc(Document.filename) if sort_order == 'desc' else Document.filename)

            # Apply pagination
            page = int(filters.get('page', 1)) if filters else 1
            per_page = int(filters.get('per_page', 20)) if filters else 20
            offset = (page - 1) * per_page

            documents = base_query.offset(offset).limit(per_page).all()

            return {
                "success": True,
                "documents": documents,
                "total_count": total_count,
                "page": page,
                "per_page": per_page,
                "total_pages": (total_count + per_page - 1) // per_page,
                "has_next": offset + per_page < total_count,
                "has_prev": page > 1
            }

        except Exception as e:
            logger.error(f"Error in document search: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "documents": [],
                "total_count": 0
            }

    def _apply_filters(self, query, filters: Dict):
        """Apply various filters to the query"""
        try:
            # Status filter
            if 'status' in filters and filters['status']:
                query = query.filter(Document.status == filters['status'])

            # File type filter
            if 'file_type' in filters and filters['file_type']:
                query = query.filter(Document.mime_type.ilike(f"%{filters['file_type']}%"))

            # Date range filters
            if 'date_from' in filters and filters['date_from']:
                try:
                    date_from = datetime.fromisoformat(filters['date_from'].replace('Z', '+00:00'))
                    query = query.filter(Document.created_at >= date_from)
                except ValueError:
                    logger.warning(f"Invalid date_from format: {filters['date_from']}")

            if 'date_to' in filters and filters['date_to']:
                try:
                    date_to = datetime.fromisoformat(filters['date_to'].replace('Z', '+00:00'))
                    query = query.filter(Document.created_at <= date_to)
                except ValueError:
                    logger.warning(f"Invalid date_to format: {filters['date_to']}")

            # File size filters
            if 'min_size' in filters and filters['min_size']:
                try:
                    min_size = int(filters['min_size'])
                    query = query.filter(Document.file_size >= min_size)
                except ValueError:
                    logger.warning(f"Invalid min_size format: {filters['min_size']}")

            if 'max_size' in filters and filters['max_size']:
                try:
                    max_size = int(filters['max_size'])
                    query = query.filter(Document.file_size <= max_size)
                except ValueError:
                    logger.warning(f"Invalid max_size format: {filters['max_size']}")

            # Content search in chunks (for processed documents)
            if 'content_search' in filters and filters['content_search']:
                content_term = f"%{filters['content_search']}%"
                # Join with chunks table for content search
                query = query.join(DocumentChunk).filter(
                    DocumentChunk.content.ilike(content_term)
                ).distinct()

            return query

        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            return query

    def search_similar_documents(self, document_id: int, limit: int = 10) -> Dict:
        """Find similar documents using embeddings"""
        try:
            # Get the source document
            source_document = self.db.query(Document).filter(Document.id == document_id).first()
            if not source_document:
                return {
                    "success": False,
                    "error": "Source document not found",
                    "similar_documents": []
                }

            # Get embeddings for the source document
            source_embeddings = self.db.query(Embedding).join(
                DocumentChunk, Embedding.chunk_id == DocumentChunk.id
            ).filter(
                DocumentChunk.document_id == document_id
            ).all()

            if not source_embeddings:
                return {
                    "success": False,
                    "error": "Source document has no embeddings",
                    "similar_documents": []
                }

            # Use hybrid search to find similar documents based on content
            # Extract key terms from the document for similarity search
            search_query = source_document.original_filename or source_document.filename
            if source_document.content:
                # Use first 50 words as search query
                words = source_document.content.split()[:50]
                search_query = " ".join(words)

            if not search_query:
                # Fallback to basic similarity
                return self._fallback_similarity_search(document_id, limit)

            # Use hybrid search with high vector weight for semantic similarity
            hybrid_service = create_hybrid_search_service(self.db)
            search_request = HybridSearchRequest(
                query=search_query,
                limit=limit * 2,  # Get more for filtering
                vector_weight=0.9,  # High vector weight for semantic similarity
                text_weight=0.1,
                similarity_threshold=0.6,  # Higher threshold for similarity
                user_id=source_document.user_id
            )

            hybrid_results = hybrid_service.hybrid_search(search_request)

            if not hybrid_results["success"]:
                return self._fallback_similarity_search(document_id, limit)

            # Filter out the source document and process results
            similar_docs = []
            seen_doc_ids = set()

            for result in hybrid_results["results"]:
                doc_id = result["document_id"]
                if doc_id != document_id and doc_id not in seen_doc_ids:
                    document = self.db.query(Document).filter(Document.id == doc_id).first()
                    if document:
                        similar_docs.append({
                            "document": document,
                            "similarity_score": result.get("combined_score", 0.5),
                            "search_types": result.get("search_types", []),
                            "chunk_text": result.get("chunk_text", "")[:200]  # Preview
                        })
                        seen_doc_ids.add(doc_id)
                
                if len(similar_docs) >= limit:
                    break

            return {
                "success": True,
                "similar_documents": similar_docs,
                "search_metadata": hybrid_results.get("search_metadata", {})
            }

        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return self._fallback_similarity_search(document_id, limit)

    def _fallback_similarity_search(self, document_id: int, limit: int = 10) -> Dict:
        """Fallback similarity search using basic criteria"""
        try:
            similar_docs = []
            recent_docs = self.db.query(Document).filter(
                Document.id != document_id,
                Document.status == "processed"
            ).order_by(desc(Document.processed_at)).limit(limit).all()

            for doc in recent_docs:
                similar_docs.append({
                    "document": doc,
                    "similarity_score": 0.5,  # Lower confidence for fallback
                    "search_types": ["fallback"],
                    "chunk_text": ""
                })

            return {
                "success": True,
                "similar_documents": similar_docs,
                "search_metadata": {"fallback_used": True}
            }

        except Exception as e:
            logger.error(f"Error in fallback similarity search: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "similar_documents": []
            }

    def hybrid_content_search(self, query: str, user_id: int = None, user_role: str = "user",
                            limit: int = 10, vector_weight: float = 0.7, text_weight: float = 0.3,
                            similarity_threshold: float = 0.5) -> Dict:
        """
        Perform hybrid semantic search on document content using vector embeddings and full-text search
        """
        try:
            # Create hybrid search service
            hybrid_service = create_hybrid_search_service(self.db)
            
            # Create search request
            search_request = HybridSearchRequest(
                query=query,
                limit=limit,
                vector_weight=vector_weight,
                text_weight=text_weight,
                similarity_threshold=similarity_threshold,
                user_id=user_id
            )
            
            # Perform hybrid search
            hybrid_results = hybrid_service.hybrid_search(search_request)
            
            if not hybrid_results["success"]:
                return {
                    "success": False,
                    "error": hybrid_results.get("error", "Hybrid search failed"),
                    "results": [],
                    "total_count": 0
                }
            
            # Filter results by user access if needed
            if user_role != "admin" and user_id:
                filtered_results = []
                for result in hybrid_results["results"]:
                    # Check if user has access to this document
                    document = self.db.query(Document).filter(
                        Document.id == result["document_id"]
                    ).first()
                    
                    if document and (document.user_id == user_id or user_role == "admin"):
                        filtered_results.append(result)
                
                hybrid_results["results"] = filtered_results
                hybrid_results["total_count"] = len(filtered_results)
            
            return {
                "success": True,
                "results": hybrid_results["results"],
                "total_count": hybrid_results["total_count"],
                "response_time_ms": hybrid_results["response_time_ms"],
                "search_metadata": hybrid_results["search_metadata"]
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid content search: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "total_count": 0
            }

    def get_search_suggestions(self, query: str, user_id: int, limit: int = 10) -> List[str]:
        """Get search suggestions based on existing document names and content"""
        try:
            suggestions = set()

            if len(query) < 2:
                return list(suggestions)

            search_term = f"%{query}%"

            # Get suggestions from filenames
            filename_results = self.db.query(Document.original_filename).filter(
                Document.original_filename.ilike(search_term),
                Document.user_id == user_id
            ).distinct().limit(limit // 2).all()

            for result in filename_results:
                suggestions.add(result.original_filename)

            # Get suggestions from content (for processed documents)
            content_results = self.db.query(Document.content).filter(
                Document.content.ilike(search_term),
                Document.user_id == user_id,
                Document.content.isnot(None)
            ).distinct().limit(limit // 2).all()

            for result in content_results:
                # Extract meaningful phrases from content
                words = result.content.split()[:5]  # First 5 words
                if words:
                    suggestions.add(" ".join(words) + "...")

            return list(suggestions)[:limit]

        except Exception as e:
            logger.error(f"Error getting search suggestions: {str(e)}")
            return []

    def get_document_statistics(self, user_id: int = None, user_role: str = "user") -> Dict:
        """Get statistics about documents in the system"""
        try:
            base_query = self.db.query(Document)

            if user_role != "admin":
                base_query = base_query.filter(Document.user_id == user_id)

            # Overall statistics
            total_documents = base_query.count()

            # Status breakdown
            status_counts = dict(
                base_query.with_entities(Document.status, func.count(Document.id))
                .group_by(Document.status).all()
            )

            # File type breakdown
            file_types = dict(
                base_query.with_entities(Document.mime_type, func.count(Document.id))
                .group_by(Document.mime_type).limit(10).all()
            )

            # Size statistics
            size_stats = base_query.with_entities(
                func.avg(Document.file_size),
                func.min(Document.file_size),
                func.max(Document.file_size)
            ).first()

            # Recent activity (last 30 days)
            thirty_days_ago = datetime.utcnow().replace(day=1)
            recent_docs = base_query.filter(Document.created_at >= thirty_days_ago).count()

            return {
                "success": True,
                "statistics": {
                    "total_documents": total_documents,
                    "status_breakdown": status_counts,
                    "file_types": file_types,
                    "size_statistics": {
                        "average_size": size_stats[0] or 0,
                        "min_size": size_stats[1] or 0,
                        "max_size": size_stats[2] or 0
                    },
                    "recent_activity": recent_docs,
                    "generated_at": datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Error getting document statistics: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

# Search router functions
def create_search_service(db: Session) -> AdvancedSearch:
    """Factory function to create search service"""
    return AdvancedSearch(db)

def search_documents_endpoint(
    query: str = None,
    status: str = None,
    file_type: str = None,
    date_from: str = None,
    date_to: str = None,
    min_size: str = None,
    max_size: str = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    page: int = 1,
    per_page: int = 20,
    current_user: User = None,
    db: Session = None
) -> Dict:
    """Main search endpoint function"""
    search_service = AdvancedSearch(db)

    # Build filters
    filters = {
        'status': status,
        'file_type': file_type,
        'date_from': date_from,
        'date_to': date_to,
        'min_size': min_size,
        'max_size': max_size,
        'sort_by': sort_by,
        'sort_order': sort_order,
        'page': page,
        'per_page': per_page
    }

    # Remove None values
    filters = {k: v for k, v in filters.items() if v is not None}

    return search_service.search_documents(
        query=query,
        filters=filters,
        user_id=current_user.id if current_user else None,
        user_role=current_user.role if current_user else "user"
    )