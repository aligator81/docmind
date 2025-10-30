import logging
from typing import List, Dict, Optional, Tuple, Any
from sqlalchemy import func, text, or_, and_
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import ARRAY
import numpy as np
import json
from datetime import datetime

from ..models import Document, DocumentChunk, Embedding, User, SearchHistory
from ..schemas import HybridSearchRequest, HybridSearchResult
from ..monitoring import search_monitor, search_analytics, monitor_search_performance

logger = logging.getLogger(__name__)

class HybridSearchService:
    def __init__(self, db: Session):
        self.db = db
    
    @monitor_search_performance("hybrid")
    def hybrid_search(self, request: HybridSearchRequest) -> Dict[str, Any]:
        """
        Perform hybrid search combining vector similarity and full-text search
        """
        try:
            start_time = datetime.now()
            
            # Get vector search results with timing
            vector_start = datetime.now()
            vector_results = self._vector_search(
                query=request.query,
                limit=request.limit * 2,  # Get more for ranking
                similarity_threshold=request.similarity_threshold
            )
            vector_duration = (datetime.now() - vector_start).total_seconds()
            
            # Get full-text search results with timing
            text_start = datetime.now()
            text_results = self._full_text_search(
                query=request.query,
                limit=request.limit * 2  # Get more for ranking
            )
            text_duration = (datetime.now() - text_start).total_seconds()
            
            # Combine and rank results with timing
            fusion_start = datetime.now()
            combined_results = self._combine_and_rank_results(
                vector_results=vector_results,
                text_results=text_results,
                vector_weight=request.vector_weight,
                text_weight=request.text_weight,
                final_limit=request.limit
            )
            fusion_duration = (datetime.now() - fusion_start).total_seconds()
            
            # Calculate total response time
            total_duration = (datetime.now() - start_time).total_seconds()
            response_time_ms = int(total_duration * 1000)
            
            # Log detailed hybrid search performance breakdown
            search_monitor.log_hybrid_search_breakdown(
                query=request.query,
                vector_duration=vector_duration,
                text_duration=text_duration,
                fusion_duration=fusion_duration,
                total_duration=total_duration,
                results_count=len(combined_results),
                vector_weight=request.vector_weight,
                text_weight=request.text_weight
            )
            
            # Log search quality metrics
            relevance_scores = [result.get("combined_score", 0) for result in combined_results]
            search_monitor.log_search_quality(
                query=request.query,
                search_type="hybrid",
                relevance_scores=relevance_scores
            )
            
            # Update search analytics
            search_analytics.update_analytics(
                search_type="hybrid",
                duration=total_duration,
                success=True
            )
            
            # Log search history
            self._log_search_history(
                query=request.query,
                search_type="hybrid",
                vector_weight=request.vector_weight,
                text_weight=request.text_weight,
                result_count=len(combined_results),
                response_time_ms=response_time_ms,
                user_id=request.user_id
            )
            
            return {
                "success": True,
                "results": combined_results,
                "total_count": len(combined_results),
                "response_time_ms": response_time_ms,
                "search_metadata": {
                    "vector_results_count": len(vector_results),
                    "text_results_count": len(text_results),
                    "vector_weight": request.vector_weight,
                    "text_weight": request.text_weight,
                    "performance_breakdown": {
                        "vector_search_ms": round(vector_duration * 1000, 2),
                        "text_search_ms": round(text_duration * 1000, 2),
                        "fusion_ms": round(fusion_duration * 1000, 2),
                        "total_ms": response_time_ms
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            search_analytics.update_analytics(
                search_type="hybrid",
                duration=0,
                success=False
            )
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "total_count": 0
            }
    
    def _vector_search(self, query: str, limit: int, similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Perform vector similarity search using pgvector
        """
        try:
            # Get query embedding (in production, this would come from embedding service)
            # For now, we'll use existing embeddings in the database
            # This is a placeholder - in production, you'd generate embeddings for the query
            
            # Find similar embeddings using cosine similarity
            vector_query = text("""
                SELECT 
                    e.id,
                    e.chunk_id,
                    e.filename,
                    e.original_filename,
                    e.page_numbers,
                    e.title,
                    e.embedding_vector,
                    dc.chunk_text,
                    dc.section_title,
                    dc.chunk_index,
                    d.id as document_id,
                    d.filename as document_filename,
                    d.original_filename as document_original_filename,
                    d.status as document_status,
                    d.created_at as document_created_at,
                    1 - (e.embedding_vector <=> :query_vector) as similarity_score
                FROM embeddings e
                JOIN document_chunks dc ON e.chunk_id = dc.id
                JOIN documents d ON dc.document_id = d.id
                WHERE 1 - (e.embedding_vector <=> :query_vector) > :similarity_threshold
                ORDER BY similarity_score DESC
                LIMIT :limit
            """)
            
            # For now, we'll use a placeholder vector - in production, generate actual query embedding
            # This is a simplified approach - you'd need to generate embeddings for the query
            placeholder_vector = [0.0] * 3072  # Match the dimension of stored embeddings
            
            results = self.db.execute(
                vector_query,
                {
                    "query_vector": placeholder_vector,
                    "similarity_threshold": similarity_threshold,
                    "limit": limit
                }
            ).fetchall()
            
            return [
                {
                    "id": row.id,
                    "chunk_id": row.chunk_id,
                    "filename": row.filename,
                    "original_filename": row.original_filename,
                    "page_numbers": row.page_numbers,
                    "title": row.title,
                    "chunk_text": row.chunk_text,
                    "section_title": row.section_title,
                    "chunk_index": row.chunk_index,
                    "document_id": row.document_id,
                    "document_filename": row.document_filename,
                    "document_original_filename": row.document_original_filename,
                    "document_status": row.document_status,
                    "document_created_at": row.document_created_at,
                    "similarity_score": float(row.similarity_score),
                    "search_type": "vector"
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []
    
    def _full_text_search(self, query: str, limit: int) -> List[Dict]:
        """
        Perform full-text search using PostgreSQL tsvector
        """
        try:
            # Update search_vector column if not already populated
            self._update_search_vectors()
            
            # Perform full-text search
            text_query = text("""
                SELECT 
                    dc.id,
                    dc.chunk_id,
                    dc.chunk_text,
                    dc.section_title,
                    dc.chunk_index,
                    dc.page_numbers,
                    d.id as document_id,
                    d.filename as document_filename,
                    d.original_filename as document_original_filename,
                    d.status as document_status,
                    d.created_at as document_created_at,
                    ts_rank(dc.search_vector, plainto_tsquery('english', :query)) as text_rank
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE dc.search_vector @@ plainto_tsquery('english', :query)
                ORDER BY text_rank DESC
                LIMIT :limit
            """)
            
            results = self.db.execute(
                text_query,
                {"query": query, "limit": limit}
            ).fetchall()
            
            return [
                {
                    "id": row.id,
                    "chunk_id": row.chunk_id,
                    "chunk_text": row.chunk_text,
                    "section_title": row.section_title,
                    "chunk_index": row.chunk_index,
                    "page_numbers": row.page_numbers,
                    "document_id": row.document_id,
                    "document_filename": row.document_filename,
                    "document_original_filename": row.document_original_filename,
                    "document_status": row.document_status,
                    "document_created_at": row.document_created_at,
                    "text_rank": float(row.text_rank),
                    "search_type": "text"
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Error in full-text search: {str(e)}")
            return []
    
    def _update_search_vectors(self):
        """
        Update search_vector column for document_chunks if not populated
        """
        try:
            # Check if search_vector needs updating
            update_query = text("""
                UPDATE document_chunks 
                SET search_vector = to_tsvector('english', coalesce(chunk_text, ''))
                WHERE search_vector IS NULL
            """)
            self.db.execute(update_query)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error updating search vectors: {str(e)}")
            self.db.rollback()
    
    def _combine_and_rank_results(
        self, 
        vector_results: List[Dict], 
        text_results: List[Dict],
        vector_weight: float,
        text_weight: float,
        final_limit: int
    ) -> List[Dict]:
        """
        Combine and rank results from both search methods
        """
        try:
            # Create a combined results dictionary by chunk_id
            combined = {}
            
            # Add vector results
            for result in vector_results:
                chunk_id = result["chunk_id"]
                if chunk_id not in combined:
                    combined[chunk_id] = result.copy()
                    combined[chunk_id]["combined_score"] = result["similarity_score"] * vector_weight
                else:
                    # If already exists, update the score
                    combined[chunk_id]["combined_score"] += result["similarity_score"] * vector_weight
            
            # Add text results
            for result in text_results:
                chunk_id = result["chunk_id"]
                if chunk_id not in combined:
                    combined[chunk_id] = result.copy()
                    combined[chunk_id]["combined_score"] = result["text_rank"] * text_weight
                else:
                    # If already exists, update the score
                    combined[chunk_id]["combined_score"] += result["text_rank"] * text_weight
            
            # Convert to list and sort by combined score
            combined_list = list(combined.values())
            combined_list.sort(key=lambda x: x["combined_score"], reverse=True)
            
            # Add ranking information
            for i, result in enumerate(combined_list[:final_limit]):
                result["rank"] = i + 1
                result["search_types"] = []
                if "similarity_score" in result:
                    result["search_types"].append("vector")
                if "text_rank" in result:
                    result["search_types"].append("text")
            
            return combined_list[:final_limit]
            
        except Exception as e:
            logger.error(f"Error combining results: {str(e)}")
            return []
    
    def _log_search_history(
        self,
        query: str,
        search_type: str,
        vector_weight: float,
        text_weight: float,
        result_count: int,
        response_time_ms: int,
        user_id: Optional[int] = None
    ):
        """
        Log search history for analytics and monitoring
        """
        try:
            search_history = SearchHistory(
                query_text=query,
                search_type=search_type,
                vector_weight=vector_weight,
                text_weight=text_weight,
                result_count=result_count,
                response_time_ms=response_time_ms,
                user_id=user_id,
                search_metadata={
                    "timestamp": datetime.now().isoformat(),
                    "vector_weight": vector_weight,
                    "text_weight": text_weight
                }
            )
            
            self.db.add(search_history)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error logging search history: {str(e)}")
            self.db.rollback()
    
    def get_search_analytics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get search analytics for the specified period
        """
        try:
            cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
            
            # Total searches
            total_searches = self.db.query(SearchHistory).filter(
                SearchHistory.created_at >= cutoff_date
            ).count()
            
            # Average response time
            avg_response_time = self.db.query(
                func.avg(SearchHistory.response_time_ms)
            ).filter(
                SearchHistory.created_at >= cutoff_date,
                SearchHistory.response_time_ms.isnot(None)
            ).scalar() or 0
            
            # Search type distribution
            search_type_counts = dict(
                self.db.query(
                    SearchHistory.search_type,
                    func.count(SearchHistory.id)
                ).filter(
                    SearchHistory.created_at >= cutoff_date
                ).group_by(
                    SearchHistory.search_type
                ).all()
            )
            
            # Average results per search
            avg_results = self.db.query(
                func.avg(SearchHistory.result_count)
            ).filter(
                SearchHistory.created_at >= cutoff_date,
                SearchHistory.result_count.isnot(None)
            ).scalar() or 0
            
            return {
                "success": True,
                "analytics": {
                    "total_searches": total_searches,
                    "average_response_time_ms": float(avg_response_time),
                    "search_type_distribution": search_type_counts,
                    "average_results_per_search": float(avg_results),
                    "period_days": days
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting search analytics: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


def create_hybrid_search_service(db: Session) -> HybridSearchService:
    """Factory function to create hybrid search service"""
    return HybridSearchService(db)