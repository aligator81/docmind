"""
Pydantic schemas for request/response validation.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# Base schemas
class BaseResponse(BaseModel):
    """Base response schema."""
    success: bool
    message: Optional[str] = None


# Authentication Schemas
class UserCreate(BaseModel):
    """Schema for user registration."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    """Schema for user login."""
    username: str
    password: str


class Token(BaseModel):
    """Schema for JWT token response."""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Schema for token payload data."""
    username: Optional[str] = None
    role: Optional[str] = None


# User Management Schemas
class User(BaseModel):
    """Schema for user response."""
    id: int
    username: str
    email: Optional[str] = None
    role: str
    is_active: bool
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class PasswordReset(BaseModel):
    """Schema for password reset."""
    new_password: str = Field(..., min_length=6)


class SystemStats(BaseModel):
    """Schema for system statistics."""
    total_users: int
    total_documents: int
    total_chunks: int
    total_embeddings: int
    active_sessions: int


class APIConfigCreate(BaseModel):
    """Schema for API configuration creation."""
    provider: str = Field(..., pattern=r'^(openai|mistral)$')
    api_key: str
    is_active: bool = True
    model: Optional[str] = Field(None, description="Chat model to use")
    embedding_model: Optional[str] = Field(None, description="Embedding model to use")
    max_tokens: Optional[int] = Field(1000, ge=100, le=4000, description="Maximum tokens for chat responses")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Temperature for chat responses")


class APIConfig(BaseModel):
    """Schema for API configuration."""
    id: int
    provider: str
    is_active: bool
    model: Optional[str] = None
    embedding_model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class CompanyBrandingCreate(BaseModel):
    """Schema for company branding creation."""
    company_name: str
    logo_url: Optional[str] = None


class CompanyBranding(BaseModel):
    """Schema for company branding."""
    id: int
    company_name: str
    logo_url: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Search Feedback Schemas
class FeedbackRequest(BaseModel):
    """Request schema for submitting search feedback."""
    query: str = Field(..., description="Search query")
    document_ids: List[int] = Field(..., description="List of document IDs in results")
    feedback_type: str = Field(..., description="Type of feedback (positive, negative, neutral, click, skip)")
    user_id: Optional[int] = Field(None, description="Optional user ID")


class FeedbackResponse(BaseResponse):
    """Response schema for feedback submission."""
    feedback_id: Optional[int] = None
    query_id: Optional[int] = None


class AnalysisResponse(BaseResponse):
    """Response schema for feedback analysis."""
    analysis_period: Optional[str] = None
    feedback_statistics: Dict[str, Any] = {}
    query_analysis: Dict[str, Any] = {}
    relevance_issues: List[Dict[str, Any]] = []
    tuning_recommendations: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}


class TuningRequest(BaseModel):
    """Request schema for relevance tuning."""
    strategy: str = Field(..., description="Tuning strategy to use")
    use_precomputed_analysis: bool = Field(True, description="Use pre-computed feedback analysis")


class TuningResponse(BaseResponse):
    """Response schema for tuning operation."""
    strategy: Optional[str] = None
    parameters_used: Dict[str, Any] = {}
    tuning_results: Dict[str, Any] = {}
    effectiveness_estimate: Dict[str, Any] = {}
    recommendations: List[Dict[str, Any]] = []


class ABTestRequest(BaseModel):
    """Request schema for A/B test creation."""
    strategy_a: str = Field(..., description="First tuning strategy")
    strategy_b: str = Field(..., description="Second tuning strategy")
    test_duration_days: int = Field(7, description="Test duration in days", ge=1, le=30)


class ABTestResponse(BaseResponse):
    """Response schema for A/B test creation."""
    test_id: Optional[str] = None
    strategy_a: Optional[str] = None
    strategy_b: Optional[str] = None
    test_duration_days: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    status: Optional[str] = None
    metrics_to_track: List[str] = []


# Document Processing Schemas
class DocumentCreate(BaseModel):
    """Schema for document creation."""
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    mime_type: str
    user_id: int
    status: str = "not processed"


class Document(BaseModel):
    """Schema for document response."""
    id: int
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    mime_type: str
    user_id: int
    status: str
    content: Optional[str] = None
    created_at: datetime
    processed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DocumentUploadRequest(BaseModel):
    """Request schema for document upload."""
    filename: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentUploadResponse(BaseResponse):
    """Response schema for document upload."""
    document_id: Optional[int] = None
    chunk_count: Optional[int] = None
    processing_time: Optional[float] = None
    document: Optional[Document] = None
    message: Optional[str] = None


class DocumentChunk(BaseModel):
    """Schema for document chunk."""
    id: int
    document_id: int
    chunk_text: str
    chunk_index: int
    page_numbers: List[int]
    section_title: Optional[str] = None
    metadata: Dict[str, Any] = {}


class SearchRequest(BaseModel):
    """Request schema for search."""
    query: str
    limit: int = Field(10, ge=1, le=100)
    semantic_weight: float = Field(0.7, ge=0.0, le=1.0)
    lexical_weight: float = Field(0.3, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """Schema for search result."""
    chunk_id: int
    document_id: int
    document_name: str
    chunk_text: str
    page_numbers: List[int]
    section_title: Optional[str] = None
    similarity_score: float
    lexical_score: Optional[float] = None
    combined_score: Optional[float] = None
    citation: str


class SearchResponse(BaseResponse):
    """Response schema for search."""
    results: List[SearchResult] = []
    total_results: int = 0
    search_time: Optional[float] = None


# Hybrid Search Schemas
class HybridSearchRequest(BaseModel):
    """Request schema for hybrid search."""
    query: str
    limit: int = Field(10, ge=1, le=100)
    semantic_weight: float = Field(0.7, ge=0.0, le=1.0)
    lexical_weight: float = Field(0.3, ge=0.0, le=1.0)
    user_id: Optional[int] = None


class HybridSearchResult(BaseModel):
    """Schema for hybrid search result."""
    id: int
    chunk_id: int
    chunk_text: str
    section_title: Optional[str] = None
    chunk_index: int
    page_numbers: List[int]
    document_id: int
    document_filename: str
    document_original_filename: str
    document_status: str
    document_created_at: datetime
    similarity_score: Optional[float] = None
    text_rank: Optional[float] = None
    combined_score: float
    rank: int
    search_types: List[str] = []
    search_type: str = "hybrid"


class HybridSearchResponse(BaseResponse):
    """Response schema for hybrid search."""
    results: List[HybridSearchResult] = []
    total_results: int = 0
    search_time: Optional[float] = None
    vector_results_count: int = 0
    text_results_count: int = 0
    combined_results_count: int = 0


class SearchAnalyticsResponse(BaseResponse):
    """Response schema for search analytics."""
    period_days: int
    total_searches: int
    average_results_per_search: float
    top_queries: List[Dict[str, Any]] = []
    search_success_rate: float
    performance_metrics: Dict[str, Any] = {}


# Chat Schemas
class ChatMessage(BaseModel):
    """Schema for chat message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """Request schema for chat."""
    message: str
    conversation_id: Optional[str] = None
    use_context: bool = Field(True, description="Use document context for response")
    max_tokens: int = Field(1000, ge=100, le=4000)


class DocumentChatRequest(BaseModel):
    """Request schema for document-based chat."""
    message: str
    document_ids: Optional[List[int]] = None


class ChatResponse(BaseResponse):
    """Response schema for chat."""
    response: str
    conversation_id: Optional[str] = None
    citations: List[Dict[str, Any]] = []
    context_used: bool = False
    context_docs: Optional[List[int]] = None
    model_used: Optional[str] = None
    references: Optional[List[Dict[str, Any]]] = None


# Document Analysis Schemas
class DocumentAnalysisRequest(BaseModel):
    """Request schema for document analysis."""
    document_ids: List[int]
    analysis_type: str = Field("structure", description="Type of analysis (structure, relationships, quality)")


class DocumentAnalysisResponse(BaseResponse):
    """Response schema for document analysis."""
    analysis_type: str
    results: Dict[str, Any] = {}
    recommendations: List[Dict[str, Any]] = []


class MultiDocumentAnalysisRequest(BaseModel):
    """Request schema for multi-document analysis."""
    document_ids: List[int]
    analysis_types: List[str] = Field(["comparative", "synthesis"], description="Types of analysis to perform")


class MultiDocumentAnalysisResponse(BaseResponse):
    """Response schema for multi-document analysis."""
    analysis_types: List[str]
    results: Dict[str, Any] = {}
    relationships: List[Dict[str, Any]] = []
    insights: List[str] = []


# Chunking Schemas
class ChunkingStrategyRequest(BaseModel):
    """Request schema for chunking strategy selection."""
    document_id: int
    strategy: str = Field("adaptive", description="Chunking strategy to use")
    custom_parameters: Optional[Dict[str, Any]] = None


class ChunkingStrategyResponse(BaseResponse):
    """Response schema for chunking strategy."""
    strategy: str
    parameters_used: Dict[str, Any] = {}
    chunk_count: int
    quality_metrics: Dict[str, Any] = {}
    recommendations: List[Dict[str, Any]] = []


# Relationship Analysis Schemas
class RelationshipAnalysisRequest(BaseModel):
    """Request schema for relationship analysis."""
    document_ids: List[int]
    relationship_types: List[str] = Field(["semantic", "structural"], description="Types of relationships to analyze")


class RelationshipAnalysisResponse(BaseResponse):
    """Response schema for relationship analysis."""
    relationship_types: List[str]
    relationships: List[Dict[str, Any]] = []
    graph_data: Optional[Dict[str, Any]] = None
    insights: List[str] = []


# System Monitoring Schemas
class SystemMetrics(BaseModel):
    """Schema for system metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    active_connections: int
    queue_size: int
    processing_rate: float


class MetricsResponse(BaseResponse):
    """Response schema for system metrics."""
    metrics: List[SystemMetrics] = []
    time_range: str


# Error Schemas
class ErrorResponse(BaseResponse):
    """Schema for error responses."""
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Processing Schemas
class ProcessingStatus(BaseModel):
    """Schema for processing status."""
    document_id: int
    status: str
    content_length: int
    chunks_count: int
    embeddings_count: int
    created_at: datetime
    processed_at: Optional[datetime] = None


class ProcessingResult(BaseResponse):
    """Schema for processing result."""
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = {}


# Health Check Schema
class HealthResponse(BaseResponse):
    """Schema for health check response."""
    status: str
    version: str
    timestamp: datetime
    services: Dict[str, str]


# Question-Answer Export Schemas
class QuestionAnswerRequest(BaseModel):
    """Request schema for question-answer processing."""
    questions: List[str] = Field(..., description="List of questions to process")
    document_ids: Optional[List[int]] = Field(None, description="Optional document IDs for context")
    export_name: Optional[str] = Field(None, description="Custom name for the export file")


class QuestionAnswerResponse(BaseResponse):
    """Response schema for question-answer processing."""
    export_id: Optional[int] = None
    questions_processed: int = 0
    answers_generated: int = 0
    processing_time: Optional[float] = None
    export_file_url: Optional[str] = None
    session_id: Optional[int] = None


class QuestionAnswerExportSchema(BaseModel):
    """Schema for question-answer export response."""
    id: int
    filename: str
    file_size: int
    questions_count: int
    created_at: datetime
    status: str
    
    class Config:
        from_attributes = True


class ProgressResponse(BaseModel):
    """Schema for progress tracking response."""
    session_id: int
    user_id: int
    total_questions: int
    processed_questions: int
    current_question: str
    current_question_index: int
    status: str
    progress_percentage: float