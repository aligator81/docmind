"""
Database models for the application.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY
import json
from typing import List
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True)
    role = Column(String(20), default="user", nullable=False)  # user, admin, super_admin
    created_at = Column(DateTime, default=func.now(), nullable=False)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True, nullable=False)

    __table_args__ = (
        Index('idx_users_role', 'role'),
    )

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(20), default="not processed", nullable=False)  # not processed, extracted, chunked, processed
    created_at = Column(DateTime, default=func.now(), nullable=False)
    processed_at = Column(DateTime)

    # Add missing columns that exist in the database
    file_type = Column(String(100))  # Keep for backward compatibility
    content = Column(Text)  # Keep for backward compatibility
    upload_date = Column(DateTime, default=func.now())  # Keep for backward compatibility
    processed = Column(Boolean, default=False)  # Keep for backward compatibility
    processing_date = Column(DateTime)  # Keep for backward compatibility
    metadata_ = Column(Text)  # JSON metadata as text, keep for backward compatibility

    __table_args__ = (
        Index('idx_documents_user_id', 'user_id'),
        Index('idx_documents_status', 'status'),
    )

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_text = Column(Text, nullable=False)  # Actual column in database
    chunk_index = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Additional columns that exist in the database
    page_numbers = Column(Text)  # JSON array as text for SQLite compatibility
    section_title = Column(String(255))
    chunk_type = Column(String(50))
    token_count = Column(Integer)

    @property
    def content(self) -> str:
        """Get content from chunk_text for compatibility"""
        return self.chunk_text

    @content.setter
    def content(self, value: str) -> None:
        """Set content to chunk_text for compatibility"""
        self.chunk_text = value

    @property
    def metadata_(self) -> str:
        """Get metadata for compatibility"""
        return ""

    @metadata_.setter
    def metadata_(self, value: str) -> None:
        """Set metadata for compatibility (no-op since column doesn't exist)"""
        pass

    @property
    def page_numbers_list(self) -> List[int]:
        """Get page numbers as list."""
        if self.page_numbers:
            try:
                return json.loads(self.page_numbers)
            except (json.JSONDecodeError, TypeError):
                return []
        return []

    @page_numbers_list.setter
    def page_numbers_list(self, value: List[int]) -> None:
        """Set page numbers from list."""
        self.page_numbers = json.dumps(value) if value else None

    __table_args__ = (
        Index('idx_chunks_document_id', 'document_id'),
        Index('idx_chunks_text', 'chunk_text', postgresql_using='gin'),
    )

class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255))
    page_numbers = Column(Text)  # JSON array as text for SQLite compatibility
    title = Column(String(255))
    embedding_vector = Column(Text, nullable=False)  # JSON array as text (matches database schema)
    embedding_provider = Column(String(100), nullable=False)
    embedding_model = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    @property
    def page_numbers_list(self) -> List[int]:
        """Get page numbers as list."""
        if self.page_numbers:
            try:
                return json.loads(self.page_numbers)
            except (json.JSONDecodeError, TypeError):
                return []
        return []

    @page_numbers_list.setter
    def page_numbers_list(self, value: List[int]) -> None:
        """Set page numbers from list."""
        self.page_numbers = json.dumps(value) if value else None

    __table_args__ = (
        Index('idx_embeddings_chunk_id', 'chunk_id'),
        Index('idx_embeddings_provider', 'embedding_provider'),
        Index('idx_embeddings_model', 'embedding_model'),
    )

class APISession(Base):
    __tablename__ = "api_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    __table_args__ = (
        Index('idx_sessions_user_id', 'user_id'),
        Index('idx_sessions_access_token', 'access_token'),
    )

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    message = Column(Text, nullable=False)
    response = Column(Text)
    context_docs = Column(Text)  # JSON array of document IDs used for context
    model_used = Column(String(50))
    created_at = Column(DateTime, default=func.now(), nullable=False)

    __table_args__ = (
        Index('idx_chat_user_id', 'user_id'),
        Index('idx_chat_created_at', 'created_at'),
    )

class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    version_number = Column(Integer, nullable=False)
    content = Column(Text)
    changes = Column(Text)  # JSON describing changes made
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Relationships
    creator = relationship("User", backref="document_versions")

    __table_args__ = (
        Index('idx_version_document_id', 'document_id'),
        Index('idx_version_number', 'version_number'),
        Index('idx_version_created_at', 'created_at'),
    )

class DocumentCollaborator(Base):
    __tablename__ = "document_collaborators"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    permission_level = Column(String(20), default="viewer", nullable=False)  # viewer, editor, owner
    added_at = Column(DateTime, default=func.now(), nullable=False)
    added_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    user = relationship("User", foreign_keys=[user_id], backref="document_collaborations")
    adder = relationship("User", foreign_keys=[added_by], backref="added_collaborations")

    __table_args__ = (
        Index('idx_collaborator_document_id', 'document_id'),
        Index('idx_collaborator_user_id', 'user_id'),
        Index('idx_collaborator_permission', 'permission_level'),
        Index('unique_collaborator', 'document_id', 'user_id'),  # Prevent duplicate collaborations
    )

class DocumentComment(Base):
    __tablename__ = "document_comments"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    comment_type = Column(String(20), default="general")  # general, suggestion, question, issue
    position_data = Column(Text)  # JSON for position information (page, coordinates, etc.)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    is_resolved = Column(Boolean, default=False, nullable=False)

    # Relationships
    user = relationship("User", backref="document_comments")

    __table_args__ = (
        Index('idx_comment_document_id', 'document_id'),
        Index('idx_comment_user_id', 'user_id'),
        Index('idx_comment_type', 'comment_type'),
        Index('idx_comment_created_at', 'created_at'),
    )

class DocumentActivity(Base):
    __tablename__ = "document_activities"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    activity_type = Column(String(50), nullable=False)  # created, updated, shared, commented, version_created
    description = Column(Text)
    metadata_ = Column(Text)  # JSON metadata about the activity
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", backref="document_activities")

    __table_args__ = (
        Index('idx_activity_document_id', 'document_id'),
        Index('idx_activity_user_id', 'user_id'),
        Index('idx_activity_type', 'activity_type'),
        Index('idx_activity_created_at', 'created_at'),
    )

class CompanyBranding(Base):
    __tablename__ = "company_branding"

    id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String(255), nullable=False)
    logo_url = Column(Text)

class SystemPrompt(Base):
    __tablename__ = "system_prompt"

    id = Column(Integer, primary_key=True, index=True)
    prompt_text = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class SearchHistory(Base):
    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    search_type = Column(String(20), nullable=False, default='hybrid')  # hybrid, vector, text
    vector_weight = Column(Float, nullable=False, default=0.5)
    text_weight = Column(Float, nullable=False, default=0.5)
    search_results = Column(Text)  # JSON array of result IDs
    result_count = Column(Integer)
    response_time_ms = Column(Integer)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    search_metadata = Column(Text)  # JSON metadata about the search

    __table_args__ = (
        Index('idx_search_history_user_id', 'user_id'),
        Index('idx_search_history_created_at', 'created_at'),
        Index('idx_search_history_search_type', 'search_type'),
    )


class SearchQuery(Base):
    """Model for tracking search queries for relevance tuning."""
    __tablename__ = "search_queries"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    search_count = Column(Integer, default=1, nullable=False)
    first_seen = Column(DateTime, default=func.now(), nullable=False)
    last_seen = Column(DateTime, default=func.now(), nullable=False)
    success_rate = Column(Float, default=0.0)  # Calculated success rate based on feedback

    __table_args__ = (
        Index('idx_search_query_text', 'query_text'),
        Index('idx_search_query_last_seen', 'last_seen'),
        Index('idx_search_query_success_rate', 'success_rate'),
    )


class UserFeedback(Base):
    """Model for storing user feedback on search results."""
    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("search_queries.id", ondelete="CASCADE"), nullable=False)
    feedback_type = Column(String(20), nullable=False)  # positive, negative, neutral, click, skip
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    metadata_ = Column(Text)  # JSON metadata about the feedback

    # Relationships
    query = relationship("SearchQuery", backref="feedback")
    user = relationship("User", backref="feedback")

    __table_args__ = (
        Index('idx_feedback_query_id', 'query_id'),
        Index('idx_feedback_user_id', 'user_id'),
        Index('idx_feedback_timestamp', 'timestamp'),
        Index('idx_feedback_type', 'feedback_type'),
    )

class QuestionAnswerExport(Base):
    """Model for tracking question-answer Excel exports."""
    __tablename__ = "question_answer_exports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    questions_count = Column(Integer, nullable=False)
    document_ids = Column(Text)  # JSON array of document IDs used
    created_at = Column(DateTime, default=func.now(), nullable=False)
    status = Column(String(20), default="completed", nullable=False)  # processing, completed, failed
    
    __table_args__ = (
        Index('idx_export_user_id', 'user_id'),
        Index('idx_export_created_at', 'created_at'),
    )