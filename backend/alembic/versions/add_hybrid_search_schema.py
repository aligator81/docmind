"""Add hybrid search schema

Revision ID: add_hybrid_search_schema
Revises: 09f3b85b358c
Create Date: 2025-10-28 16:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'add_hybrid_search_schema'
down_revision: Union[str, Sequence[str], None] = '1b89493c4887'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Enable pgvector extension for vector operations
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Create search_history table for tracking hybrid search queries
    op.create_table('search_history',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('query_text', sa.Text(), nullable=False),
        sa.Column('search_type', sa.String(20), nullable=False, default='hybrid'),  # hybrid, vector, text
        sa.Column('vector_weight', sa.Float(), nullable=False, default=0.5),
        sa.Column('text_weight', sa.Float(), nullable=False, default=0.5),
        sa.Column('search_results', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('result_count', sa.Integer(), nullable=True),
        sa.Column('response_time_ms', sa.Integer(), nullable=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('search_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    
    # Add full-text search columns to document_chunks
    op.add_column('document_chunks', sa.Column('search_vector', postgresql.TSVECTOR(), nullable=True))
    
    # Convert embedding_vector from text to proper vector type
    op.execute('''
        ALTER TABLE embeddings
        ALTER COLUMN embedding_vector TYPE vector(3072)
        USING embedding_vector::vector(3072)
    ''')
    
    # Add search performance indexes
    op.create_index('idx_search_history_user_id', 'search_history', ['user_id'], unique=False)
    op.create_index('idx_search_history_created_at', 'search_history', ['created_at'], unique=False)
    op.create_index('idx_search_history_search_type', 'search_history', ['search_type'], unique=False)
    
    # Create GIN index for full-text search on document_chunks
    op.execute('CREATE INDEX idx_chunks_search_vector ON document_chunks USING GIN(search_vector)')
    
    # Skip vector index creation due to 2000 dimension limit (3072 dimensions present)
    # Note: Vector similarity search will work without index, but may be slower for large datasets
    
    # Add search configuration table
    op.create_table('search_config',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('config_key', sa.String(255), nullable=False, unique=True),
        sa.Column('config_value', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
    )
    
    # Insert default search configuration
    op.execute('''
        INSERT INTO search_config (config_key, config_value, description, is_active) VALUES
        ('hybrid_search_weights', '{"vector_weight": 0.6, "text_weight": 0.4}', 'Default weights for hybrid search', true),
        ('vector_search_threshold', '{"similarity_threshold": 0.7}', 'Minimum similarity threshold for vector search', true),
        ('full_text_config', '{"language": "english", "stemming": true}', 'Full-text search configuration', true),
        ('search_performance', '{"max_results": 50, "timeout_ms": 5000}', 'Search performance settings', true)
    ''')


def downgrade() -> None:
    """Downgrade schema."""
    # Drop search configuration table
    op.drop_table('search_config')
    
    # Drop indexes
    op.drop_index('idx_embeddings_vector', table_name='embeddings')
    op.drop_index('idx_chunks_search_vector', table_name='document_chunks')
    op.drop_index('idx_search_history_search_type', table_name='search_history')
    op.drop_index('idx_search_history_created_at', table_name='search_history')
    op.drop_index('idx_search_history_user_id', table_name='search_history')
    
    # Convert embedding_vector back to text
    op.execute('''
        ALTER TABLE embeddings 
        ALTER COLUMN embedding_vector TYPE TEXT 
        USING embedding_vector::text
    ''')
    
    # Remove full-text search column
    op.drop_column('document_chunks', 'search_vector')
    
    # Drop search_history table
    op.drop_table('search_history')
    
    # Disable pgvector extension (optional - might be used by other tables)
    # op.execute('DROP EXTENSION IF EXISTS vector')