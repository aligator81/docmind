"""
Embedding Service

This service replaces the functionality of 3-embedding-neon.py by providing
embedding creation capabilities within the FastAPI backend.

Features:
- Generate embeddings using OpenAI or Mistral
- Batch processing for efficiency
- Robust error handling and retries
- Rate limiting and timeout management
- Progress tracking and resume capability
"""

import os
import sys
import warnings
import time
import logging
import signal
import pickle
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import asyncio
import numpy as np

# API clients
from openai import OpenAI
from mistralai import Mistral

# Import settings
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import settings

# Fix Unicode encoding issues on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

# Import the tokenizer from utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'utils'))
try:
    from tokenizer import OpenAITokenizerWrapper
except ImportError:
    # Fallback if utils not available
    class OpenAITokenizerWrapper:
        def encode(self, text: str) -> List[int]:
            # Simple fallback tokenizer
            return text.split()

@dataclass
class EmbeddingResult:
    """Embedding creation result data class"""
    success: bool
    embeddings_created: int
    processing_time: float
    metadata: Dict = None

class EmbeddingService:
    """Embedding creation service"""

    def __init__(self, provider: str = "openai"):
        self.provider = provider.lower()
        self.tokenizer = OpenAITokenizerWrapper()

        # Configuration constants for robust large file handling
        self.embedding_timeout = 1800  # 30 minutes timeout per chunk
        self.max_retries = 8  # Maximum retries per chunk
        self.retry_delay = 15  # Delay between retries in seconds
        self.rate_limit_delay = 3  # Delay between API calls
        self.processing_timeout = 14400  # 4 hour overall timeout
        self.progress_save_interval = 3  # Save progress every 3 chunks
        self.checkpoint_file = 'embedding_checkpoint.pkl'

        # Chunk size optimization for large files - reduced for safety with OpenAI's 8192 token limit
        self.max_chunk_size = 3000  # Reduced from 4000 for safety buffer
        self.optimal_chunk_size = 1500
        self.emergency_chunk_size = 800

        # Initialize API clients
        self.openai_client = None
        self.mistral_client = None
        self._initialize_clients()

        # Progress tracking
        self.processed_chunks = set()
        self.failed_chunks = set()
        self.start_time = None
        self.last_progress_save = 0

    def _initialize_clients(self):
        """Initialize API clients based on provider"""
        if self.provider == "openai":
            api_key = settings.openai_api_key
            if not api_key:
                raise ValueError("OpenAI API key not found in configuration")
            self.openai_client = OpenAI(api_key=api_key)
            print("✅ Initialized OpenAI client for embeddings")
        elif self.provider == "mistral":
            api_key = settings.mistral_api_key
            if not api_key:
                raise ValueError("Mistral API key not found in configuration")
            self.mistral_client = Mistral(api_key=api_key)
            print("✅ Initialized Mistral client for embeddings")
        else:
            raise ValueError(f"Invalid provider: {self.provider}. Use 'openai' or 'mistral'")

    def save_checkpoint(self, chunks, current_index):
        """Save processing progress to checkpoint file"""
        checkpoint_data = {
            'processed_chunks': list(self.processed_chunks),
            'failed_chunks': list(self.failed_chunks),
            'current_index': current_index,
            'total_chunks': len(chunks),
            'timestamp': datetime.now().isoformat(),
            'embedding_provider': self.provider
        }

        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"💾 Checkpoint saved - processed: {len(self.processed_chunks)}, current index: {current_index}")
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")

    def load_checkpoint(self):
        """Load processing progress from checkpoint file"""
        if not os.path.exists(self.checkpoint_file):
            return None

        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)

            # Verify checkpoint is for the same provider
            if checkpoint_data.get('embedding_provider') != self.provider:
                print(f"⚠️ Checkpoint is for different provider ({checkpoint_data.get('embedding_provider')}) than current ({self.provider})")
                return None

            print(f"📋 Loaded checkpoint - processed: {len(checkpoint_data['processed_chunks'])}, current index: {checkpoint_data['current_index']}")
            return checkpoint_data
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return None

    def cleanup_checkpoint(self):
        """Clean up checkpoint file"""
        try:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                print("🗑️ Checkpoint file cleaned up")
        except Exception as e:
            print(f"❌ Failed to cleanup checkpoint: {e}")

    def validate_and_split_chunk(self, text: str, max_tokens: int = None, emergency_mode: bool = False) -> Tuple[List[str], List[int]]:
        """Validate chunk size and split if necessary"""
        if max_tokens is None:
            max_tokens = self.max_chunk_size

        # Use emergency chunk size if in emergency mode
        if emergency_mode:
            max_tokens = self.emergency_chunk_size
            print(f"🚨 Emergency mode: Using reduced chunk size of {max_tokens} tokens")

        token_count = len(self.tokenizer.encode(text))
        print(f"📏 Chunk token count: {token_count} (max: {max_tokens})")

        if token_count <= max_tokens:
            return [text], [token_count]

        # Enhanced splitting logic for large chunks
        print(f"⚠️ Chunk too large ({token_count} tokens), splitting into smaller chunks")

        # Try to split by paragraphs first for better semantic boundaries
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            chunks = []
            current_chunk = []
            current_tokens = 0

            for paragraph in paragraphs:
                para_tokens = len(self.tokenizer.encode(paragraph))

                if para_tokens > max_tokens:
                    # Paragraph itself is too large, split by sentences
                    sentences = paragraph.split('. ')
                    for sentence in sentences:
                        sentence_tokens = len(self.tokenizer.encode(sentence))

                        if current_tokens + sentence_tokens > max_tokens and current_chunk:
                            chunks.append(" ".join(current_chunk))
                            current_chunk = [sentence]
                            current_tokens = sentence_tokens
                        else:
                            current_chunk.append(sentence)
                            current_tokens += sentence_tokens
                else:
                    if current_tokens + para_tokens > max_tokens and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [paragraph]
                        current_tokens = para_tokens
                    else:
                        current_chunk.append(paragraph)
                        current_tokens += para_tokens

            # Add final chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
        else:
            # Fallback to word-based splitting
            words = text.split()
            chunks = []
            current_chunk = []
            current_tokens = 0

            for word in words:
                word_tokens = len(self.tokenizer.encode(word + " "))

                if current_tokens + word_tokens > max_tokens and current_chunk:
                    # Save current chunk and start new one
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_tokens = word_tokens
                else:
                    current_chunk.append(word)
                    current_tokens += word_tokens

            # Add final chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))

        # Calculate token counts for all chunks
        token_counts = [len(self.tokenizer.encode(chunk)) for chunk in chunks]

        print(f"✅ Split into {len(chunks)} chunks with token counts: {token_counts}")

        return chunks, token_counts

    async def get_embedding(self, text: str, emergency_mode: bool = False) -> List[float]:
        """Get embedding for text using configured provider"""
        # Validate chunk size first with emergency mode if needed
        sub_chunks, token_counts = self.validate_and_split_chunk(text, emergency_mode=emergency_mode)

        if len(sub_chunks) > 1:
            print(f"🔄 Processing {len(sub_chunks)} sub-chunks for embedding")
            embeddings = []

            for i, sub_chunk in enumerate(sub_chunks):
                print(f"🔄 Getting embedding for sub-chunk {i+1}/{len(sub_chunks)} ({token_counts[i]} tokens)")

                if self.provider == "openai":
                    try:
                        # Get configured embedding model from settings
                        embedding_model = settings.openai_embedding_model
                        response = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.openai_client.embeddings.create(
                                model=embedding_model,
                                input=sub_chunk,
                                timeout=self.embedding_timeout
                            )
                        )
                        embeddings.append(response.data[0].embedding)
                        print(f"✅ Sub-chunk {i+1}/{len(sub_chunks)} embedded successfully using {embedding_model}")
                    except Exception as e:
                        print(f"❌ OpenAI API error for sub-chunk {i+1}: {e}")
                        if "rate limit" in str(e).lower():
                            print("💡 Consider increasing RATE_LIMIT_DELAY to avoid rate limits")
                        elif "timeout" in str(e).lower():
                            print(f"💡 OpenAI request timed out after {self.embedding_timeout}s")
                        raise
                elif self.provider == "mistral":
                    try:
                        # Get configured embedding model from settings
                        embedding_model = settings.mistral_embedding_model
                        response = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.mistral_client.embeddings.create(
                                model=embedding_model,
                                inputs=[sub_chunk]
                            )
                        )
                        embeddings.append(response.data[0].embedding)
                        print(f"✅ Sub-chunk {i+1}/{len(sub_chunks)} embedded successfully using {embedding_model}")
                    except Exception as e:
                        print(f"❌ Mistral API error for sub-chunk {i+1}: {e}")
                        if "rate limit" in str(e).lower():
                            print("💡 Consider increasing RATE_LIMIT_DELAY to avoid rate limits")
                        raise

                # Rate limiting delay between sub-chunks
                if i < len(sub_chunks) - 1:
                    print(f"⏳ Rate limiting delay: {self.rate_limit_delay}s")
                    await asyncio.sleep(self.rate_limit_delay)

            # Average the embeddings for the final result
            if embeddings:
                print(f"✅ Generated {len(embeddings)} embeddings, averaging for final result")
                return np.mean(embeddings, axis=0).tolist()

        else:
            # Single chunk processing
            if self.provider == "openai":
                try:
                    # Get configured embedding model from settings
                    embedding_model = settings.openai_embedding_model
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.openai_client.embeddings.create(
                            model=embedding_model,
                            input=text,
                            timeout=self.embedding_timeout
                        )
                    )
                    return response.data[0].embedding
                except Exception as e:
                    print(f"❌ OpenAI API error: {e}")
                    if "rate limit" in str(e).lower():
                        print("💡 Consider increasing RATE_LIMIT_DELAY to avoid rate limits")
                    elif "timeout" in str(e).lower():
                        print(f"💡 OpenAI request timed out after {self.embedding_timeout}s")
                    raise
            elif self.provider == "mistral":
                try:
                    # Get configured embedding model from settings
                    embedding_model = settings.mistral_embedding_model
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.mistral_client.embeddings.create(
                            model=embedding_model,
                            inputs=[text]
                        )
                    )
                    return response.data[0].embedding
                except Exception as e:
                    print(f"❌ Mistral API error: {e}")
                    if "rate limit" in str(e).lower():
                        print("💡 Consider increasing RATE_LIMIT_DELAY to avoid rate limits")
                    raise

    async def get_embedding_with_emergency_fallback(self, text: str) -> List[float]:
        """Get embedding with emergency fallback for problematic chunks"""
        try:
            # First attempt with normal mode
            return await self.get_embedding(text, emergency_mode=False)
        except Exception as e:
            if "timeout" in str(e).lower() or "too large" in str(e).lower():
                print("🚨 First attempt failed, trying emergency mode with smaller chunks...")
                try:
                    # Second attempt with emergency mode
                    return await self.get_embedding(text, emergency_mode=True)
                except Exception as e2:
                    print(f"❌ Emergency mode also failed: {e2}")
                    raise e2
            else:
                # Re-raise other types of errors
                raise

    async def process_chunk_embedding(self, db, chunk_data: Tuple, chunk_index: int, total_chunks: int) -> bool:
        """Process a single chunk for embedding"""
        from ..models import Document, DocumentChunk, Embedding

        chunk_id, document_id, chunk_text, chunk_idx, page_numbers, section_title, chunk_type, token_count, document_filename = chunk_data

        try:
            print(f"🤖 Processing chunk {chunk_index + 1}/{total_chunks} from {document_filename}")
            print(f"📏 Chunk size: {len(chunk_text)} characters, ~{token_count} tokens")

            # Check chunk size before processing
            if token_count > self.max_chunk_size:
                print(f"⚠️ Large chunk detected: {token_count} tokens (max: {self.max_chunk_size})")

            # Generate embedding
            embedding = await self.get_embedding_with_emergency_fallback(chunk_text)

            print(f"✅ Generated embedding in {len(embedding)} dimensions")

            # Store embedding in database
            # Get configured embedding model from settings
            if self.provider == "openai":
                embedding_model = settings.openai_embedding_model
            else:
                embedding_model = settings.mistral_embedding_model
            
            db_embedding = Embedding(
                chunk_id=chunk_id,
                filename=document_filename,
                original_filename=document_filename,
                page_numbers=page_numbers,
                title=section_title,
                embedding_vector=embedding,
                embedding_provider=self.provider,
                embedding_model=embedding_model
            )

            db.add(db_embedding)
            db.commit()

            print(f"✅ Successfully processed chunk {chunk_index + 1}/{total_chunks} for {document_filename}")
            return True

        except Exception as e:
            print(f"❌ Error processing chunk {chunk_index + 1}: {e}")
            db.rollback()
            return False

    async def process_embeddings_from_db(self, db, resume: bool = False) -> EmbeddingResult:
        """Process all chunks that need embeddings from database"""
        from ..models import Document, DocumentChunk, Embedding

        try:
            # Get chunks that don't have embeddings yet for this provider
            chunks = db.query(DocumentChunk).join(
                Document, DocumentChunk.document_id == Document.id
            ).outerjoin(
                Embedding, DocumentChunk.id == Embedding.chunk_id
            ).filter(
                Embedding.id.is_(None)  # No embedding exists
            ).all()

            if not chunks:
                print("✅ No chunks found that need embedding processing")
                return EmbeddingResult(
                    success=True,
                    embeddings_created=0,
                    processing_time=0.0,
                    metadata={"message": "No chunks need processing"}
                )

            print(f"🔍 Found {len(chunks)} chunk(s) that need embedding processing")

            # Load checkpoint if resume requested
            resume_index = 0
            if resume:
                checkpoint = self.load_checkpoint()
                if checkpoint:
                    self.processed_chunks = set(checkpoint['processed_chunks'])
                    self.failed_chunks = set(checkpoint['failed_chunks'])
                    resume_index = checkpoint['current_index']

            self.start_time = time.time()
            successful_embeddings = 0
            failed_embeddings = 0

            print(f"🧬 Starting embedding generation using {self.provider}")
            print(f"📝 Processing {len(chunks)} chunks with resume capability")

            for i in range(resume_index, len(chunks)):
                chunk = chunks[i]

                # Skip if already processed
                if chunk.id in self.processed_chunks:
                    print(f"⏭️ Skipping already processed chunk {i + 1}/{len(chunks)}")
                    continue

                # Skip if previously failed
                if chunk.id in self.failed_chunks:
                    print(f"⏭️ Skipping previously failed chunk {i + 1}/{len(chunks)}")
                    failed_embeddings += 1
                    continue

                try:
                    print(f"🔄 Processing chunk {i + 1}/{len(chunks)} from document ID: {chunk.document_id}")

                    # Convert chunk to tuple format expected by process_chunk_embedding
                    chunk_data = (
                        chunk.id,
                        chunk.document_id,
                        chunk.chunk_text,
                        chunk.chunk_index,
                        chunk.page_numbers,
                        chunk.section_title,
                        chunk.chunk_type,
                        chunk.token_count,
                        "Unknown Document"  # We don't have filename in this context
                    )

                    # Process chunk
                    if await self.process_chunk_embedding(db, chunk_data, i, len(chunks)):
                        successful_embeddings += 1
                        self.processed_chunks.add(chunk.id)
                    else:
                        failed_embeddings += 1
                        self.failed_chunks.add(chunk.id)

                    # Rate limiting delay
                    if i < len(chunks) - 1:
                        print(f"⏳ Rate limiting delay: {self.rate_limit_delay}s")
                        await asyncio.sleep(self.rate_limit_delay)

                    # Save progress periodically
                    if (i + 1) % self.progress_save_interval == 0:
                        self.save_checkpoint(chunks, i + 1)

                    # Log progress
                    elapsed_time = time.time() - self.start_time
                    chunks_per_second = (i + 1) / elapsed_time if elapsed_time > 0 else 0
                    eta_seconds = (len(chunks) - i - 1) / chunks_per_second if chunks_per_second > 0 else 0

                    print(f"📊 Progress: {i + 1}/{len(chunks)} ({((i + 1) / len(chunks)) * 100:.1f}%) - "
                          f"Success: {successful_embeddings}, Failed: {failed_embeddings}, "
                          f"Rate: {chunks_per_second:.2f} chunks/s")

                except Exception as e:
                    failed_embeddings += 1
                    self.failed_chunks.add(chunk.id)
                    print(f"❌ Error processing chunk {i + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Save final progress
            self.save_checkpoint(chunks, len(chunks))

            processing_time = time.time() - self.start_time

            print("🎉 Embedding generation completed!")
            print(f"📊 Results: {successful_embeddings} successful, {failed_embeddings} failed")
            print(f"⏱️ Total processing time: {processing_time:.2f} seconds")

            # Verify embeddings were actually stored
            final_count = db.query(Embedding).filter(
                Embedding.embedding_provider == self.provider
            ).count()

            print(f"📊 Total embeddings in database for {self.provider}: {final_count}")

            return EmbeddingResult(
                success=successful_embeddings > 0,
                embeddings_created=successful_embeddings,
                processing_time=processing_time,
                metadata={
                    "total_chunks": len(chunks),
                    "failed_embeddings": failed_embeddings,
                    "final_embedding_count": final_count
                }
            )

        except Exception as e:
            print(f"❌ Error in embedding processing: {e}")
            db.rollback()
            return EmbeddingResult(
                success=False,
                embeddings_created=0,
                processing_time=0.0,
                metadata={"error": str(e)}
            )

    def get_chunks_needing_embeddings(self, db) -> List:
        """Get chunks that need embeddings for this provider"""
        from ..models import Document, DocumentChunk, Embedding

        chunks = db.query(DocumentChunk).outerjoin(
            Embedding, DocumentChunk.id == Embedding.chunk_id
        ).filter(
            Embedding.id.is_(None)  # No embedding exists
        ).all()

        return chunks

    async def process_embeddings_for_document(self, db, document_id: int) -> EmbeddingResult:
        """Process embeddings for chunks of a specific document"""
        from ..models import Document, DocumentChunk, Embedding

        try:
            # Get chunks that don't have embeddings yet for this specific document
            chunks = db.query(DocumentChunk).join(
                Document, DocumentChunk.document_id == Document.id
            ).outerjoin(
                Embedding, DocumentChunk.id == Embedding.chunk_id
            ).filter(
                DocumentChunk.document_id == document_id,
                Embedding.id.is_(None)  # No embedding exists
            ).all()

            if not chunks:
                print(f"✅ No chunks found that need embedding processing for document {document_id}")
                return EmbeddingResult(
                    success=True,
                    embeddings_created=0,
                    processing_time=0.0,
                    metadata={"message": "No chunks need processing"}
                )

            print(f"🔍 Found {len(chunks)} chunk(s) that need embedding processing for document {document_id}")

            self.start_time = time.time()
            successful_embeddings = 0
            failed_embeddings = 0

            print(f"🧬 Starting embedding generation for document {document_id} using {self.provider}")
            print(f"📝 Processing {len(chunks)} chunks")

            for i in range(len(chunks)):
                chunk = chunks[i]

                try:
                    print(f"🔄 Processing chunk {i + 1}/{len(chunks)} from document {document_id}")

                    # Convert chunk to tuple format expected by process_chunk_embedding
                    chunk_data = (
                        chunk.id,
                        chunk.document_id,
                        chunk.chunk_text,
                        chunk.chunk_index,
                        chunk.page_numbers,
                        chunk.section_title,
                        chunk.chunk_type,
                        chunk.token_count,
                        "Unknown Document"  # We don't have filename in this context
                    )

                    # Process chunk
                    if await self.process_chunk_embedding(db, chunk_data, i, len(chunks)):
                        successful_embeddings += 1
                    else:
                        failed_embeddings += 1

                    # Rate limiting delay
                    if i < len(chunks) - 1:
                        print(f"⏳ Rate limiting delay: {self.rate_limit_delay}s")
                        await asyncio.sleep(self.rate_limit_delay)

                    # Log progress
                    elapsed_time = time.time() - self.start_time
                    chunks_per_second = (i + 1) / elapsed_time if elapsed_time > 0 else 0

                    print(f"📊 Progress: {i + 1}/{len(chunks)} ({((i + 1) / len(chunks)) * 100:.1f}%) - "
                          f"Success: {successful_embeddings}, Failed: {failed_embeddings}, "
                          f"Rate: {chunks_per_second:.2f} chunks/s")

                except Exception as e:
                    failed_embeddings += 1
                    print(f"❌ Error processing chunk {i + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            processing_time = time.time() - self.start_time

            print("🎉 Embedding generation completed for document!")
            print(f"📊 Results: {successful_embeddings} successful, {failed_embeddings} failed")
            print(f"⏱️ Total processing time: {processing_time:.2f} seconds")

            return EmbeddingResult(
                success=successful_embeddings > 0,
                embeddings_created=successful_embeddings,
                processing_time=processing_time,
                metadata={
                    "document_id": document_id,
                    "total_chunks": len(chunks),
                    "failed_embeddings": failed_embeddings
                }
            )

        except Exception as e:
            print(f"❌ Error in document-specific embedding processing: {e}")
            db.rollback()
            return EmbeddingResult(
                success=False,
                embeddings_created=0,
                processing_time=0.0,
                metadata={"error": str(e)}
            )

    def get_embedding_stats(self, db) -> Dict:
        """Get embedding statistics"""
        from ..models import Embedding

        try:
            total_embeddings = db.query(Embedding).filter(
                Embedding.embedding_provider == self.provider
            ).count()

            # Get breakdown by model
            model_breakdown = {}
            for model in ["text-embedding-3-large", "mistral-embed"]:
                count = db.query(Embedding).filter(
                    Embedding.embedding_provider == self.provider,
                    Embedding.embedding_model == model
                ).count()
                if count > 0:
                    model_breakdown[model] = count

            return {
                "provider": self.provider,
                "total_embeddings": total_embeddings,
                "model_breakdown": model_breakdown,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}