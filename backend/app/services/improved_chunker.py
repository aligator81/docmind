"""
Improved Document Chunking Service

This service provides better chunking logic that creates multiple semantic chunks
for documents, fixing the issue where only 1 chunk was created for entire documents.
"""

import os
import re
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
import asyncio

from transformers import AutoTokenizer

@dataclass
class ChunkingResult:
    """Document chunking result data class"""
    success: bool
    chunks_created: int
    processing_time: float
    metadata: Dict = None

class ImprovedDocumentChunker:
    """Improved document chunking service with better semantic splitting"""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Optimized chunking parameters for better semantic preservation
        self.chunk_size_config = {
            'max_tokens': 512,  # Smaller chunks for better semantic search
            'optimal_chunk_size': 384,
            'min_chunk_size': 128,
            'semantic_overlap': 64,
        }

    def extract_page_numbers_from_text(self, text: str) -> Optional[str]:
        """Extract page numbers from chunk text content"""
        if not text:
            return None

        # Look for page number patterns
        page_patterns = [
            r'---\s*Page\s+(\d+)\s*---',  # --- Page 1 ---
            r'page\s+(\d+)',               # page 23
            r'Page\s+(\d+)',               # Page 23
            r'p\.\s*(\d+)',                # p. 23
        ]
        
        found_pages = []
        for pattern in page_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.isdigit():
                    page_num = int(match)
                    if 1 <= page_num <= 1000:  # Reasonable page range
                        found_pages.append(page_num)
        
        if found_pages:
            return ",".join(str(p) for p in sorted(set(found_pages)))
        return None

    def extract_section_title_from_text(self, text: str) -> Optional[str]:
        """Extract section title from chunk text content"""
        if not text:
            return None

        lines = text.strip().split('\n')
        
        # Look for section headers
        section_patterns = [
            r'^\s*(\d+\.\d+\.?\s+.+?)\s*$',  # "3.3. Planification hebdomadaire"
            r'^\s*(\d+\.\s+.+?)\s*$',        # "3. Planification hebdomadaire"
            r'^\s*([A-Z][^.!?]*[A-Z])\s*$',  # "ALL CAPS TITLES"
            r'^\s*#\s+(.+?)\s*$',            # "# Title"
        ]

        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if not line or len(line) < 3:
                continue

            for pattern in section_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    title = match.group(1).strip()
                    if len(title) > 3 and len(title) < 200:
                        return title[:200]
        
        # Look for meaningful first lines
        for line in lines[:3]:
            line = line.strip()
            if (len(line) > 8 and len(line) < 150 and
                line[0].isupper() and
                not line.isdigit() and
                not any(keyword in line.lower() for keyword in ['http', 'www.', '![', 'img-'])):
                return line[:150]
        
        return None

    def split_content_into_chunks(self, content: str, filename: str) -> List[Dict]:
        """Split document content into semantic chunks with improved logic"""
        print(f"🔄 Splitting content from {filename} ({len(content)} characters)")
        
        if not content or content.strip() == '':
            print(f"❌ No content to chunk for {filename}")
            return []

        # Split by major sections first (page breaks, headers, etc.)
        sections = self._split_into_sections(content)
        print(f"📄 Found {len(sections)} major sections")

        chunks = []
        chunk_index = 0

        for section_idx, section in enumerate(sections):
            # Further split each section into smaller chunks
            section_chunks = self._split_section_into_chunks(section, section_idx)
            
            for chunk_text in section_chunks:
                # Extract metadata
                page_numbers = self.extract_page_numbers_from_text(chunk_text)
                section_title = self.extract_section_title_from_text(chunk_text)
                
                # Calculate token count
                token_count = len(self.tokenizer.encode(chunk_text))
                
                # Skip chunks that are too small
                if token_count < self.chunk_size_config['min_chunk_size']:
                    continue

                chunk_data = {
                    "chunk_text": chunk_text,
                    "chunk_index": chunk_index,
                    "page_numbers": page_numbers,
                    "section_title": section_title,
                    "chunk_type": "text",
                    "token_count": token_count,
                    "filename": filename
                }
                
                chunks.append(chunk_data)
                chunk_index += 1

        print(f"✅ Created {len(chunks)} chunks for {filename}")
        return chunks

    def _split_into_sections(self, content: str) -> List[str]:
        """Split content into major sections based on document structure"""
        sections = []
        
        # Split by page breaks (--- Page X ---)
        page_splits = re.split(r'---\s*Page\s+\d+\s*---', content)
        
        for page_split in page_splits:
            if not page_split.strip():
                continue
            
            # Further split by major headers (all caps, numbered sections)
            header_patterns = [
                r'\n\s*[A-Z][A-Z\s]{10,}\n',  # ALL CAPS HEADERS
                r'\n\s*\d+\.\s+[A-Z]',        # Numbered sections
                r'\n\s*[A-Z][^.!?]{5,50}\n',  # Title-like lines
            ]
            
            current_section = page_split
            for pattern in header_patterns:
                splits = re.split(pattern, current_section)
                if len(splits) > 1:
                    # Keep the splits as separate sections
                    sections.extend([s for s in splits if s.strip()])
                    break
            else:
                # No header splits found, use the whole page
                sections.append(current_section)
        
        # If no page splits found, split by double newlines
        if not sections:
            sections = [s for s in content.split('\n\n') if s.strip()]
        
        return sections

    def _split_section_into_chunks(self, section: str, section_idx: int) -> List[str]:
        """Split a section into smaller chunks based on token limits"""
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        # Split section into paragraphs
        paragraphs = section.split('\n\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            para_tokens = len(self.tokenizer.encode(paragraph))
            
            # If paragraph itself is too large, split by sentences
            if para_tokens > self.chunk_size_config['max_tokens']:
                sentences = re.split(r'[.!?]+', paragraph)
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    sentence_tokens = len(self.tokenizer.encode(sentence))
                    
                    if current_tokens + sentence_tokens > self.chunk_size_config['max_tokens'] and current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                        current_tokens = sentence_tokens
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                        current_tokens += sentence_tokens
            else:
                # Check if adding this paragraph would exceed token limit
                if current_tokens + para_tokens > self.chunk_size_config['max_tokens'] and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                    current_tokens = para_tokens
                else:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                    current_tokens += para_tokens
        
        # Add the final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    async def process_document_from_db(self, db, document_id: int) -> ChunkingResult:
        """Process a single document from database with improved chunking"""
        from ..models import Document, DocumentChunk

        try:
            # Get document from database
            document = db.query(Document).filter(Document.id == document_id).first()

            if not document:
                return ChunkingResult(
                    success=False,
                    chunks_created=0,
                    processing_time=0.0,
                    metadata={"error": "Document not found"}
                )

            if not document.content:
                return ChunkingResult(
                    success=False,
                    chunks_created=0,
                    processing_time=0.0,
                    metadata={"error": "Document has no extracted content"}
                )

            print(f"🔄 Improved chunking for: {document.filename} (ID: {document_id})")

            # Delete existing chunks for this document
            deleted_chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).delete()
            if deleted_chunks > 0:
                print(f"🗑️ Deleted {deleted_chunks} existing chunks for reprocessing")

            db.commit()

            # Chunk the document content with improved logic
            start_time = time.time()
            chunks = self.split_content_into_chunks(document.content, document.filename)
            processing_time = time.time() - start_time

            if not chunks:
                return ChunkingResult(
                    success=False,
                    chunks_created=0,
                    processing_time=processing_time,
                    metadata={"error": "No chunks created"}
                )

            # Insert chunks into database
            chunks_created = 0
            for chunk_data in chunks:
                # Convert page_numbers to integer array if present
                page_numbers_array = None
                if chunk_data["page_numbers"]:
                    try:
                        page_nums = [int(p.strip()) for p in chunk_data["page_numbers"].split(",") if p.strip().isdigit()]
                        page_numbers_array = page_nums if page_nums else None
                    except (ValueError, AttributeError):
                        page_numbers_array = None

                db_chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_text=chunk_data["chunk_text"],
                    chunk_index=chunk_data["chunk_index"],
                    page_numbers=page_numbers_array,
                    section_title=chunk_data["section_title"],
                    chunk_type=chunk_data["chunk_type"],
                    token_count=chunk_data["token_count"]
                )

                db.add(db_chunk)
                chunks_created += 1

            db.commit()

            # Update document status
            document.status = "chunked"
            from datetime import datetime
            document.processed_at = datetime.utcnow()
            db.commit()

            print(f"✅ Successfully chunked {document.filename} - created {chunks_created} chunks in {processing_time:.2f} seconds")

            return ChunkingResult(
                success=True,
                chunks_created=chunks_created,
                processing_time=processing_time,
                metadata={
                    "document_id": document_id,
                    "filename": document.filename,
                    "chunks_with_pages": len([c for c in chunks if c["page_numbers"]]),
                    "chunks_with_titles": len([c for c in chunks if c["section_title"]])
                }
            )

        except Exception as e:
            print(f"❌ Error processing document {document_id}: {e}")
            import traceback
            traceback.print_exc()
            db.rollback()
            return ChunkingResult(
                success=False,
                chunks_created=0,
                processing_time=0.0,
                metadata={"error": str(e)}
            )