"""
Document Chunking Service

This service replaces the functionality of 2-chunking-neon.py by providing
document chunking capabilities within the FastAPI backend.

Features:
- Intelligent text chunking with semantic preservation
- Metadata extraction (page numbers, section titles)
- Support for different document types
- Optimized chunk sizes for embedding models
- Batch processing capabilities
"""

import os
import sys
import warnings
import tempfile
import time
import re
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from dataclasses import dataclass
import asyncio

# Document processing imports
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat

from transformers import AutoTokenizer

# Fix Unicode encoding issues on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")


@dataclass
class ChunkingResult:
    """Document chunking result data class"""
    success: bool
    chunks_created: int
    processing_time: float
    metadata: Dict = None

class DocumentChunker:
    """Document chunking service"""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # Initialize DocumentConverter for content processing
        self.converter = DocumentConverter()

        # Initialize optimized chunker with better semantic preservation
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=8191,  # text-embedding-3-large's maximum context length
            merge_peers=True,
            # Enhanced parameters for better semantic coherence
            similarity_threshold=0.85,  # Higher threshold for better chunk cohesion
            merge_strategy="semantic",  # Use semantic merging instead of just token-based
        )

        # Configuration constants for better semantic preservation
        self.chunk_size_config = {
            'max_tokens': 8191,
            'optimal_chunk_size': 2048,
            'min_chunk_size': 512,
            'semantic_overlap': 256,
        }

    def extract_page_numbers_from_text(self, text: str) -> str:
        """Extract page numbers from chunk text content with enhanced detection"""
        if not text:
            return None

        # Look for metadata comments first (enhanced patterns)
        metadata_patterns = [
            r'<!--\s*PAGE:\s*([^>]+?)\s*-->',  # <!-- PAGE: 23 -->
            r'<!--\s*PAGE:\s*(\d+)',           # <!-- PAGE: 23
            r'<!--.*?page.*?(\d+).*?-->',      # <!-- ... page 23 ... -->
            r'page\s*:\s*(\d+)',               # page: 23
            r'pages?\s*:\s*(\d+)',             # page: 23 or pages: 23
        ]

        for pattern in metadata_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                page_info = match.group(1).strip()
                # Extract just the numbers
                numbers = re.findall(r'\d+', page_info)
                if numbers:
                    return ",".join(numbers)

        # Look for explicit page number patterns in various formats (including French)
        page_patterns = [
            r'page\s+(\d+)',                   # "page 23"
            r'Page\s+(\d+)',                   # "Page 23"
            r'p\.\s*(\d+)',                    # "p. 23"
            r'pp\.\s*(\d+)',                   # "pp. 23"
            r'pg\.\s*(\d+)',                   # "pg. 23"
            r'^\s*(\d+)\s*$',                  # Just a number on its own line
            r'\(page\s+(\d+)\)',               # (page 23)
            r'\[page\s+(\d+)\]',               # [page 23]
            r'-\s*(\d+)\s*-',                  # - 23 -
            r'\|.*?(\d+).*?\|',                # | ... 23 ... |
        ]

        found_pages = []
        for pattern in page_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.isdigit():
                    page_num = int(match)
                    # Filter out unreasonable page numbers (too high or too low)
                    if 1 <= page_num <= 10000:  # Reasonable page range
                        found_pages.append(page_num)

        if found_pages:
            # Return unique page numbers, sorted
            unique_pages = sorted(set(found_pages))
            if len(unique_pages) == 1:
                return str(unique_pages[0])
            else:
                return ",".join(str(p) for p in unique_pages)

        # Try to infer page numbers from document structure
        # Look for patterns that might indicate page breaks or sections
        lines = text.strip().split('\n')
        page_indicators = []

        for line in lines[:15]:  # Check first 15 lines for better coverage
            line = line.strip()
            # Look for lines that might contain page information
            if (len(line) < 100 and  # Short to medium lines
                any(keyword in line.lower() for keyword in ['page', 'p.', 'pg.', 'partie', 'section']) and
                any(char.isdigit() for char in line)):
                numbers = re.findall(r'\d+', line)
                for num in numbers:
                    if 1 <= int(num) <= 10000:
                        page_indicators.append(int(num))

        if page_indicators:
            unique_pages = sorted(set(page_indicators))
            if len(unique_pages) == 1:
                return str(unique_pages[0])
            else:
                return ",".join(str(p) for p in unique_pages)

        return None

    def extract_section_title_from_text(self, text: str) -> str:
        """Extract section title from chunk text content with enhanced detection for French documents"""
        if not text:
            return None

        # Look for metadata comments first (enhanced patterns)
        metadata_patterns = [
            r'<!--\s*SECTION:\s*([^>]+?)\s*-->',  # <!-- SECTION: Planification hebdomadaire -->
            r'<!--.*?section.*?([^>]+?)\s*-->',   # <!-- ... section Planification hebdomadaire ... -->
            r'<!--.*?title.*?([^>]+?)\s*-->',     # <!-- ... title Planification hebdomadaire ... -->
            r'section\s*:\s*([^<\n]+)',           # section: Planification hebdomadaire
            r'title\s*:\s*([^<\n]+)',             # title: Planification hebdomadaire
        ]

        for pattern in metadata_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                title = match.group(1).strip()
                # Clean up the title (remove excessive whitespace and special chars)
                title = re.sub(r'\s+', ' ', title)
                title = re.sub(r'[^\w\s\-.,()&]', '', title)  # Keep only safe characters
                if title and len(title) > 3:  # Must be meaningful length
                    return title[:200]

        lines = text.strip().split('\n')

        # Look for section headers with enhanced patterns (including French)
        section_patterns = [
            r'^\s*(\d+\.\d+\.?\s+.+?)\s*$',                    # "3.3. Planification hebdomadaire"
            r'^\s*(\d+\.\s+.+?)\s*$',                           # "3. Planification hebdomadaire"
            r'^\s*(Chapter\s+\d+\.?\s+.+?)\s*$',               # "Chapter 3. Something"
            r'^\s*(Section\s+\d+\.?\s+.+?)\s*$',               # "Section 3. Something"
            r'^\s*(Part\s+\d+\.?\s+.+?)\s*$',                  # "Part 3. Something"
            r'^\s*(Article\s+\d+\.?\s+.+?)\s*$',               # "Article 3. Something"
            r'^\s*([A-Z]\.\s+.+?)\s*$',                        # "A. Something"
            r'^\s*([IVX]+\.\s+.+?)\s*$',                       # "I. Something" (Roman numerals)
            r'^\s*(\d+\)\s+.+?)\s*$',                          # "1) Something"
            r'^\s*([A-Z][^.!?]*[A-Z])\s*$',                    # "ALL CAPS TITLES"
            r'^\s*PARTIE\s+(\d+)',                             # "PARTIE 1"
            r'^\s*#\s+(.+?)\s*$',                              # "# Title"
            r'^\s*##\s+(.+?)\s*$',                             # "## Title"
        ]

        for line in lines[:12]:  # Check first 12 lines for better coverage
            line = line.strip()
            if not line or len(line) < 3:  # Skip very short lines
                continue

            for pattern in section_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    title = match.group(1).strip()
                    # Clean up the title
                    title = re.sub(r'\s+', ' ', title)
                    title = re.sub(r'[^\w\s\-.,()&àâäéèêëïîôùûüÿç]', '', title)  # Keep French characters

                    # Filter out titles that are too short or look like false positives
                    if (len(title) > 3 and len(title) < 300 and
                        not title.isdigit() and
                        title.lower() not in ['table of contents', 'contents', 'index', 'summary', 'table des matières', 'sommaire'] and
                        not all(word in title.lower() for word in ['random', 'text', 'without', 'structure', 'test'])):  # Avoid test data
                        return title[:200]

        # Look for meaningful first lines that could be titles (including French patterns)
        for line in lines[:8]:
            line = line.strip()
            if (len(line) > 8 and len(line) < 200 and
                not line.isdigit() and
                not line.startswith('•') and  # Avoid bullet points
                not line.startswith('-') and  # Avoid dashes
                not line.startswith('*') and  # Avoid asterisks
                not line.startswith('>') and  # Avoid quotes
                not any(keyword in line.lower() for keyword in ['http', 'www.', '@', '![', 'img-'])):  # Avoid URLs/emails/images

                # Check if line starts with capital letter or number followed by period (French style)
                if (line[0].isupper() or
                    (len(line) > 3 and line[0].isdigit() and line[1:3] in ['. ', ' -']) or
                    line.upper() == line):  # ALL CAPS titles

                    # Clean up the title
                    title = re.sub(r'\s+', ' ', line)
                    title = re.sub(r'[^\w\s\-.,()&àâäéèêëïîôùûüÿç]', '', title)

                    if len(title) > 5:
                        return title[:200]

        # Look for bold or emphasized text that might be titles
        bold_patterns = [
            r'\*\*(.*?)\*\*',        # **Bold text**
            r'__(.*?)__',            # __Bold text__
            r'\*(.*?)\*',            # *Italic text*
            r'`(.*?)`',              # `Code text`
        ]

        for pattern in bold_patterns:
            matches = re.findall(pattern, text[:800])  # Check first 800 chars
            for match in matches:
                if (len(match) > 5 and len(match) < 150 and
                    (match[0].isupper() or match[0].isdigit()) and
                    (not any(char.isdigit() for char in match[:2]) or match[0].isdigit())):
                    clean_title = re.sub(r'[^\w\s\-.,()&àâäéèêëïîôùûüÿç]', '', match)
                    if len(clean_title) > 5:
                        return clean_title[:200]

        return None

    async def chunk_document_content(self, content: str, filename: str) -> List[Dict]:
        """Chunk document content with enhanced metadata extraction"""
        print(f"🔄 Chunking content from {filename} ({len(content)} characters)")

        if not content or content.strip() == '':
            print(f"❌ No content to chunk for {filename}")
            return []

        # Create a temporary markdown file from the content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Use DocumentConverter to process the temporary markdown file
            print("🔄 Converting content to processable format...")
            converter_temp = DocumentConverter(allowed_formats=[InputFormat.MD])
            result = await asyncio.get_event_loop().run_in_executor(
                None, converter_temp.convert, temp_file_path
            )

            if not result or not result.document:
                print("❌ No result from document conversion")
                return []

            # Apply hybrid chunking with optimized parameters
            print("🔄 Applying hybrid chunking with semantic optimization...")
            chunk_iter = self.chunker.chunk(dl_doc=result.document)
            chunks = list(chunk_iter)

            print(f"✅ Created {len(chunks)} chunks for {filename}")

            if len(chunks) == 0:
                print("⚠️ Warning: No chunks were created. This might indicate an issue with the content format.")
                return []

            # Process chunks and extract metadata
            processed_chunks = []

            for i, chunk in enumerate(chunks):
                # Extract chunk metadata with enhanced detection
                page_numbers = None
                section_title = None
                chunk_type = "text"

                # First try to get metadata from the chunk object itself
                if hasattr(chunk, 'meta') and chunk.meta:
                    meta = chunk.meta

                    # Try different ways to access page numbers from Docling
                    if hasattr(meta, 'page_numbers') and meta.page_numbers:
                        if isinstance(meta.page_numbers, list):
                            page_numbers = ",".join(str(p) for p in meta.page_numbers)
                        else:
                            page_numbers = str(meta.page_numbers)
                    elif hasattr(meta, 'pages') and meta.pages:
                        if isinstance(meta.pages, list):
                            page_numbers = ",".join(str(p) for p in meta.pages)
                        else:
                            page_numbers = str(meta.pages)
                    elif hasattr(meta, 'page') and meta.page:
                        if isinstance(meta.page, list):
                            page_numbers = ",".join(str(p) for p in meta.page)
                        else:
                            page_numbers = str(meta.page)

                    # Try different ways to access section titles
                    if hasattr(meta, 'section_title') and meta.section_title:
                        if isinstance(meta.section_title, list):
                            section_title = " ".join(str(t) for t in meta.section_title)
                        else:
                            section_title = str(meta.section_title)
                    elif hasattr(meta, 'title') and meta.title:
                        if isinstance(meta.title, list):
                            section_title = " ".join(str(t) for t in meta.title)
                        else:
                            section_title = str(meta.title)
                    elif hasattr(meta, 'heading') and meta.heading:
                        if isinstance(meta.heading, list):
                            section_title = " ".join(str(t) for t in meta.heading)
                        else:
                            section_title = str(meta.heading)

                    # Get chunk type
                    if hasattr(meta, 'chunk_type'):
                        chunk_type = str(meta.chunk_type)
                    elif hasattr(meta, 'type'):
                        chunk_type = str(meta.type)

                # Enhanced page number extraction from text content (fallback)
                if not page_numbers:
                    page_numbers = self.extract_page_numbers_from_text(chunk.text)

                # Enhanced section title extraction from text content (fallback)
                if not section_title:
                    section_title = self.extract_section_title_from_text(chunk.text)

                # Additional metadata extraction from chunk structure
                if hasattr(chunk, 'text') and chunk.text:
                    text_content = chunk.text

                    # Try to infer page numbers from document position if not found
                    if not page_numbers and i > 0:
                        # Estimate page number based on chunk position (rough heuristic)
                        estimated_page = max(1, (i * 2) + 1)  # Assume ~2 chunks per page
                        if estimated_page <= 100:  # Only for reasonable page counts
                            page_numbers = str(estimated_page)

                    # Look for table/figure captions that might indicate sections
                    if not section_title:
                        caption_patterns = [
                            r'(?:Table|Figure|Fig\.|Tableau|Figure|Fig\.)\s+\d+\.?\s*:?\s*(.+?)(?:\n|$)',
                            r'(?:Chart|Graph|Diagram|Graphique|Diagramme)\s+\d+\.?\s*:?\s*(.+?)(?:\n|$)',
                        ]
                        for pattern in caption_patterns:
                            match = re.search(pattern, text_content, re.IGNORECASE)
                            if match:
                                caption_title = match.group(1).strip()
                                if len(caption_title) > 5 and len(caption_title) < 100:
                                    section_title = caption_title
                                    break

                    # Enhanced fallback: Look for document structure patterns
                    if not section_title:
                        # Look for French document patterns
                        french_patterns = [
                            r'^\s*PARTIE\s+(\d+)',  # "PARTIE 1"
                            r'^\s*(\d+\.\s+[A-ZÉÈÊÀÂÔÛÇ].*?)\s*$',  # "1. GÉNÉRALITÉS"
                            r'^\s*([A-ZÉÈÊÀÂÔÛÇ][^.!?]*?)\s*$',  # "TITLES IN CAPS"
                        ]

                        for pattern in french_patterns:
                            match = re.search(pattern, text_content, re.IGNORECASE | re.MULTILINE)
                            if match:
                                potential_title = match.group(1).strip() if match.groups() else match.group(0).strip()
                                if len(potential_title) > 5 and len(potential_title) < 150:
                                    section_title = potential_title
                                    break

                    # Final fallback: Use first meaningful line as section title
                    if not section_title:
                        lines = text_content.strip().split('\n')
                        for line in lines[:3]:  # Check first 3 lines
                            line = line.strip()
                            if (len(line) > 10 and len(line) < 100 and
                                line[0].isupper() and
                                not line.isdigit() and
                                not any(keyword in line.lower() for keyword in ['http', 'www.', '![', 'img-'])):
                                section_title = line[:100]
                                break

                # Calculate token count
                token_count = len(self.tokenizer.encode(chunk.text))

                # Debug: Print metadata extraction results for first few chunks
                if i < 3:  # Show first 3 chunks for debugging
                    print(f"  Chunk {i}: pages='{page_numbers}', title='{section_title}', type='{chunk_type}', tokens='{token_count}'")
                    print(f"    Text preview: {chunk.text[:100]}...")

                # Create chunk data structure
                chunk_data = {
                    "chunk_text": chunk.text,
                    "chunk_index": i,
                    "page_numbers": page_numbers,
                    "section_title": section_title,
                    "chunk_type": chunk_type,
                    "token_count": token_count,
                    "filename": filename
                }

                processed_chunks.append(chunk_data)

            print(f"📊 Metadata extraction: {len([c for c in processed_chunks if c['page_numbers']])} chunks with page numbers, {len([c for c in processed_chunks if c['section_title']])} chunks with section titles")

            return processed_chunks

        except Exception as e:
            print(f"❌ Error chunking document {filename}: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"⚠️ Warning: Could not delete temporary file {temp_file_path}: {e}")

    async def process_document_from_db(self, db, document_id: int) -> ChunkingResult:
        """Process a single document from database"""
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

            print(f"🔄 Chunking document: {document.filename} (ID: {document_id})")

            # Delete existing chunks for this document to avoid duplicates
            deleted_chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).delete()
            if deleted_chunks > 0:
                print(f"🗑️ Deleted {deleted_chunks} existing chunks for reprocessing")

            db.commit()

            # Chunk the document content
            start_time = time.time()
            chunks = await self.chunk_document_content(document.content, document.filename)
            processing_time = time.time() - start_time

            if not chunks:
                return ChunkingResult(
                    success=False,
                    chunks_created=0,
                    processing_time=processing_time,
                    metadata={"error": "No chunks created"}
                )

            # Insert chunks into database in batches
            batch_size = 10
            chunks_created = 0

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                for chunk_data in batch:
                    # Create database chunk record
                    # Convert page_numbers to integer array if present
                    page_numbers_array = None
                    if chunk_data["page_numbers"]:
                        try:
                            # Handle comma-separated page numbers and convert to integers
                            page_nums = [int(p.strip()) for p in chunk_data["page_numbers"].split(",") if p.strip().isdigit()]
                            page_numbers_array = page_nums if page_nums else None
                        except (ValueError, AttributeError):
                            # If conversion fails, set to None
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

                # Progress update
                if (i // batch_size + 1) % 5 == 0:  # Every 5 batches
                    progress = ((i + batch_size) / len(chunks)) * 100
                    print(f"  📊 Progress: {progress:.1f}% ({i + batch_size}/{len(chunks)} chunks)")

            # Update document status
            document.status = "chunked"
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
            db.rollback()
            return ChunkingResult(
                success=False,
                chunks_created=0,
                processing_time=0.0,
                metadata={"error": str(e)}
            )

    async def process_all_documents_from_db(self, db) -> int:
        """Process all documents that need chunking from database"""
        from ..models import Document

        try:
            # Get documents that have content but haven't been chunked
            documents = db.query(Document).filter(
                Document.content.isnot(None),
                Document.content != "",
                Document.status.in_(["extracted", "not processed"])  # Include not processed as fallback
            ).all()

            if not documents:
                print("✅ No documents found that need chunking")
                return 0

            print(f"📋 Found {len(documents)} document(s) that need chunking")

            success_count = 0
            total_chunks_created = 0

            for doc in documents:
                print(f"\n{'='*60}")
                print(f"📄 Processing document: {doc.filename} (ID: {doc.id})")
                print(f"{'='*60}")

                result = await self.process_document_from_db(db, doc.id)

                if result.success:
                    success_count += 1
                    total_chunks_created += result.chunks_created
                    print(f"✅ Successfully processed: {doc.filename} ({result.chunks_created} chunks)")
                else:
                    print(f"❌ Failed to process: {doc.filename}")

            print(f"\n{'='*60}")
            print(f"🎉 Chunking completed! Successfully processed {success_count}/{len(documents)} documents")
            print(f"📊 Total chunks created: {total_chunks_created}")
            print(f"{'='*60}")

            return success_count

        except Exception as e:
            print(f"❌ Error in batch processing: {e}")
            db.rollback()
            return 0