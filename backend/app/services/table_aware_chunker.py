"""
Table-Aware Chunking Service

This service extends the existing chunking system with specialized table processing
capabilities, ensuring table structure is preserved during chunking and embedding.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .improved_chunker import ImprovedDocumentChunker, ChunkingResult
from .pdf_table_processor import PDFTableProcessor, TableStructure, TableCell

logger = logging.getLogger(__name__)

class TableChunkingStrategy(Enum):
    """Table-specific chunking strategies"""
    HORIZONTAL_SLICES = "horizontal_slices"  # Row-based chunks
    VERTICAL_COLUMNS = "vertical_columns"    # Column-based chunks
    SEMANTIC_BLOCKS = "semantic_blocks"      # Logical groupings
    COMPLETE_TABLE = "complete_table"        # Small tables as single chunks
    HYBRID = "hybrid"                        # Mixed approach

@dataclass
class TableChunk:
    """Specialized chunk for table data"""
    chunk_text: str
    chunk_index: int
    page_numbers: Optional[str]
    section_title: Optional[str]
    chunk_type: str
    token_count: int
    filename: str
    table_metadata: Dict[str, Any]
    table_slice: Dict[str, Any]

class TableAwareChunker(ImprovedDocumentChunker):
    """
    Enhanced chunker that preserves table structure and semantics
    """
    
    def __init__(self):
        super().__init__()
        self.table_processor = PDFTableProcessor()
        
        # Table chunking parameters
        self.max_rows_per_chunk = 10
        self.max_columns_per_chunk = 4
        self.min_table_chunk_size = 100  # Minimum characters for table chunk
        
    def split_content_into_chunks(self, content: str, filename: str) -> List[Dict]:
        """
        Enhanced chunking with table structure preservation
        """
        logger.info(f"🔄 Table-aware chunking for: {filename}")
        
        # Extract tables from content
        tables = self._extract_tables_from_content(content, filename)
        logger.info(f"📊 Found {len(tables)} tables in content")
        
        # Create table chunks
        table_chunks = self._create_table_chunks(tables, filename)
        
        # Remove table content from main text to avoid duplication
        content_without_tables = self._remove_tables_from_content(content, tables)
        
        # Process remaining text with original logic
        text_chunks = super().split_content_into_chunks(content_without_tables, filename)
        
        # Combine table and text chunks
        all_chunks = text_chunks + table_chunks
        
        # Sort chunks by original position
        all_chunks.sort(key=lambda x: x.get('original_position', 0))
        
        logger.info(f"✅ Created {len(all_chunks)} total chunks ({len(table_chunks)} table chunks)")
        return all_chunks
    
    def _extract_tables_from_content(self, content: str, filename: str) -> List[TableStructure]:
        """
        Extract tables from document content
        """
        tables = []
        
        # Look for table markers in content
        table_markers = self._find_table_markers(content)
        
        for marker in table_markers:
            try:
                table_structure = self._parse_table_from_marker(marker, filename)
                if table_structure:
                    tables.append(table_structure)
            except Exception as e:
                logger.warning(f"Failed to parse table from marker: {e}")
                continue
        
        return tables
    
    def _find_table_markers(self, content: str) -> List[Dict]:
        """
        Find table markers in the content
        """
        markers = []
        
        # Look for table start/end comments
        table_pattern = r'<!-- TABLE_START: (.+?) -->(.*?)<!-- TABLE_END: \1 -->'
        
        for match in re.finditer(table_pattern, content, re.DOTALL):
            table_name = match.group(1)
            table_content = match.group(2)
            
            markers.append({
                "name": table_name,
                "content": table_content,
                "position": match.start(),
                "full_match": match.group(0)
            })
        
        return markers
    
    def _parse_table_from_marker(self, marker: Dict, filename: str) -> Optional[TableStructure]:
        """
        Parse table structure from content marker
        """
        content = marker["content"]
        
        # Extract table metadata
        metadata = self._extract_table_metadata(content)
        
        # Parse markdown table
        table_data = self._parse_markdown_table(content)
        
        if not table_data or not table_data.get("rows"):
            return None
        
        # Create table structure
        return TableStructure(
            headers=table_data.get("headers", []),
            rows=table_data.get("rows", []),
            caption=metadata.get("caption"),
            page_number=metadata.get("page_number", 0),
            bbox=(0, 0, 0, 0),  # Position-based bbox
            complexity=metadata.get("complexity", "simple"),
            extraction_method=metadata.get("extraction_method", "text_based"),
            confidence_score=metadata.get("confidence", 0.7)
        )
    
    def _extract_table_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extract table metadata from content
        """
        metadata = {}
        
        # Extract caption
        caption_match = re.search(r'\*\*Table \d+\*\*:?\s*(.+?)(?:\n|\*)', content)
        if caption_match:
            metadata["caption"] = caption_match.group(1).strip()
        
        # Extract extraction method and confidence
        method_match = re.search(r'Extracted with (\w+).*?confidence:?[\s]*([\d.]+)', content)
        if method_match:
            metadata["extraction_method"] = method_match.group(1).lower()
            metadata["confidence"] = float(method_match.group(2))
        
        # Extract page number
        page_match = re.search(r'Page\s+(\d+)', content)
        if page_match:
            metadata["page_number"] = int(page_match.group(1)) - 1
        
        return metadata
    
    def _parse_markdown_table(self, content: str) -> Dict[str, Any]:
        """
        Parse markdown table into structured data
        """
        lines = content.strip().split('\n')
        
        # Find the actual table lines (skip metadata lines)
        table_lines = []
        in_table = False
        
        for line in lines:
            if '|' in line and '---' not in line:
                in_table = True
                table_lines.append(line)
            elif in_table and not line.strip():
                break
        
        if len(table_lines) < 2:
            return {}
        
        # Parse headers (first line)
        headers = self._parse_markdown_row(table_lines[0])
        
        # Parse data rows
        rows = []
        for i, line in enumerate(table_lines[1:]):
            row_cells = self._parse_markdown_row(line)
            cells = [
                TableCell(row=i, col=j, content=cell_content)
                for j, cell_content in enumerate(row_cells)
            ]
            rows.append(cells)
        
        return {
            "headers": headers,
            "rows": rows
        }
    
    def _parse_markdown_row(self, line: str) -> List[str]:
        """Parse a single markdown table row"""
        clean_line = line.strip().strip('|')
        return [cell.strip() for cell in clean_line.split('|')]
    
    def _create_table_chunks(self, tables: List[TableStructure], filename: str) -> List[Dict]:
        """
        Create specialized chunks for table data
        """
        table_chunks = []
        chunk_index = 0
        
        for table_idx, table in enumerate(tables):
            # Determine optimal chunking strategy for this table
            strategy = self._determine_chunking_strategy(table)
            
            # Create chunks based on strategy
            if strategy == TableChunkingStrategy.COMPLETE_TABLE:
                chunks = self._chunk_complete_table(table, table_idx, filename, chunk_index)
            elif strategy == TableChunkingStrategy.HORIZONTAL_SLICES:
                chunks = self._chunk_horizontal_slices(table, table_idx, filename, chunk_index)
            elif strategy == TableChunkingStrategy.VERTICAL_COLUMNS:
                chunks = self._chunk_vertical_columns(table, table_idx, filename, chunk_index)
            else:
                chunks = self._chunk_semantic_blocks(table, table_idx, filename, chunk_index)
            
            table_chunks.extend(chunks)
            chunk_index += len(chunks)
        
        return table_chunks
    
    def _determine_chunking_strategy(self, table: TableStructure) -> TableChunkingStrategy:
        """
        Determine optimal chunking strategy for a table
        """
        row_count = len(table.rows)
        col_count = len(table.headers) if table.headers else 0
        
        # Small tables as complete chunks
        if row_count <= 5 and col_count <= 4:
            return TableChunkingStrategy.COMPLETE_TABLE
        
        # Large row count - use horizontal slices
        if row_count > 10:
            return TableChunkingStrategy.HORIZONTAL_SLICES
        
        # Large column count - use vertical slices
        if col_count > 6:
            return TableChunkingStrategy.VERTICAL_COLUMNS
        
        # Default to semantic blocks
        return TableChunkingStrategy.SEMANTIC_BLOCKS
    
    def _chunk_complete_table(self, table: TableStructure, table_idx: int, 
                            filename: str, start_index: int) -> List[Dict]:
        """Chunk small tables as complete units"""
        table_text = self.table_processor.export_table_to_markdown(table)
        token_count = len(self.tokenizer.encode(table_text))
        
        return [{
            "chunk_text": table_text,
            "chunk_index": start_index,
            "page_numbers": str(table.page_number + 1) if table.page_number is not None else None,
            "section_title": f"Table {table_idx + 1}",
            "chunk_type": "table_complete",
            "token_count": token_count,
            "filename": filename,
            "table_metadata": {
                "table_number": table_idx + 1,
                "strategy": "complete_table",
                "row_count": len(table.rows),
                "column_count": len(table.headers),
                "confidence": table.confidence_score
            },
            "table_slice": {
                "start_row": 0,
                "end_row": len(table.rows) - 1,
                "start_col": 0,
                "end_col": len(table.headers) - 1
            }
        }]
    
    def _chunk_horizontal_slices(self, table: TableStructure, table_idx: int,
                               filename: str, start_index: int) -> List[Dict]:
        """Chunk table by rows (horizontal slices)"""
        chunks = []
        
        for i in range(0, len(table.rows), self.max_rows_per_chunk):
            end_row = min(i + self.max_rows_per_chunk, len(table.rows))
            slice_rows = table.rows[i:end_row]
            
            # Create slice table structure
            slice_table = TableStructure(
                headers=table.headers,
                rows=slice_rows,
                caption=table.caption,
                page_number=table.page_number,
                bbox=table.bbox,
                complexity=table.complexity,
                extraction_method=table.extraction_method,
                confidence_score=table.confidence_score
            )
            
            table_text = self.table_processor.export_table_to_markdown(slice_table)
            token_count = len(self.tokenizer.encode(table_text))
            
            chunks.append({
                "chunk_text": table_text,
                "chunk_index": start_index + len(chunks),
                "page_numbers": str(table.page_number + 1) if table.page_number is not None else None,
                "section_title": f"Table {table_idx + 1} (Rows {i+1}-{end_row})",
                "chunk_type": "table_horizontal_slice",
                "token_count": token_count,
                "filename": filename,
                "table_metadata": {
                    "table_number": table_idx + 1,
                    "strategy": "horizontal_slices",
                    "total_rows": len(table.rows),
                    "confidence": table.confidence_score
                },
                "table_slice": {
                    "start_row": i,
                    "end_row": end_row - 1,
                    "start_col": 0,
                    "end_col": len(table.headers) - 1
                }
            })
        
        return chunks
    
    def _chunk_vertical_columns(self, table: TableStructure, table_idx: int,
                              filename: str, start_index: int) -> List[Dict]:
        """Chunk table by columns (vertical slices)"""
        chunks = []
        
        if not table.headers:
            return chunks
        
        for i in range(0, len(table.headers), self.max_columns_per_chunk):
            end_col = min(i + self.max_columns_per_chunk, len(table.headers))
            slice_headers = table.headers[i:end_col]
            
            # Extract corresponding column data
            slice_rows = []
            for row in table.rows:
                slice_cells = row[i:end_col]
                slice_rows.append(slice_cells)
            
            # Create slice table structure
            slice_table = TableStructure(
                headers=slice_headers,
                rows=slice_rows,
                caption=table.caption,
                page_number=table.page_number,
                bbox=table.bbox,
                complexity=table.complexity,
                extraction_method=table.extraction_method,
                confidence_score=table.confidence_score
            )
            
            table_text = self.table_processor.export_table_to_markdown(slice_table)
            token_count = len(self.tokenizer.encode(table_text))
            
            chunks.append({
                "chunk_text": table_text,
                "chunk_index": start_index + len(chunks),
                "page_numbers": str(table.page_number + 1) if table.page_number is not None else None,
                "section_title": f"Table {table_idx + 1} (Columns {i+1}-{end_col})",
                "chunk_type": "table_vertical_slice",
                "token_count": token_count,
                "filename": filename,
                "table_metadata": {
                    "table_number": table_idx + 1,
                    "strategy": "vertical_columns",
                    "total_columns": len(table.headers),
                    "confidence": table.confidence_score
                },
                "table_slice": {
                    "start_row": 0,
                    "end_row": len(table.rows) - 1,
                    "start_col": i,
                    "end_col": end_col - 1
                }
            })
        
        return chunks
    
    def _chunk_semantic_blocks(self, table: TableStructure, table_idx: int,
                             filename: str, start_index: int) -> List[Dict]:
        """Chunk table into semantic blocks (logical groupings)"""
        # For now, use complete table as fallback
        # In practice, you'd implement more sophisticated semantic grouping
        return self._chunk_complete_table(table, table_idx, filename, start_index)
    
    def _remove_tables_from_content(self, content: str, tables: List[TableStructure]) -> str:
        """
        Remove table content to avoid duplication
        """
        content_clean = content
        
        # Remove table markers
        table_pattern = r'<!-- TABLE_START: .+? -->.*?<!-- TABLE_END: .+? -->'
        content_clean = re.sub(table_pattern, '', content_clean, flags=re.DOTALL)
        
        return content_clean
    
    def validate_table_chunks(self, chunks: List[Dict]) -> Dict[str, Any]:
        """
        Validate table chunks for quality and consistency
        """
        table_chunks = [chunk for chunk in chunks if chunk.get('chunk_type', '').startswith('table_')]
        
        metrics = {
            "total_table_chunks": len(table_chunks),
            "chunk_types": {},
            "average_confidence": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        if not table_chunks:
            return metrics
        
        # Count chunk types
        for chunk in table_chunks:
            chunk_type = chunk.get('chunk_type', 'unknown')
            metrics["chunk_types"][chunk_type] = metrics["chunk_types"].get(chunk_type, 0) + 1
        
        # Calculate average confidence
        total_confidence = 0.0
        for chunk in table_chunks:
            metadata = chunk.get('table_metadata', {})
            total_confidence += metadata.get('confidence', 0.5)
        
        metrics["average_confidence"] = total_confidence / len(table_chunks)
        
        # Check for issues
        for chunk in table_chunks:
            token_count = chunk.get('token_count', 0)
            if token_count < self.min_table_chunk_size:
                metrics["issues"].append(f"Chunk {chunk['chunk_index']} is too small ({token_count} tokens)")
        
        # Generate recommendations
        if metrics["average_confidence"] < 0.6:
            metrics["recommendations"].append("Consider improving table extraction quality")
        
        if len(table_chunks) == 0:
            metrics["recommendations"].append("No table chunks found - check table extraction")
        
        return metrics