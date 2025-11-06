
"""
Advanced PDF Table Processor for Complex Table Extraction

This service provides specialized table extraction capabilities for PDF files,
handling complex table structures, merged cells, and preserving table semantics.
"""

import os
import re
import json
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TableExtractionMethod(Enum):
    """Available table extraction methods"""
    PYMUPDF_TEXT = "pymupdf_text"
    PYMUPDF_TABLE = "pymupdf_table"
    OCR_BASED = "ocr_based"
    HYBRID = "hybrid"

class TableComplexity(Enum):
    """Table complexity classification"""
    SIMPLE = "simple"  # Clear borders, regular structure
    COMPLEX = "complex"  # Merged cells, irregular structure
    VERY_COMPLEX = "very_complex"  # Multi-level headers, nested tables
    SCANNED = "scanned"  # Image-based tables requiring OCR

@dataclass
class TableCell:
    """Represents a single table cell"""
    row: int
    col: int
    content: str
    rowspan: int = 1
    colspan: int = 1
    bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 1.0

@dataclass
class TableStructure:
    """Complete table structure with metadata"""
    headers: List[str]
    rows: List[List[TableCell]]
    caption: Optional[str]
    page_number: int
    bbox: Tuple[float, float, float, float]
    complexity: TableComplexity
    extraction_method: TableExtractionMethod
    confidence_score: float

class PDFTableProcessor:
    """
    Advanced PDF table processor that combines multiple extraction methods
    for optimal table structure preservation
    """
    
    def __init__(self):
        self.min_table_area = 1000  # Minimum area to consider as table
        self.max_cell_gap = 20      # Maximum gap between cells
        self.min_rows = 2           # Minimum rows to be considered a table
        self.min_cols = 2           # Minimum columns to be considered a table
        
    def extract_tables_from_pdf(self, pdf_path: str, pages: Optional[List[int]] = None) -> List[TableStructure]:
        """
        Extract tables from PDF using multiple methods and combine results
        """
        logger.info(f"Extracting tables from PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Open PDF document
        doc = fitz.open(pdf_path)
        
        tables = []
        target_pages = pages if pages else range(len(doc))
        
        for page_num in target_pages:
            page = doc[page_num]
            logger.info(f"Processing page {page_num + 1}")
            
            # Extract tables using multiple methods
            page_tables = self._extract_tables_from_page(page, page_num)
            tables.extend(page_tables)
        
        doc.close()
        
        logger.info(f"Extracted {len(tables)} tables from PDF")
        return tables
    
    def _extract_tables_from_page(self, page: fitz.Page, page_num: int) -> List[TableStructure]:
        """Extract tables from a single PDF page using multiple methods"""
        tables = []
        
        # Method 1: PyMuPDF built-in table extraction
        try:
            pymupdf_tables = self._extract_with_pymupdf_tables(page, page_num)
            tables.extend(pymupdf_tables)
        except Exception as e:
            logger.warning(f"PyMuPDF table extraction failed: {e}")
        
        # Method 2: Text-based table detection
        try:
            text_tables = self._extract_with_text_analysis(page, page_num)
            tables.extend(text_tables)
        except Exception as e:
            logger.warning(f"Text-based table extraction failed: {e}")
        
        # Remove duplicate tables
        tables = self._deduplicate_tables(tables)
        
        return tables
    
    def _extract_with_pymupdf_tables(self, page: fitz.Page, page_num: int) -> List[TableStructure]:
        """Extract tables using PyMuPDF's built-in table detection"""
        tables = []
        
        # Get text blocks with detailed information
        text_blocks = page.get_text("dict")
        
        # Detect table-like structures from text blocks
        table_candidates = self._detect_table_candidates(text_blocks)
        
        for candidate in table_candidates:
            try:
                table_structure = self._build_table_from_blocks(candidate, page_num)
                if table_structure and len(table_structure.rows) >= self.min_rows:
                    tables.append(table_structure)
            except Exception as e:
                logger.warning(f"Failed to build table from candidate: {e}")
                continue
        
        return tables
    
    def _detect_table_candidates(self, text_blocks: Dict) -> List[Dict]:
        """Detect potential table structures from text blocks"""
        candidates = []
        
        # Group text blocks by alignment and proximity
        aligned_blocks = self._group_aligned_blocks(text_blocks["blocks"])
        
        for group in aligned_blocks:
            if self._is_likely_table(group):
                candidates.append({
                    "blocks": group,
                    "type": "aligned_group"
                })
        
        return candidates
    
    def _group_aligned_blocks(self, blocks: List[Dict]) -> List[List[Dict]]:
        """Group text blocks by alignment patterns"""
        if not blocks:
            return []
        
        # Sort blocks by vertical position
        sorted_blocks = sorted(blocks, key=lambda b: b["bbox"][1])
        
        groups = []
        current_group = [sorted_blocks[0]]
        
        for block in sorted_blocks[1:]:
            last_block = current_group[-1]
            
            # Check if blocks are aligned horizontally and close vertically
            if (self._are_blocks_aligned(last_block, block) and 
                self._are_blocks_close(last_block, block)):
                current_group.append(block)
            else:
                if len(current_group) > 1:
                    groups.append(current_group)
                current_group = [block]
        
        if len(current_group) > 1:
            groups.append(current_group)
        
        return groups
    
    def _are_blocks_aligned(self, block1: Dict, block2: Dict) -> bool:
        """Check if two blocks are horizontally aligned"""
        bbox1 = block1["bbox"]
        bbox2 = block2["bbox"]
        
        # Check if blocks are on similar horizontal lines
        y1_center = (bbox1[1] + bbox1[3]) / 2
        y2_center = (bbox2[1] + bbox2[3]) / 2
        
        return abs(y1_center - y2_center) < 10
    
    def _are_blocks_close(self, block1: Dict, block2: Dict) -> bool:
        """Check if two blocks are close enough to be in the same table"""
        bbox1 = block1["bbox"]
        bbox2 = block2["bbox"]
        
        # Vertical gap between blocks
        vertical_gap = bbox2[1] - bbox1[3]
        return vertical_gap < self.max_cell_gap
    
    def _is_likely_table(self, blocks: List[Dict]) -> bool:
        """Determine if a group of blocks is likely a table"""
        if len(blocks) < self.min_rows:
            return False
        
        # Check for regular column structure
        x_positions = []
        for block in blocks:
            bbox = block["bbox"]
            x_positions.append(bbox[0])  # Left edge
        
        # Count unique x-positions to estimate columns
        unique_x = len(set(round(x) for x in x_positions))
        
        return unique_x >= self.min_cols
    
    def _build_table_from_blocks(self, candidate: Dict, page_num: int) -> Optional[TableStructure]:
        """Build table structure from grouped blocks"""
        blocks = candidate["blocks"]
        
        # Create grid structure
        grid = self._create_grid_from_blocks(blocks)
        if not grid:
            return None
        
        # Extract headers (first row)
        headers = []
        if grid:
            first_row = grid[0]
            headers = [cell.content for cell in first_row]
        
        # Build table structure
        table_bbox = self._calculate_table_bbox(blocks)
        
        return TableStructure(
            headers=headers,
            rows=grid,
            caption=self._extract_table_caption(blocks),
            page_number=page_num,
            bbox=table_bbox,
            complexity=self._assess_table_complexity(grid),
            extraction_method=TableExtractionMethod.PYMUPDF_TABLE,
            confidence_score=self._calculate_confidence(grid)
        )
    
    def _create_grid_from_blocks(self, blocks: List[Dict]) -> List[List[TableCell]]:
        """Create a grid structure from text blocks"""
        # Sort blocks into rows and columns
        rows = self._group_into_rows(blocks)
        
        grid = []
        for row_idx, row_blocks in enumerate(rows):
            row_cells = []
            sorted_blocks = sorted(row_blocks, key=lambda b: b["bbox"][0])
            
            for col_idx, block in enumerate(sorted_blocks):
                cell = TableCell(
                    row=row_idx,
                    col=col_idx,
                    content=block.get("text", "").strip(),
                    bbox=block["bbox"]
                )
                row_cells.append(cell)
            
            grid.append(row_cells)
        
        return grid
    
    def _group_into_rows(self, blocks: List[Dict]) -> List[List[Dict]]:
        """Group blocks into rows based on vertical alignment"""
        if not blocks:
            return []
        
        sorted_blocks = sorted(blocks, key=lambda b: b["bbox"][1])
        rows = []
        current_row = [sorted_blocks[0]]
        
        for block in sorted_blocks[1:]:
            last_block = current_row[-1]
            
            if self._are_blocks_in_same_row(last_block, block):
                current_row.append(block)
            else:
                rows.append(current_row)
                current_row = [block]
        
        rows.append(current_row)
        return rows
    
    def _are_blocks_in_same_row(self, block1: Dict, block2: Dict) -> bool:
        """Check if two blocks are in the same row"""
        bbox1 = block1["bbox"]
        bbox2 = block2["bbox"]
        
        # Blocks are in same row if their vertical positions overlap significantly
        y1_top, y1_bottom = bbox1[1], bbox1[3]
        y2_top, y2_bottom = bbox2[1], bbox2[3]
        
        overlap = min(y1_bottom, y2_bottom) - max(y1_top, y2_top)
        height1 = y1_bottom - y1_top
        height2 = y2_bottom - y2_top
        
        return overlap > 0.5 * min(height1, height2)
    
    def _calculate_table_bbox(self, blocks: List[Dict]) -> Tuple[float, float, float, float]:
        """Calculate bounding box for the entire table"""
        if not blocks:
            return (0, 0, 0, 0)
        
        x0 = min(block["bbox"][0] for block in blocks)
        y0 = min(block["bbox"][1] for block in blocks)
        x1 = max(block["bbox"][2] for block in blocks)
        y1 = max(block["bbox"][3] for block in blocks)
        
        return (x0, y0, x1, y1)
    
    def _extract_table_caption(self, blocks: List[Dict]) -> Optional[str]:
        """Extract table caption from surrounding text"""
        # Look for text above the table with common caption patterns
        table_bbox = self._calculate_table_bbox(blocks)
        
        # This would require access to the full page text
        # For now, return None - can be enhanced with page context
        return None
    
    def _assess_table_complexity(self, grid: List[List[TableCell]]) -> TableComplexity:
        """Assess the complexity of the table structure"""
        if not grid:
            return TableComplexity.SIMPLE
        
        # Check for merged cells
        has_merged_cells = False
        total_cells = sum(len(row) for row in grid)
        
        if len(grid) > 10:
            return TableComplexity.VERY_COMPLEX
        
        # Simple heuristic for complexity
        if len(grid) > 5 and any(len(row) > 5 for row in grid):
            return TableComplexity.COMPLEX
        
        return TableComplexity.SIMPLE
    
    def _calculate_confidence(self, grid: List[List[TableCell]]) -> float:
        """Calculate confidence score for table extraction"""
        if not grid:
            return 0.0
        
        # Base confidence on structure regularity
        row_lengths = [len(row) for row in grid]
        avg_length = sum(row_lengths) / len(row_lengths)
        
        # Check consistency
        consistent = all(abs(len(row) - avg_length) <= 1 for row in grid)
        
        if consistent and len(grid) >= 2:
            return 0.9
        elif len(grid) >= 2:
            return 0.7
        else:
            return 0.5
    
    def _extract_with_text_analysis(self, page: fitz.Page, page_num: int) -> List[TableStructure]:
        """Extract tables using text pattern analysis"""
        # Extract raw text and look for table patterns
        raw_text = page.get_text()
        
        tables = []
        
        # Look for markdown-style table patterns
        markdown_tables = self._extract_markdown_tables(raw_text, page_num)
        tables.extend(markdown_tables)
        
        return tables
    
    def _extract_markdown_tables(self, text: str, page_num: int) -> List[TableStructure]:
        """Extract markdown-style tables from text"""
        tables = []
        
        # Pattern for markdown tables
        pattern = r'(\|.*\|[\r\n]*)+'
        
        for match in re.finditer(pattern, text):
            table_text = match.group()
            lines = table_text.strip().split('\n')
            
            if len(lines) >= 3:  # Header, separator, and data
                try:
                    table_structure = self._parse_markdown_table(lines, page_num, match.start())
                    tables.append(table_structure)
                except Exception as e:
                    logger.warning(f"Failed to parse markdown table: {e}")
        
        return tables
    
    def _parse_markdown_table(self, lines: List[str], page_num: int, position: int) -> TableStructure:
        """Parse markdown table into structured format"""
        headers = self._parse_markdown_row(lines[0])
        separator = lines[1]
        
        rows = []
        for i, line in enumerate(lines[2:]):
            if line.strip():
                row_cells = self._parse_markdown_row(line)
                cells = [
                    TableCell(row=i, col=j, content=cell_content)
                    for j, cell_content in enumerate(row_cells)
                ]
                rows.append(cells)
        
        return TableStructure(
            headers=headers,
            rows=rows,
            caption=None,
            page_number=page_num,
            bbox=(0, 0, 0, 0),  # Position-based bbox
            complexity=TableComplexity.SIMPLE,
            extraction_method=TableExtractionMethod.PYMUPDF_TEXT,
            confidence_score=0.8
        )
    
    def _parse_markdown_row(self, line: str) -> List[str]:
        """Parse a single markdown table row"""
        # Remove leading/trailing pipes and split
        clean_line = line.strip().strip('|')
        return [cell.strip() for cell in clean_line.split('|')]
    
    def _deduplicate_tables(self, tables: List[TableStructure]) -> List[TableStructure]:
        """Remove duplicate and overlapping tables"""
        if not tables:
            return []
        
        # Sort by confidence score (highest first)
        sorted_tables = sorted(tables, key=lambda t: t.confidence_score, reverse=True)
        
        unique_tables = []
        used_areas = []
        
        for table in sorted_tables:
            # Check if this table overlaps significantly with already used tables
            is_duplicate = False
            for used_area in used_areas:
                if self._calculate_overlap(table.bbox, used_area) > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tables.append(table)
                used_areas.append(table.bbox)
        
        return unique_tables
    
    def _calculate_overlap(self, bbox1: Tuple[float, float, float, float], 
                          bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        
        intersection_area = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        
        # Return overlap ratio (intersection over union)
        return intersection_area / min(area1, area2)
    
    def export_table_to_markdown(self, table: TableStructure) -> str:
        """Export table structure to markdown format"""
        if not table.rows:
            return ""
        
        # Create header row
        header_row = "| " + " | ".join(table.headers) + " |"
        separator_row = "|" + "|".join([" --- "] * len(table.headers)) + "|"
        
        # Create data rows
        data_rows = []
        for row in table.rows:
            row_content = "| " + " | ".join(cell.content for cell in row) + " |"
            data_rows.append(row_content)
        
        # Add caption if available
        caption = f"\n*Table: {table.caption}*" if table.caption else ""
        
        return f"{header_row}\n{separator_row}\n" + "\n".join(data_rows) + caption
    
    def export_table_to_dataframe(self, table: TableStructure) -> pd.DataFrame:
        """Export table structure to pandas DataFrame"""
        if not table.rows:
            return pd.DataFrame()
        
        # Extract data
        data = []
        for row in table.rows:
            row_data = [cell.content for cell in row]
            data.append(row_data)
        
        return pd.DataFrame(data, columns=table.headers)
    
    def validate_table_structure(self, table: TableStructure) -> Dict[str, Any]:
        """Validate table structure and return quality metrics"""
        metrics = {
            "row_count": len(table.rows),
            "column_count": len(table.headers) if table.headers else 0,
            "total_cells": sum(len(row) for row in table.rows),
            "empty_cells": 0,
            "confidence_score": table.confidence_score,
            "complexity": table.complexity.value,
            "issues": []
        }
        
        # Check for empty cells
        for row in table.rows:
            for cell in row:
                if not cell.content.strip():
                    metrics["empty_cells"] += 1
        
        # Check structure consistency
        if table.rows:
            expected_cols = len(table.rows[0])
            for i, row in enumerate(table.rows):
                if len(row) != expected_cols:
                    metrics["issues"].append(f"Row {i} has inconsistent column count")
        
        # Calculate quality score
        if metrics["total_cells"] > 0:
            metrics["empty_cell_ratio"] = metrics["empty_cells"] / metrics["total_cells"]
            metrics["quality_score"] = max(0, 1 - metrics["empty_cell_ratio"] - 0.1 * len(metrics["issues"]))
        else:
            metrics["empty_cell_ratio"] = 0
            metrics["quality_score"] = 0
        
        return metrics
