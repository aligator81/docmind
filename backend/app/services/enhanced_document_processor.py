"""
Enhanced Document Processor with Advanced Table Extraction

This service extends the existing document processor with specialized table extraction
capabilities for complex PDF tables, ensuring correct data extraction and structure preservation.
"""

import os
import re
import json
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .document_processor import DocumentProcessor, ProcessingResult
from .pdf_table_processor import PDFTableProcessor, TableStructure, TableCell

logger = logging.getLogger(__name__)

@dataclass
class EnhancedProcessingResult:
    """Enhanced processing result with table extraction metadata"""
    success: bool
    content: str
    method: str
    processing_time: float
    metadata: Dict = None
    tables_extracted: int = 0
    table_quality_score: float = 0.0

class EnhancedDocumentProcessor(DocumentProcessor):
    """
    Enhanced document processor with advanced table extraction capabilities
    """
    
    def __init__(self):
        super().__init__()
        self.table_processor = PDFTableProcessor()
        self.table_extraction_enabled = True
        
    async def extract_document_with_tables(self, file_path: str, 
                                         extract_tables: bool = True,
                                         prefer_cloud: bool = False,
                                         use_cache: bool = True,
                                         original_filename: Optional[str] = None) -> EnhancedProcessingResult:
        """
        Extract document with enhanced table processing capabilities
        """
        logger.info(f"🔄 Enhanced processing with table extraction: {file_path}")
        
        # First, extract document content using the base processor
        base_result = await self.extract_document(
            file_path, prefer_cloud, use_cache, original_filename
        )
        
        if not base_result.success:
            return EnhancedProcessingResult(
                success=False,
                content="",
                method=base_result.method,
                processing_time=base_result.processing_time,
                metadata=base_result.metadata,
                tables_extracted=0,
                table_quality_score=0.0
            )
        
        # Extract tables if enabled and file is PDF
        tables = []
        table_processing_time = 0.0
        
        if extract_tables and self.table_extraction_enabled:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                logger.info("📊 Extracting tables from PDF...")
                table_start_time = time.time()
                
                try:
                    tables = self.table_processor.extract_tables_from_pdf(file_path)
                    table_processing_time = time.time() - table_start_time
                    
                    logger.info(f"✅ Extracted {len(tables)} tables in {table_processing_time:.2f}s")
                    
                except Exception as e:
                    logger.warning(f"❌ Table extraction failed: {e}")
                    tables = []
        
        # Enhance content with table structure
        enhanced_content = self._enhance_content_with_tables(base_result.content, tables)
        
        # Calculate table quality metrics
        table_quality_score = self._calculate_table_quality_score(tables)
        
        return EnhancedProcessingResult(
            success=True,
            content=enhanced_content,
            method=f"{base_result.method}_with_tables",
            processing_time=base_result.processing_time + table_processing_time,
            metadata={
                **base_result.metadata,
                "tables_extracted": len(tables),
                "table_quality_score": table_quality_score,
                "table_details": [self._serialize_table_metadata(table) for table in tables]
            },
            tables_extracted=len(tables),
            table_quality_score=table_quality_score
        )
    
    def _enhance_content_with_tables(self, base_content: str, tables: List[TableStructure]) -> str:
        """
        Enhance document content by preserving table structure in markdown format
        """
        if not tables:
            return base_content
        
        enhanced_content = base_content
        
        # For each table, replace or insert structured table representation
        for i, table in enumerate(tables):
            table_markdown = self._convert_table_to_markdown(table, i + 1)
            
            # Try to find the table position in the original content
            table_position = self._find_table_position(enhanced_content, table)
            
            if table_position >= 0:
                # Replace existing table with structured version
                enhanced_content = self._replace_table_content(enhanced_content, table_position, table_markdown)
            else:
                # Insert structured table at appropriate position
                enhanced_content = self._insert_table_content(enhanced_content, table, table_markdown)
        
        return enhanced_content
    
    def _convert_table_to_markdown(self, table: TableStructure, table_number: int) -> str:
        """
        Convert table structure to well-formatted markdown with metadata
        """
        markdown_table = self.table_processor.export_table_to_markdown(table)
        
        # Add table metadata and numbering
        table_metadata = f"\n\n<!-- TABLE_START: Table {table_number} -->\n"
        table_metadata += f"**Table {table_number}**"
        
        if table.caption:
            table_metadata += f": {table.caption}"
        
        table_metadata += f"\n*Extracted with {table.extraction_method.value} (confidence: {table.confidence_score:.2f})*\n\n"
        table_metadata += markdown_table
        table_metadata += f"\n<!-- TABLE_END: Table {table_number} -->\n"
        
        return table_metadata
    
    def _find_table_position(self, content: str, table: TableStructure) -> int:
        """
        Find the approximate position of the table in the content
        """
        # This is a simplified implementation
        # In practice, you'd use more sophisticated text matching
        
        # Look for table headers in the content
        if table.headers:
            for header in table.headers[:3]:  # Check first 3 headers
                if header and len(header) > 3:
                    header_position = content.find(header)
                    if header_position >= 0:
                        return header_position
        
        return -1
    
    def _replace_table_content(self, content: str, position: int, table_markdown: str) -> str:
        """
        Replace existing table content with structured markdown
        """
        # Find the end of the current table (next major section)
        section_end = self._find_section_end(content, position)
        
        if section_end > position:
            # Replace the table section
            return content[:position] + table_markdown + content[section_end:]
        else:
            # Just insert at position
            return content[:position] + table_markdown + content[position:]
    
    def _insert_table_content(self, content: str, table: TableStructure, table_markdown: str) -> str:
        """
        Insert structured table at appropriate position in content
        """
        # Insert based on page number and table position
        page_marker = f"--- Page {table.page_number + 1} ---"
        page_position = content.find(page_marker)
        
        if page_position >= 0:
            # Insert after page marker
            insert_position = page_position + len(page_marker)
            return content[:insert_position] + "\n\n" + table_markdown + content[insert_position:]
        else:
            # Append to end
            return content + "\n\n" + table_markdown
    
    def _find_section_end(self, content: str, start_position: int) -> int:
        """
        Find the end of the current section (table)
        """
        # Look for next major section marker
        section_markers = [
            "\n\n## ",
            "\n\n# ",
            "\n\n--- Page",
            "\n\n**Table",
            "\n\n<!-- TABLE"
        ]
        
        min_position = len(content)
        for marker in section_markers:
            marker_pos = content.find(marker, start_position + 1)
            if marker_pos >= 0 and marker_pos < min_position:
                min_position = marker_pos
        
        return min_position if min_position < len(content) else len(content)
    
    def _calculate_table_quality_score(self, tables: List[TableStructure]) -> float:
        """
        Calculate overall table extraction quality score
        """
        if not tables:
            return 0.0
        
        total_score = 0.0
        for table in tables:
            validation = self.table_processor.validate_table_structure(table)
            total_score += validation.get("quality_score", 0.0)
        
        return total_score / len(tables)
    
    def _serialize_table_metadata(self, table: TableStructure) -> Dict[str, Any]:
        """
        Serialize table metadata for storage
        """
        return {
            "page_number": table.page_number,
            "headers": table.headers,
            "row_count": len(table.rows),
            "column_count": len(table.headers) if table.headers else 0,
            "complexity": table.complexity.value,
            "extraction_method": table.extraction_method.value,
            "confidence_score": table.confidence_score,
            "caption": table.caption,
            "bbox": table.bbox
        }
    
    def get_table_extraction_stats(self, file_path: str) -> Dict[str, Any]:
        """
        Get detailed table extraction statistics for a file
        """
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension != '.pdf':
            return {"error": "Table extraction only supported for PDF files"}
        
        try:
            tables = self.table_processor.extract_tables_from_pdf(file_path)
            
            stats = {
                "total_tables": len(tables),
                "table_details": [],
                "quality_metrics": {
                    "average_confidence": 0.0,
                    "complexity_distribution": {},
                    "extraction_methods": {}
                }
            }
            
            total_confidence = 0.0
            complexity_counts = {}
            method_counts = {}
            
            for i, table in enumerate(tables):
                table_stats = self._serialize_table_metadata(table)
                table_stats["table_number"] = i + 1
                
                # Add validation metrics
                validation = self.table_processor.validate_table_structure(table)
                table_stats["validation"] = validation
                
                stats["table_details"].append(table_stats)
                
                # Aggregate statistics
                total_confidence += table.confidence_score
                
                complexity = table.complexity.value
                complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
                
                method = table.extraction_method.value
                method_counts[method] = method_counts.get(method, 0) + 1
            
            # Calculate averages
            if tables:
                stats["quality_metrics"]["average_confidence"] = total_confidence / len(tables)
                stats["quality_metrics"]["complexity_distribution"] = complexity_counts
                stats["quality_metrics"]["extraction_methods"] = method_counts
            
            return stats
            
        except Exception as e:
            return {"error": f"Table extraction failed: {str(e)}"}
    
    def export_tables_to_excel(self, file_path: str, output_path: Optional[str] = None) -> str:
        """
        Export extracted tables to Excel format
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension != '.pdf':
            raise ValueError("Table extraction only supported for PDF files")
        
        # Extract tables
        tables = self.table_processor.extract_tables_from_pdf(file_path)
        
        if not tables:
            raise ValueError("No tables found in PDF")
        
        # Generate output path if not provided
        if not output_path:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = f"{self.output_dir}/{base_name}_tables.xlsx"
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write each table to a separate sheet
            for i, table in enumerate(tables):
                sheet_name = f"Table_{i+1}"
                if len(sheet_name) > 31:  # Excel sheet name limit
                    sheet_name = f"T_{i+1}"
                
                df = self.table_processor.export_table_to_dataframe(table)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Create summary sheet
            summary_data = []
            for i, table in enumerate(tables):
                validation = self.table_processor.validate_table_structure(table)
                summary_data.append({
                    "Table": i + 1,
                    "Page": table.page_number + 1,
                    "Rows": len(table.rows),
                    "Columns": len(table.headers),
                    "Complexity": table.complexity.value,
                    "Method": table.extraction_method.value,
                    "Confidence": f"{table.confidence_score:.2f}",
                    "Quality": f"{validation.get('quality_score', 0):.2f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
        
        logger.info(f"✅ Tables exported to Excel: {output_path}")
        return output_path