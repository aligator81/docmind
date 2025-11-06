"""
Table-Aware Retrieval Service

This service enhances the retrieval process to properly handle table chunks
and ensure accurate AI responses for table-based queries.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TableQueryContext:
    """Context for table-aware query processing"""
    query: str
    table_columns: List[str]
    table_operations: List[str]
    requires_aggregation: bool
    requires_comparison: bool
    target_table: Optional[str]

class TableAwareRetrieval:
    """
    Enhanced retrieval service that understands table structure
    and optimizes for table-based queries
    """
    
    def __init__(self):
        # Table operation patterns
        self.aggregation_patterns = [
            r'sum\s+of', r'total\s+', r'average\s+', r'avg\s+', r'max\s+', 
            r'min\s+', r'count\s+', r'highest\s+', r'lowest\s+', r'most\s+',
            r'least\s+', r'greater\s+than', r'less\s+than', r'between\s+'
        ]
        
        self.comparison_patterns = [
            r'compare\s+', r'vs\s+', r'versus\s+', r'difference\s+between',
            r'change\s+from', r'growth\s+', r'decline\s+', r'increase\s+',
            r'decrease\s+', r'better\s+than', r'worse\s+than'
        ]
        
        self.table_column_patterns = [
            r'column\s+(\w+)', r'(\w+)\s+column', r'in\s+(\w+)', 
            r'for\s+(\w+)', r'(\w+)\s+data', r'(\w+)\s+values'
        ]
    
    def analyze_table_query(self, query: str) -> TableQueryContext:
        """
        Analyze query to understand table-related requirements
        """
        query_lower = query.lower()
        
        # Check for aggregation operations
        requires_aggregation = any(
            re.search(pattern, query_lower) 
            for pattern in self.aggregation_patterns
        )
        
        # Check for comparison operations
        requires_comparison = any(
            re.search(pattern, query_lower)
            for pattern in self.comparison_patterns
        )
        
        # Extract potential table columns
        table_columns = []
        for pattern in self.table_column_patterns:
            matches = re.findall(pattern, query_lower)
            table_columns.extend(matches)
        
        # Extract table operations
        table_operations = []
        if requires_aggregation:
            table_operations.append("aggregation")
        if requires_comparison:
            table_operations.append("comparison")
        
        # Identify target table if mentioned
        target_table = None
        table_mentions = re.findall(r'table\s+(\w+)', query_lower)
        if table_mentions:
            target_table = table_mentions[0]
        
        return TableQueryContext(
            query=query,
            table_columns=table_columns,
            table_operations=table_operations,
            requires_aggregation=requires_aggregation,
            requires_comparison=requires_comparison,
            target_table=target_table
        )
    
    def enhance_table_chunks_for_retrieval(self, chunks: List[Dict]) -> List[Dict]:
        """
        Enhance table chunks with additional metadata for better retrieval
        """
        enhanced_chunks = []
        
        for chunk in chunks:
            if chunk.get('chunk_type', '').startswith('table_'):
                enhanced_chunk = self._enhance_single_table_chunk(chunk)
                enhanced_chunks.append(enhanced_chunk)
            else:
                enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def _enhance_single_table_chunk(self, chunk: Dict) -> Dict:
        """
        Enhance a single table chunk with retrieval metadata
        """
        chunk_text = chunk.get('chunk_text', '')
        table_metadata = chunk.get('table_metadata', {})
        
        # Extract table structure information
        headers = self._extract_table_headers(chunk_text)
        data_types = self._infer_data_types(chunk_text)
        numerical_columns = self._identify_numerical_columns(chunk_text)
        
        # Create enhanced metadata
        enhanced_metadata = {
            **table_metadata,
            'table_headers': headers,
            'data_types': data_types,
            'numerical_columns': numerical_columns,
            'aggregation_capable': len(numerical_columns) > 0,
            'comparison_capable': len(headers) > 1,
            'retrieval_boost': self._calculate_retrieval_boost(chunk)
        }
        
        # Add table summary for better retrieval
        summary = self._create_table_summary(chunk_text, headers, numerical_columns)
        
        enhanced_chunk = chunk.copy()
        enhanced_chunk['table_metadata'] = enhanced_metadata
        enhanced_chunk['enhanced_text'] = f"{summary}\n\n{chunk_text}"
        
        return enhanced_chunk
    
    def _extract_table_headers(self, table_text: str) -> List[str]:
        """Extract table headers from markdown table"""
        headers = []
        
        # Look for header row in markdown table
        lines = table_text.split('\n')
        for line in lines:
            if line.strip().startswith('|') and '---' not in line:
                # Extract header cells
                cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
                if cells and not headers:  # First valid row is headers
                    headers = cells
                    break
        
        return headers
    
    def _infer_data_types(self, table_text: str) -> Dict[str, str]:
        """Infer data types for table columns"""
        data_types = {}
        headers = self._extract_table_headers(table_text)
        
        if not headers:
            return data_types
        
        # Extract sample data rows
        lines = table_text.split('\n')
        data_rows = []
        
        for line in lines:
            if line.strip().startswith('|') and '---' not in line:
                cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
                if cells and cells != headers:  # Skip header row
                    data_rows.append(cells)
        
        # Analyze each column
        for i, header in enumerate(headers):
            column_values = [row[i] for row in data_rows if i < len(row)]
            
            if not column_values:
                data_types[header] = 'unknown'
                continue
            
            # Check data type patterns
            sample_value = column_values[0] if column_values else ''
            
            # Check for numerical data
            if any(self._is_numerical(val) for val in column_values[:3]):
                data_types[header] = 'numerical'
            # Check for percentage
            elif any('%' in val for val in column_values[:3]):
                data_types[header] = 'percentage'
            # Check for currency
            elif any('$' in val or 'USD' in val.upper() for val in column_values[:3]):
                data_types[header] = 'currency'
            # Check for date
            elif any(self._is_date_like(val) for val in column_values[:3]):
                data_types[header] = 'date'
            else:
                data_types[header] = 'text'
        
        return data_types
    
    def _identify_numerical_columns(self, table_text: str) -> List[str]:
        """Identify columns that contain numerical data"""
        data_types = self._infer_data_types(table_text)
        return [header for header, dtype in data_types.items() 
                if dtype in ['numerical', 'percentage', 'currency']]
    
    def _is_numerical(self, value: str) -> bool:
        """Check if a value is numerical"""
        if not value or value.strip() == '':
            return False
        
        # Remove commas and currency symbols
        clean_value = re.sub(r'[,$%]', '', value.strip())
        
        # Check for numbers with optional decimal points
        return bool(re.match(r'^-?\d+(\.\d+)?$', clean_value))
    
    def _is_date_like(self, value: str) -> bool:
        """Check if a value looks like a date"""
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{4}-\d{2}-\d{2}',
            r'[A-Za-z]{3}\s+\d{1,2},?\s+\d{4}',
            r'Q[1-4]\s+\d{4}'
        ]
        return any(re.search(pattern, value) for pattern in date_patterns)
    
    def _calculate_retrieval_boost(self, chunk: Dict) -> float:
        """Calculate retrieval boost score for table chunks"""
        boost = 1.0
        
        table_metadata = chunk.get('table_metadata', {})
        
        # Boost for complete tables
        if chunk.get('chunk_type') == 'table_complete':
            boost *= 1.2
        
        # Boost for high confidence tables
        confidence = table_metadata.get('confidence', 0.5)
        boost *= (0.5 + confidence)  # Range: 0.75 to 1.5
        
        # Boost for tables with numerical data
        if table_metadata.get('aggregation_capable', False):
            boost *= 1.3
        
        return min(boost, 2.0)  # Cap at 2.0
    
    def _create_table_summary(self, table_text: str, headers: List[str], 
                            numerical_columns: List[str]) -> str:
        """Create a summary of the table for better retrieval"""
        summary_parts = []
        
        if headers:
            summary_parts.append(f"Table with columns: {', '.join(headers)}")
        
        if numerical_columns:
            summary_parts.append(f"Numerical data in: {', '.join(numerical_columns)}")
        
        # Add table size information
        lines = table_text.split('\n')
        data_rows = [line for line in lines if line.strip().startswith('|') and '---' not in line]
        if len(data_rows) > 1:  # More than just headers
            summary_parts.append(f"Contains {len(data_rows)-1} data rows")
        
        return ". ".join(summary_parts)
    
    def prioritize_table_chunks_for_query(self, chunks: List[Dict], 
                                        query_context: TableQueryContext) -> List[Dict]:
        """
        Prioritize table chunks based on query requirements
        """
        prioritized_chunks = []
        
        for chunk in chunks:
            if not chunk.get('chunk_type', '').startswith('table_'):
                # Non-table chunks get base priority
                prioritized_chunks.append((chunk, 1.0))
                continue
            
            table_metadata = chunk.get('table_metadata', {})
            priority_score = 1.0
            
            # Boost for aggregation queries
            if query_context.requires_aggregation:
                if table_metadata.get('aggregation_capable', False):
                    priority_score *= 1.5
            
            # Boost for comparison queries  
            if query_context.requires_comparison:
                if table_metadata.get('comparison_capable', False):
                    priority_score *= 1.3
            
            # Boost for matching columns
            matching_columns = set(query_context.table_columns) & set(table_metadata.get('table_headers', []))
            if matching_columns:
                priority_score *= (1.0 + 0.1 * len(matching_columns))
            
            # Apply retrieval boost
            priority_score *= table_metadata.get('retrieval_boost', 1.0)
            
            prioritized_chunks.append((chunk, priority_score))
        
        # Sort by priority score (descending)
        prioritized_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, score in prioritized_chunks]
    
    def format_table_context_for_prompt(self, table_chunks: List[Dict]) -> str:
        """
        Format table chunks for inclusion in AI prompt
        """
        if not table_chunks:
            return ""
        
        context_parts = ["Here are relevant tables from the document:"]
        
        for i, chunk in enumerate(table_chunks[:3]):  # Limit to top 3 tables
            chunk_text = chunk.get('chunk_text', '')
            table_metadata = chunk.get('table_metadata', {})
            
            context_parts.append(f"\nTable {i+1}:")
            
            # Add table summary
            if 'enhanced_text' in chunk:
                summary_match = re.search(r'^(.*?)\n\n', chunk['enhanced_text'])
                if summary_match:
                    context_parts.append(f"Summary: {summary_match.group(1)}")
            
            # Add the actual table
            context_parts.append(chunk_text)
            
            # Add metadata if available
            if table_metadata.get('confidence', 0) < 0.7:
                context_parts.append(f"Note: This table has lower confidence ({table_metadata['confidence']:.2f})")
        
        return "\n".join(context_parts)