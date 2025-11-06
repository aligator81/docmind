"""
Enhanced Chat Service with Table-Aware Retrieval

This service integrates table-aware retrieval to ensure accurate AI responses
for table-based queries.
"""

import logging
from typing import List, Dict, Any, Optional
import re

from .table_aware_retrieval import TableAwareRetrieval, TableQueryContext
from .hybrid_search_service import HybridSearchService

logger = logging.getLogger(__name__)

class EnhancedChatService:
    """
    Enhanced chat service that properly handles table-based queries
    """
    
    def __init__(self, search_service=None):
        self.table_retrieval = TableAwareRetrieval()
        self.search_service = search_service
        
        # Table-specific prompt templates
        self.table_prompt_templates = {
            "aggregation": """
When answering questions about numerical data from tables:

1. **Extract the relevant numbers** from the table cells
2. **Perform the requested calculation** (sum, average, max, min, etc.)
3. **Cite the specific table and rows** used for the calculation
4. **Show your reasoning** for how you arrived at the answer

Example format:
- "Based on Table 1, the sum of Revenue for Q1-Q4 is: [calculation]"
- "From the data in rows 2-5 of Table 2, the average is: [calculation]"
""",
            "comparison": """
When comparing data from tables:

1. **Identify the items being compared** (rows, columns, or time periods)
2. **Extract the relevant values** for comparison
3. **Calculate differences or ratios** as requested
4. **Provide clear comparison results** with proper context

Example format:
- "Comparing Q1 and Q4 in Table 1: Q1 was [value], Q4 was [value], difference: [calculation]"
- "Table 2 shows that [item A] is [X]% higher than [item B]"
""",
            "general_table": """
When answering questions about table data:

1. **Read the table structure carefully** - understand headers and data types
2. **Extract specific values** mentioned in the question
3. **Provide exact values** from the table cells when possible
4. **Reference the table and specific location** of the data

Important: Always verify you're reading the correct row and column before answering.
"""
        }
    
    def process_chat_query(self, query: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a chat query with enhanced table awareness
        """
        logger.info(f"🔍 Processing table-aware chat query: {query}")
        
        # Analyze the query for table requirements
        query_context = self.table_retrieval.analyze_table_query(query)
        logger.info(f"📊 Query analysis: {query_context}")
        
        # Perform enhanced retrieval
        search_results = self._perform_enhanced_retrieval(query, query_context, document_id)
        
        # Generate enhanced prompt
        enhanced_prompt = self._create_enhanced_prompt(query, query_context, search_results)
        
        # Prepare response with table context
        response_data = {
            "query": query,
            "query_context": {
                "table_columns": query_context.table_columns,
                "table_operations": query_context.table_operations,
                "requires_aggregation": query_context.requires_aggregation,
                "requires_comparison": query_context.requires_comparison,
                "target_table": query_context.target_table
            },
            "retrieved_chunks": len(search_results.get('chunks', [])),
            "table_chunks_count": len(search_results.get('table_chunks', [])),
            "enhanced_prompt": enhanced_prompt,
            "search_metadata": search_results.get('metadata', {})
        }
        
        return response_data
    
    def _perform_enhanced_retrieval(self, query: str, query_context: TableQueryContext, 
                                  document_id: Optional[str]) -> Dict[str, Any]:
        """
        Perform enhanced retrieval with table awareness
        """
        # Get initial search results
        if document_id:
            search_results = self.search_service.search_in_document(document_id, query)
        else:
            search_results = self.search_service.hybrid_search(query)
        
        chunks = search_results.get('chunks', [])
        
        if not chunks:
            return search_results
        
        # Enhance table chunks for better retrieval
        enhanced_chunks = self.table_retrieval.enhance_table_chunks_for_retrieval(chunks)
        
        # Prioritize chunks based on query requirements
        prioritized_chunks = self.table_retrieval.prioritize_table_chunks_for_query(
            enhanced_chunks, query_context
        )
        
        # Separate table and text chunks for analysis
        table_chunks = [chunk for chunk in prioritized_chunks 
                       if chunk.get('chunk_type', '').startswith('table_')]
        text_chunks = [chunk for chunk in prioritized_chunks 
                      if not chunk.get('chunk_type', '').startswith('table_')]
        
        # Update search results with enhanced chunks
        enhanced_results = search_results.copy()
        enhanced_results['chunks'] = prioritized_chunks
        enhanced_results['table_chunks'] = table_chunks
        enhanced_results['text_chunks'] = text_chunks
        
        # Add retrieval metadata
        enhanced_results['metadata'] = {
            'total_chunks': len(prioritized_chunks),
            'table_chunks': len(table_chunks),
            'text_chunks': len(text_chunks),
            'query_analysis': {
                'table_operations': query_context.table_operations,
                'requires_aggregation': query_context.requires_aggregation,
                'requires_comparison': query_context.requires_comparison
            }
        }
        
        return enhanced_results
    
    def _create_enhanced_prompt(self, query: str, query_context: TableQueryContext, 
                              search_results: Dict[str, Any]) -> str:
        """
        Create an enhanced prompt with table-specific instructions
        """
        chunks = search_results.get('chunks', [])
        table_chunks = search_results.get('table_chunks', [])
        
        # Base prompt
        prompt_parts = [
            "You are an AI assistant that helps answer questions based on document content.",
            "Pay special attention to tables and numerical data.",
            f"Question: {query}",
            ""
        ]
        
        # Add table-specific instructions
        if query_context.requires_aggregation:
            prompt_parts.append(self.table_prompt_templates["aggregation"])
        elif query_context.requires_comparison:
            prompt_parts.append(self.table_prompt_templates["comparison"])
        elif table_chunks:
            prompt_parts.append(self.table_prompt_templates["general_table"])
        
        prompt_parts.append("")
        
        # Add table context
        if table_chunks:
            table_context = self.table_retrieval.format_table_context_for_prompt(table_chunks)
            prompt_parts.append(table_context)
            prompt_parts.append("")
        
        # Add text context (non-table chunks)
        text_chunks = search_results.get('text_chunks', [])
        if text_chunks:
            prompt_parts.append("Additional relevant text content:")
            for i, chunk in enumerate(text_chunks[:2]):  # Limit to top 2 text chunks
                chunk_text = chunk.get('chunk_text', '')[:500]  # Limit length
                prompt_parts.append(f"\nText excerpt {i+1}: {chunk_text}")
            prompt_parts.append("")
        
        # Add final instructions
        prompt_parts.extend([
            "Instructions:",
            "1. Answer the question based ONLY on the provided content",
            "2. For table data, be precise and cite specific table locations",
            "3. If performing calculations, show your reasoning",
            "4. If information is not available in the provided content, say so",
            "5. Be accurate and avoid hallucinations",
            "",
            "Answer:"
        ])
        
        return "\n".join(prompt_parts)
    
    def validate_table_response(self, response: str, table_chunks: List[Dict]) -> Dict[str, Any]:
        """
        Validate that the AI response correctly uses table data
        """
        validation_result = {
            "table_references_found": False,
            "numerical_accuracy": "unknown",
            "calculation_verification": "not_applicable",
            "potential_issues": [],
            "confidence_score": 0.5
        }
        
        # Check for table references in response
        table_ref_patterns = [
            r'table\s+\d+', r'row\s+\d+', r'column\s+\w+',
            r'q[1-4]', r'quarter\s+\d', r'in\s+the\s+table'
        ]
        
        table_references = any(
            re.search(pattern, response.lower()) 
            for pattern in table_ref_patterns
        )
        validation_result["table_references_found"] = table_references
        
        # Check for numerical values that might need verification
        numerical_values = re.findall(r'\b\d+\.?\d*\b', response)
        if numerical_values:
            validation_result["numerical_accuracy"] = "needs_verification"
            
            # Check if these numbers appear in table chunks
            found_in_tables = False
            for chunk in table_chunks:
                chunk_text = chunk.get('chunk_text', '')
                for num in numerical_values[:3]:  # Check first 3 numbers
                    if num in chunk_text:
                        found_in_tables = True
                        break
                if found_in_tables:
                    break
            
            if found_in_tables:
                validation_result["numerical_accuracy"] = "verified_in_tables"
                validation_result["confidence_score"] += 0.3
            else:
                validation_result["potential_issues"].append(
                    "Numerical values in response not found in retrieved tables"
                )
        
        # Check for calculation indicators
        calculation_indicators = ['sum', 'total', 'average', 'difference', 'ratio']
        has_calculations = any(indicator in response.lower() for indicator in calculation_indicators)
        
        if has_calculations:
            validation_result["calculation_verification"] = "needs_verification"
            validation_result["potential_issues"].append(
                "Response contains calculations that should be verified against table data"
            )
        
        # Boost confidence for table references
        if table_references:
            validation_result["confidence_score"] += 0.2
        
        return validation_result
    
    def get_table_query_examples(self) -> List[Dict[str, str]]:
        """
        Provide examples of well-formed table queries
        """
        return [
            {
                "query": "What is the total revenue for Q1-Q4 in Table 1?",
                "explanation": "Specific table reference with clear aggregation request"
            },
            {
                "query": "Compare the revenue between North America and Europe in the regional performance table",
                "explanation": "Clear comparison request with specific regions"
            },
            {
                "query": "What was the highest profit quarter based on the financial table?",
                "explanation": "Maximum value request with clear data column"
            },
            {
                "query": "Show me the growth percentage from Q1 to Q4 for Product A",
                "explanation": "Percentage calculation with specific time period"
            }
        ]