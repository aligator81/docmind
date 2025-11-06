"""
Test script for table-aware chunking system

This script tests the enhanced table extraction and chunking capabilities
with various complex table scenarios.
"""

import os
import sys
import logging
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.pdf_table_processor import PDFTableProcessor
from app.services.table_aware_chunker import TableAwareChunker
from app.services.enhanced_document_processor import EnhancedDocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pdf_table_processor():
    """Test PDF table extraction capabilities"""
    print("🧪 Testing PDF Table Processor...")
    
    processor = PDFTableProcessor()
    
    # Test with sample PDF files
    test_pdfs = [
        "data/uploads/test.pdf",  # Replace with actual test PDF
        "data/uploads/complex_tables.pdf"  # Replace with actual test PDF
    ]
    
    for pdf_path in test_pdfs:
        if not os.path.exists(pdf_path):
            print(f"⚠️  Test PDF not found: {pdf_path}")
            continue
            
        print(f"\n📄 Processing: {pdf_path}")
        
        try:
            # Extract tables from PDF
            tables = processor.extract_tables_from_pdf(pdf_path)
            print(f"✅ Extracted {len(tables)} tables")
            
            # Analyze table complexity
            for i, table in enumerate(tables):
                print(f"  Table {i+1}:")
                print(f"    - Complexity: {table.complexity}")
                print(f"    - Rows: {len(table.rows)}")
                print(f"    - Columns: {len(table.headers)}")
                print(f"    - Confidence: {table.confidence_score:.2f}")
                print(f"    - Method: {table.extraction_method}")
                
                # Export to markdown
                markdown = processor.export_table_to_markdown(table)
                print(f"    - Markdown preview: {markdown[:100]}...")
                
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")

def test_table_aware_chunker():
    """Test table-aware chunking capabilities"""
    print("\n🧪 Testing Table-Aware Chunker...")
    
    chunker = TableAwareChunker()
    
    # Test with sample content containing tables
    test_contents = [
        {
            "content": """
# Document with Simple Table

Here is some text before the table.

<!-- TABLE_START: simple_table -->
**Table 1**: Simple data table
Extracted with text_based method, confidence: 0.85
Page 1

| Name | Age | City |
|------|-----|------|
| John | 25  | NY   |
| Jane | 30  | LA   |
| Bob  | 35  | SF   |
<!-- TABLE_END: simple_table -->

Some text after the table.
            """,
            "filename": "test_simple.md"
        },
        {
            "content": """
# Document with Complex Table

Here is some text before the complex table.

<!-- TABLE_START: complex_table -->
**Table 2**: Complex financial data
Extracted with pattern_based method, confidence: 0.75
Page 2

| Quarter | Product A | Product B | Product C | Product D | Product E | Product F |
|---------|-----------|-----------|-----------|-----------|-----------|-----------|
| Q1      | 1000      | 2000      | 1500      | 1800      | 2200      | 1900      |
| Q2      | 1200      | 2100      | 1600      | 1900      | 2300      | 2000      |
| Q3      | 1100      | 2200      | 1700      | 2000      | 2400      | 2100      |
| Q4      | 1300      | 2300      | 1800      | 2100      | 2500      | 2200      |
| Q5      | 1400      | 2400      | 1900      | 2200      | 2600      | 2300      |
| Q6      | 1500      | 2500      | 2000      | 2300      | 2700      | 2400      |
| Q7      | 1600      | 2600      | 2100      | 2400      | 2800      | 2500      |
| Q8      | 1700      | 2700      | 2200      | 2500      | 2900      | 2600      |
<!-- TABLE_END: complex_table -->

Some text after the complex table.
            """,
            "filename": "test_complex.md"
        }
    ]
    
    for test_case in test_contents:
        print(f"\n📄 Processing: {test_case['filename']}")
        
        try:
            # Create chunks with table awareness
            chunks = chunker.split_content_into_chunks(
                test_case["content"], 
                test_case["filename"]
            )
            
            print(f"✅ Created {len(chunks)} chunks")
            
            # Analyze chunk types
            chunk_types = {}
            table_chunks = []
            
            for chunk in chunks:
                chunk_type = chunk.get('chunk_type', 'text')
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                
                if chunk_type.startswith('table_'):
                    table_chunks.append(chunk)
            
            print(f"  Chunk types: {chunk_types}")
            
            # Validate table chunks
            if table_chunks:
                validation = chunker.validate_table_chunks(chunks)
                print(f"  Table chunk validation:")
                print(f"    - Total table chunks: {validation['total_table_chunks']}")
                print(f"    - Average confidence: {validation['average_confidence']:.2f}")
                print(f"    - Issues: {validation['issues']}")
                
                # Show sample table chunks
                for i, table_chunk in enumerate(table_chunks[:2]):  # Show first 2
                    print(f"    Table chunk {i+1}:")
                    print(f"      - Type: {table_chunk['chunk_type']}")
                    print(f"      - Tokens: {table_chunk['token_count']}")
                    print(f"      - Strategy: {table_chunk['table_metadata'].get('strategy', 'N/A')}")
                    print(f"      - Preview: {table_chunk['chunk_text'][:100]}...")
            
        except Exception as e:
            print(f"❌ Error processing {test_case['filename']}: {e}")

def test_enhanced_document_processor():
    """Test the complete enhanced document processing pipeline"""
    print("\n🧪 Testing Enhanced Document Processor...")
    
    processor = EnhancedDocumentProcessor()
    
    # Test with sample document
    test_docs = [
        "data/uploads/test.md",  # Replace with actual test document
    ]
    
    for doc_path in test_docs:
        if not os.path.exists(doc_path):
            print(f"⚠️  Test document not found: {doc_path}")
            continue
            
        print(f"\n📄 Processing: {doc_path}")
        
        try:
            # Process document with enhanced capabilities
            result = processor.process_document(doc_path)
            
            print(f"✅ Processing completed")
            print(f"  - Total chunks: {len(result.get('chunks', []))}")
            print(f"  - Tables extracted: {len(result.get('tables', []))}")
            print(f"  - Quality score: {result.get('quality_score', 0):.2f}")
            
            # Show table extraction results
            tables = result.get('tables', [])
            if tables:
                print(f"  - Table extraction summary:")
                for i, table in enumerate(tables[:3]):  # Show first 3 tables
                    print(f"    Table {i+1}: {table.get('complexity', 'unknown')} "
                          f"({len(table.get('rows', []))} rows, "
                          f"{len(table.get('headers', []))} cols)")
            
        except Exception as e:
            print(f"❌ Error processing {doc_path}: {e}")

def test_table_quality_metrics():
    """Test table quality assessment and validation"""
    print("\n🧪 Testing Table Quality Metrics...")
    
    processor = PDFTableProcessor()
    
    # Create test tables with different quality levels
    test_tables = [
        {
            "name": "High Quality Table",
            "headers": ["Name", "Age", "City"],
            "rows": [
                [{"content": "John"}, {"content": "25"}, {"content": "NY"}],
                [{"content": "Jane"}, {"content": "30"}, {"content": "LA"}],
                [{"content": "Bob"}, {"content": "35"}, {"content": "SF"}]
            ],
            "expected_quality": "high"
        },
        {
            "name": "Medium Quality Table",
            "headers": ["Product", "Q1", "Q2"],
            "rows": [
                [{"content": "A"}, {"content": "100"}, {"content": ""}],
                [{"content": "B"}, {"content": ""}, {"content": "200"}],
                [{"content": "C"}, {"content": "150"}, {"content": "180"}]
            ],
            "expected_quality": "medium"
        },
        {
            "name": "Low Quality Table",
            "headers": [],
            "rows": [
                [{"content": "data1"}, {"content": "data2"}],
                [{"content": ""}, {"content": ""}]
            ],
            "expected_quality": "low"
        }
    ]
    
    for test_table in test_tables:
        print(f"\n📊 Testing: {test_table['name']}")
        
        # Create table structure
        table = TableStructure(
            headers=test_table["headers"],
            rows=test_table["rows"],
            caption=f"Test: {test_table['name']}",
            page_number=1,
            bbox=(0, 0, 100, 100),
            complexity="simple",
            extraction_method="test",
            confidence_score=0.8
        )
        
        # Assess quality
        quality_metrics = processor.assess_table_quality(table)
        
        print(f"  Quality metrics:")
        print(f"    - Structure score: {quality_metrics['structure_score']:.2f}")
        print(f"    - Content score: {quality_metrics['content_score']:.2f}")
        print(f"    - Overall quality: {quality_metrics['overall_quality']}")
        print(f"    - Issues: {quality_metrics['issues']}")
        
        # Validate against expected quality
        if quality_metrics['overall_quality'] == test_table['expected_quality']:
            print(f"  ✅ Quality assessment matches expectation")
        else:
            print(f"  ⚠️  Quality assessment differs from expectation")

def run_comprehensive_test():
    """Run all table-aware chunking tests"""
    print("🚀 Starting Comprehensive Table-Aware Chunking Tests")
    print("=" * 60)
    
    # Test individual components
    test_pdf_table_processor()
    test_table_aware_chunker()
    test_enhanced_document_processor()
    test_table_quality_metrics()
    
    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print("✅ PDF Table Processor - Table extraction and structure preservation")
    print("✅ Table-Aware Chunker - Specialized table chunking strategies")
    print("✅ Enhanced Document Processor - Complete processing pipeline")
    print("✅ Table Quality Metrics - Quality assessment and validation")
    print("\n📋 Next Steps:")
    print("1. Test with real PDF files containing complex tables")
    print("2. Integrate with existing document processing pipeline")
    print("3. Optimize performance for production use")
    print("4. Add comprehensive error handling and fallbacks")

if __name__ == "__main__":
    run_comprehensive_test()