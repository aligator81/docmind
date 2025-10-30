#!/usr/bin/env python3
"""
Debug script to test chunking process step by step
"""

import os
import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.database import SessionLocal
from app.models import Document
from app.services.document_chunker import DocumentChunker
from app.services.document_processor import DocumentProcessor

async def debug_document_processing():
    """Debug document processing step by step"""
    print("🔍 Debugging Document Processing")
    print("=" * 60)
    
    db = SessionLocal()
    try:
        # Get the failed document
        failed_docs = db.query(Document).filter(Document.status == "failed").all()
        
        if not failed_docs:
            print("❌ No failed documents found")
            return
        
        doc = failed_docs[0]
        print(f"📄 Processing failed document: {doc.filename} (ID: {doc.id})")
        print(f"📊 Current status: {doc.status}")
        print(f"📝 File path: {doc.file_path}")
        print(f"📏 File exists: {os.path.exists(doc.file_path) if doc.file_path else 'No file path'}")
        print(f"📄 Content length: {len(doc.content) if doc.content else 0} characters")
        
        # Step 1: Check if file exists and is accessible
        if not doc.file_path or not os.path.exists(doc.file_path):
            print("❌ Document file not found or inaccessible")
            return
        
        # Step 2: Try to extract content again
        print("\n🔄 Step 1: Testing document extraction...")
        processor = DocumentProcessor()
        
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        
        extraction_result = await processor.extract_document(
            doc.file_path, 
            original_filename=doc.original_filename
        )
        
        print(f"📊 Extraction result: {extraction_result.success}")
        print(f"📝 Method used: {extraction_result.method}")
        print(f"⏱️ Processing time: {extraction_result.processing_time:.2f}s")
        print(f"📄 Content length extracted: {len(extraction_result.content) if extraction_result.success else 0}")
        
        if not extraction_result.success:
            print(f"❌ Extraction failed: {extraction_result.method}")
            return
        
        # Step 3: Update document with extracted content
        print("\n🔄 Step 2: Updating document with extracted content...")
        doc.content = extraction_result.content
        doc.status = "extracted"
        db.commit()
        print("✅ Document updated with extracted content")
        
        # Step 4: Test chunking
        print("\n🔄 Step 3: Testing chunking process...")
        chunker = DocumentChunker()
        
        # Test chunking with the extracted content
        print(f"📄 Testing chunking with {len(doc.content)} characters of content...")
        
        try:
            chunks = await chunker.chunk_document_content(doc.content, doc.filename)
            print(f"📦 Chunks created: {len(chunks)}")
            
            if chunks:
                print("📊 First chunk preview:")
                first_chunk = chunks[0]
                print(f"  - Text: {first_chunk['chunk_text'][:100]}...")
                print(f"  - Page numbers: {first_chunk['page_numbers']}")
                print(f"  - Section title: {first_chunk['section_title']}")
                print(f"  - Token count: {first_chunk['token_count']}")
                
                # Test saving chunks to database
                print("\n🔄 Step 4: Testing database chunk insertion...")
                chunking_result = await chunker.process_document_from_db(db, doc.id)
                print(f"📊 Database chunking result: {chunking_result.success}")
                print(f"📦 Chunks created in DB: {chunking_result.chunks_created}")
                print(f"⏱️ Processing time: {chunking_result.processing_time:.2f}s")
                
                if chunking_result.success:
                    print("✅ Chunking process completed successfully!")
                else:
                    print(f"❌ Database chunking failed: {chunking_result.metadata}")
            else:
                print("❌ No chunks were created from the content")
                print("💡 Possible issues:")
                print("   - Content might be too short or empty")
                print("   - Docling chunker might be misconfigured")
                print("   - Content format might not be supported")
                
        except Exception as e:
            print(f"❌ Chunking process failed with error: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Debug process failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

async def test_chunking_with_sample_content():
    """Test chunking with sample content to isolate the issue"""
    print("\n" + "=" * 60)
    print("🧪 Testing Chunking with Sample Content")
    print("=" * 60)
    
    sample_content = """
# Test Document Title

This is a sample document for testing the chunking process.

## Section 1: Introduction

This is the first section of the document. It contains some sample text to test if the chunking process works correctly.

## Section 2: Main Content

This section contains more detailed content. The chunking process should be able to split this into meaningful chunks based on the document structure.

### Subsection 2.1

This is a subsection with additional details. The chunker should recognize the hierarchical structure.

## Section 3: Conclusion

This is the final section of the document. It summarizes the main points and provides closing remarks.

Page 1 of 3
"""
    
    print("📄 Sample content created (for testing)")
    print(f"📏 Content length: {len(sample_content)} characters")
    
    chunker = DocumentChunker()
    
    try:
        chunks = await chunker.chunk_document_content(sample_content, "test_document.md")
        print(f"📦 Chunks created from sample: {len(chunks)}")
        
        if chunks:
            print("📊 Sample chunks preview:")
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                print(f"  Chunk {i}:")
                print(f"    Text: {chunk['chunk_text'][:80]}...")
                print(f"    Pages: {chunk['page_numbers']}")
                print(f"    Title: {chunk['section_title']}")
                print(f"    Tokens: {chunk['token_count']}")
                print()
        else:
            print("❌ No chunks created from sample content")
            print("💡 This indicates a fundamental issue with the chunker configuration")
            
    except Exception as e:
        print(f"❌ Sample chunking failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main debug function"""
    print("🔍 Docling Chunking Debug Tool")
    print("=" * 60)
    
    # Run the main debug process
    await debug_document_processing()
    
    # Run sample content test
    await test_chunking_with_sample_content()
    
    print("\n" + "=" * 60)
    print("📋 DEBUG SUMMARY")
    print("=" * 60)
    print("Check the output above to identify where the chunking process is failing.")

if __name__ == "__main__":
    asyncio.run(main())