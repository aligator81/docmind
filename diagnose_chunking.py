#!/usr/bin/env python3
"""
Diagnostic script to identify chunking process failures
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.database import SessionLocal
from app.models import Document, DocumentChunk, Embedding

def check_database_state():
    """Check current state of documents and chunks in database"""
    print("🔍 Checking database state...")
    
    db = SessionLocal()
    try:
        # Get all documents
        documents = db.query(Document).all()
        print(f"📄 Total documents in database: {len(documents)}")
        
        # Analyze document status
        status_counts = {}
        for doc in documents:
            status = doc.status or "unknown"
            status_counts[status] = status_counts.get(status, 0) + 1
            
        print("📊 Document status breakdown:")
        for status, count in status_counts.items():
            print(f"  - {status}: {count} documents")
        
        # Check for documents with content but no chunks
        docs_with_content_no_chunks = []
        for doc in documents:
            if doc.content and doc.content.strip():
                chunk_count = db.query(DocumentChunk).filter(
                    DocumentChunk.document_id == doc.id
                ).count()
                if chunk_count == 0:
                    docs_with_content_no_chunks.append({
                        'id': doc.id,
                        'filename': doc.filename,
                        'status': doc.status,
                        'content_length': len(doc.content)
                    })
        
        print(f"\n⚠️ Documents with content but no chunks: {len(docs_with_content_no_chunks)}")
        for doc in docs_with_content_no_chunks:
            print(f"  - ID {doc['id']}: {doc['filename']} (status: {doc['status']}, content: {doc['content_length']} chars)")
        
        # Check chunk statistics
        total_chunks = db.query(DocumentChunk).count()
        print(f"\n📦 Total chunks in database: {total_chunks}")
        
        # Check embeddings
        total_embeddings = db.query(Embedding).count()
        print(f"🧠 Total embeddings in database: {total_embeddings}")
        
        # Check for chunks without embeddings
        chunks_without_embeddings = db.query(DocumentChunk).filter(
            ~DocumentChunk.id.in_(
                db.query(Embedding.chunk_id).subquery()
            )
        ).count()
        print(f"❌ Chunks without embeddings: {chunks_without_embeddings}")
        
        return {
            'total_documents': len(documents),
            'status_counts': status_counts,
            'docs_with_content_no_chunks': docs_with_content_no_chunks,
            'total_chunks': total_chunks,
            'total_embeddings': total_embeddings,
            'chunks_without_embeddings': chunks_without_embeddings
        }
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return None
    finally:
        db.close()

def check_file_system():
    """Check file system for issues"""
    print("\n📁 Checking file system...")
    
    # Check backend cache directory
    cache_dir = Path("backend/cache")
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.json"))
        print(f"📋 Cache files found: {len(cache_files)}")
        
        # Check for recent cache files
        recent_files = [f for f in cache_files if f.stat().st_mtime > (os.path.getmtime(__file__) - 3600)]
        print(f"🕒 Recent cache files (last hour): {len(recent_files)}")
    else:
        print("❌ Cache directory not found")
    
    # Check output directory
    output_dir = Path("output")
    if output_dir.exists():
        output_files = list(output_dir.glob("*.md"))
        print(f"📄 Output files found: {len(output_files)}")
    else:
        print("❌ Output directory not found")
    
    # Check data uploads
    uploads_dir = Path("data/uploads")
    if uploads_dir.exists():
        upload_files = list(uploads_dir.glob("*"))
        print(f"📤 Upload files found: {len(upload_files)}")
    else:
        print("❌ Uploads directory not found")

def check_environment():
    """Check environment and dependencies"""
    print("\n🔧 Checking environment...")
    
    # Check required packages
    required_packages = [
        'docling', 'transformers', 'sqlalchemy', 'psycopg2-binary'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}: Available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}: Missing")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
    
    # Check environment variables
    env_vars = ['NEON_CONNECTION_STRING', 'MISTRAL_API_KEY']
    missing_env_vars = []
    
    for var in env_vars:
        if os.getenv(var):
            print(f"✅ {var}: Set")
        else:
            missing_env_vars.append(var)
            print(f"❌ {var}: Not set")
    
    if missing_env_vars:
        print(f"\n⚠️ Missing environment variables: {', '.join(missing_env_vars)}")

def main():
    print("🔍 Docling Chunking Process Diagnostic")
    print("=" * 50)
    
    # Run diagnostics
    check_environment()
    check_file_system()
    db_state = check_database_state()
    
    print("\n" + "=" * 50)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    if db_state:
        if db_state['docs_with_content_no_chunks']:
            print("🚨 ISSUE DETECTED: Documents have content but no chunks")
            print("   This indicates the chunking process failed")
            print("\n💡 RECOMMENDED ACTIONS:")
            print("   1. Check the document_chunker.py service for errors")
            print("   2. Verify the HybridChunker configuration")
            print("   3. Check if Docling is properly installed and configured")
            print("   4. Try running chunking on a single document to debug")
        else:
            print("✅ No obvious chunking issues detected in database")
            
        if db_state['chunks_without_embeddings'] > 0:
            print(f"⚠️ Found {db_state['chunks_without_embeddings']} chunks without embeddings")
            print("   This is normal if embedding process hasn't run yet")
    
    print("\n🔧 NEXT STEPS:")
    print("   1. Run this diagnostic to identify specific issues")
    print("   2. Check the backend logs for error messages")
    print("   3. Test chunking on a single document manually")
    print("   4. Verify all dependencies are properly installed")

if __name__ == "__main__":
    main()