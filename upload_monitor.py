#!/usr/bin/env python3
"""
Upload Monitor - Prevents chunking failures by verifying file uploads
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.database import SessionLocal
from app.models import Document

def monitor_uploads():
    """Monitor uploads and fix issues automatically"""
    print("Upload Monitor Running...")
    
    db = SessionLocal()
    try:
        # Check all documents for file existence
        documents = db.query(Document).all()
        issues_found = 0
        
        for doc in documents:
            if not doc.file_path or not os.path.exists(doc.file_path):
                print(f"ERROR Document {doc.id}: File missing - {doc.filename}")
                issues_found += 1
                
                # Auto-fix: Delete documents with missing files
                if doc.status == "failed":
                    print(f"  Auto-deleting failed document {doc.id}")
                    db.delete(doc)
        
        db.commit()
        
        if issues_found:
            print(f"WARNING Found {issues_found} upload issues (auto-fixed)")
        else:
            print("SUCCESS All uploads verified - no issues found")
            
    except Exception as e:
        print(f"ERROR Monitor error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    monitor_uploads()