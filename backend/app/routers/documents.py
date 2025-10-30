from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import os
import uuid
import shutil
from datetime import datetime
from typing import List, Dict, Any

from ..database import get_db
from ..models import User, Document, DocumentChunk, Embedding
from ..schemas import DocumentCreate, Document as DocumentSchema, DocumentUploadResponse
from ..auth import get_current_active_user
from ..config import settings
from ..tasks import (
    process_document_task, extract_document_task, chunk_document_task, embed_document_task,
    get_processing_status, get_queue_statistics, background_task_manager
)
from ..security import FileSecurity, validate_upload_file_sync

router = APIRouter()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload and process a document"""
    print(f"📤 Starting upload for file: {file.filename}")
    print(f"🔐 Is authenticated: {current_user is not None}")
    print(f"🔑 Auth token present: {current_user is not None}")
    print(f"👤 Current user: {current_user}")

    # Validate file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    # Remove the dot from extension for comparison
    file_extension_without_dot = file_extension.lstrip('.')
    
    if file_extension_without_dot not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Allowed types: {', '.join(settings.allowed_extensions)}"
        )

    # Validate file size
    if file.size > settings.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_upload_size // (1024*1024)}MB"
        )

    # Generate unique filename
    file_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{file_id}_{timestamp}{file_extension}"
    file_path = os.path.join("data/uploads", filename)

    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )

    # Validate file content and security only if enabled
    if settings.file_validation_enabled:
        # Initialize security validator
        security = FileSecurity()

        # Validate file content and security
        is_valid, validation_message = validate_upload_file_sync(file, security)

        if not is_valid:
            # Clean up the saved file if validation fails
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File security validation failed: {validation_message}"
            )
    else:
        print(f"⚠️ File validation disabled - skipping security checks for {file.filename}")

    # Create document record
    db_document = Document(
        filename=filename,
        original_filename=file.filename,
        file_path=file_path,
        file_size=file.size,
        mime_type=file.content_type or "application/octet-stream",
        user_id=current_user.id,
        status="not processed"
    )

    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    # Don't auto-process - keep as "not processed" so user can manually trigger processing
    # This allows the "Process File" button to appear in the frontend
    print(f"📁 Document {db_document.id} uploaded successfully, waiting for manual processing")

    return DocumentUploadResponse(
        success=True,
        document=db_document,
        message="Document uploaded successfully. Processing has started automatically in the background."
    )

@router.get("", response_model=List[DocumentSchema])
@router.get("/", response_model=List[DocumentSchema])
async def list_documents(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List documents (user's documents for regular users, all documents for admins)"""
    if current_user.role in ["admin", "super_admin"]:
        # Admin and super admin users can see all documents
        documents = db.query(Document).all()
    else:
        # Regular users can only see their own documents
        documents = db.query(Document).filter(Document.user_id == current_user.id).all()
    return documents

@router.get("/{document_id}", response_model=DocumentSchema)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific document (admin/superadmin can access any document)"""
    if current_user.role in ["admin", "super_admin"]:
        document = db.query(Document).filter(Document.id == document_id).first()
    else:
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    return document

@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a document and its associated data"""
    # Allow admins and super admins to delete any document, users can only delete their own
    if current_user.role in ["admin", "super_admin"]:
        document = db.query(Document).filter(Document.id == document_id).first()
    else:
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Delete file from filesystem
    try:
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
    except Exception as e:
        # Log error but don't fail the deletion
        print(f"Warning: Failed to delete file {document.file_path}: {e}")

    # Get chunk IDs first using subquery
    chunk_ids = db.query(DocumentChunk.id).filter(
        DocumentChunk.document_id == document_id
    ).subquery()

    # Delete embeddings using subquery (safer than join-based delete)
    deleted_embeddings = db.query(Embedding).filter(
        Embedding.chunk_id.in_(chunk_ids)
    ).delete(synchronize_session=False)

    # Delete chunks (they reference the document)
    deleted_chunks = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id
    ).delete(synchronize_session=False)

    # Delete the document
    db.delete(document)
    db.commit()

    print(f"🗑️ Deleted document {document_id}: {deleted_chunks} chunks, {deleted_embeddings} embeddings removed")

    return {
        "message": "Document deleted successfully",
        "chunks_removed": deleted_chunks,
        "embeddings_removed": deleted_embeddings
    }

@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get chunks for a document"""
    # Allow admins and super admins to view any document's chunks, users can only view their own
    if current_user.role in ["admin", "super_admin"]:
        document = db.query(Document).filter(Document.id == document_id).first()
    else:
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    chunks = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id
    ).all()

    return {
        "document_id": document_id,
        "chunks": [
            {
                "id": chunk.id,
                "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                "chunk_index": chunk.chunk_index,
                "created_at": chunk.created_at
            }
            for chunk in chunks
        ]
    }

@router.put("/{document_id}/status")
async def update_document_status(
    document_id: int,
    status_update: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update document processing status (admin/superadmin can update any document)"""
    if current_user.role in ["admin", "super_admin"]:
        document = db.query(Document).filter(Document.id == document_id).first()
    else:
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    new_status = status_update.get("status")
    valid_statuses = ["not processed", "extracted", "chunked", "processed"]

    if new_status not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
        )

    document.status = new_status
    if new_status == "processed":
        document.processed_at = datetime.utcnow()

    db.commit()
    db.refresh(document)

    return {
        "message": f"Document status updated to {new_status}",
        "document": document
    }

@router.get("/{document_id}/processing-status")
async def get_document_processing_status(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get the processing status of a document"""
    # Verify document ownership (admins can see all)
    if current_user.role not in ["admin", "super_admin"]:
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        ).first()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

    # Get processing status from background task manager
    processing_status = get_processing_status(document_id)

    if not processing_status:
        # Document not in processing queue, return database status
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        return {
            "document_id": document_id,
            "status": document.status,
            "current_step": "not_processing",
            "progress": 0,
            "created_at": document.created_at.isoformat() if document.created_at else None,
            "processed_at": document.processed_at.isoformat() if document.processed_at else None
        }

    return processing_status

@router.get("/processing/queue-status")
async def get_processing_queue_status(
    current_user: User = Depends(get_current_active_user)
):
    """Get the status of the processing queue (admin only)"""
    if current_user.role not in ["admin", "super_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    return get_queue_statistics()

@router.post("/{document_id}/reprocess")
async def reprocess_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Reprocess a document in the background"""
    # Verify document ownership (admins can reprocess any document)
    if current_user.role not in ["admin", "super_admin"]:
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        ).first()
    else:
        document = db.query(Document).filter(Document.id == document_id).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Check if file exists
    if not os.path.exists(document.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found on disk"
        )

    # Add to background processing queue with higher priority
    background_task_manager.add_job(
        document_id=document_id,
        user_id=current_user.id,
        filename=document.original_filename,
        priority=2  # Higher priority for reprocessing
    )

    # Update status
    document.status = "queued"
    db.commit()

    return {
        "message": "Document queued for reprocessing",
        "document_id": document_id,
        "priority": "high"
    }

@router.post("/{document_id}/process-background")
async def process_document_background(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Start background processing for document (admin/superadmin can process any document)"""
    # Verify document ownership (admin/superadmin can process any document)
    if current_user.role in ["admin", "super_admin"]:
        document = db.query(Document).filter(Document.id == document_id).first()
    else:
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        ).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Check if file exists
    if not os.path.exists(document.file_path):
        raise HTTPException(status_code=404, detail="Document file not found on disk")

    # Start background task
    task = process_document_task.delay(document_id)

    return {
        "message": "Document processing started in background",
        "task_id": task.id,
        "status": "processing",
        "document_id": document_id
    }

@router.post("/{document_id}/extract-background")
async def extract_document_background(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Start background extraction for document (admin/superadmin can process any document)"""
    # Verify document ownership (admin/superadmin can process any document)
    if current_user.role in ["admin", "super_admin"]:
        document = db.query(Document).filter(Document.id == document_id).first()
    else:
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        ).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Check if file exists
    if not os.path.exists(document.file_path):
        raise HTTPException(status_code=404, detail="Document file not found on disk")

    # Start background task
    task = extract_document_task.delay(document_id)

    return {
        "message": "Document extraction started in background",
        "task_id": task.id,
        "status": "processing",
        "document_id": document_id
    }

@router.post("/{document_id}/chunk-background")
async def chunk_document_background(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Start background chunking for document (admin/superadmin can process any document)"""
    # Verify document ownership (admin/superadmin can process any document)
    if current_user.role in ["admin", "super_admin"]:
        document = db.query(Document).filter(Document.id == document_id).first()
    else:
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        ).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Check if document has extracted content
    if not document.content:
        raise HTTPException(status_code=400, detail="Document has no extracted content. Please extract first.")

    # Start background task
    task = chunk_document_task.delay(document_id)

    return {
        "message": "Document chunking started in background",
        "task_id": task.id,
        "status": "processing",
        "document_id": document_id
    }

@router.post("/{document_id}/embed-background")
async def embed_document_background(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Start background embedding generation for document (admin/superadmin can process any document)"""
    # Verify document ownership (admin/superadmin can process any document)
    if current_user.role in ["admin", "super_admin"]:
        document = db.query(Document).filter(Document.id == document_id).first()
    else:
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        ).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Check if document has chunks
    chunk_count = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id
    ).count()

    if chunk_count == 0:
        raise HTTPException(status_code=400, detail="Document has no chunks. Please chunk first.")

    # Start background task
    task = embed_document_task.delay(document_id)

    return {
        "message": "Document embedding generation started in background",
        "task_id": task.id,
        "status": "processing",
        "document_id": document_id
    }

@router.post("/bulk-delete")
async def bulk_delete_documents(
    document_ids: List[int],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete multiple documents and their associated data (admin/superadmin can delete any documents)"""
    deleted_count = 0
    errors = []

    for document_id in document_ids:
        try:
            # Allow admins and super admins to delete any document, users can only delete their own
            if current_user.role in ["admin", "super_admin"]:
                document = db.query(Document).filter(Document.id == document_id).first()
            else:
                document = db.query(Document).filter(
                    Document.id == document_id,
                    Document.user_id == current_user.id
                ).first()

            if not document:
                errors.append(f"Document {document_id} not found")
                continue

            # Delete file from filesystem
            try:
                if os.path.exists(document.file_path):
                    os.remove(document.file_path)
            except Exception as e:
                errors.append(f"Failed to delete file for document {document_id}: {e}")

            # Get chunk IDs first using subquery
            chunk_ids = db.query(DocumentChunk.id).filter(
                DocumentChunk.document_id == document_id
            ).subquery()
        
            # Delete embeddings using subquery (safer than join-based delete)
            deleted_embeddings = db.query(Embedding).filter(
                Embedding.chunk_id.in_(chunk_ids)
            ).delete(synchronize_session=False)
        
            # Delete chunks (they reference the document)
            deleted_chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).delete(synchronize_session=False)

            # Delete the document
            db.delete(document)
            deleted_count += 1

            print(f"🗑️ Bulk deleted document {document_id}: {deleted_chunks} chunks, {deleted_embeddings} embeddings removed")

        except Exception as e:
            errors.append(f"Error deleting document {document_id}: {str(e)}")

    db.commit()

    return {
        "message": f"Successfully deleted {deleted_count} documents",
        "deleted_count": deleted_count,
        "errors": errors
    }

@router.post("/bulk-extract")
async def bulk_extract_documents(
    document_ids: List[int],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Mark multiple documents as extracted"""
    return await bulk_update_document_status(document_ids, "extracted", current_user, db)

@router.post("/bulk-chunk")
async def bulk_chunk_documents(
    document_ids: List[int],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Mark multiple documents as chunked"""
    return await bulk_update_document_status(document_ids, "chunked", current_user, db)

@router.post("/bulk-embed")
async def bulk_embed_documents(
    document_ids: List[int],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Mark multiple documents as processed"""
    return await bulk_update_document_status(document_ids, "processed", current_user, db)

async def bulk_update_document_status(document_ids: List[int], new_status: str, current_user: User, db: Session):
    """Internal function to update status of multiple documents (admin/superadmin can update any documents)"""
    updated_count = 0
    errors = []

    for document_id in document_ids:
        try:
            # Allow admins and super admins to update any document, users can only update their own
            if current_user.role in ["admin", "super_admin"]:
                document = db.query(Document).filter(Document.id == document_id).first()
            else:
                document = db.query(Document).filter(
                    Document.id == document_id,
                    Document.user_id == current_user.id
                ).first()

            if not document:
                errors.append(f"Document {document_id} not found")
                continue

            # Validate status transition
            valid_transitions = {
                "not processed": ["extracted"],
                "extracted": ["chunked"],
                "chunked": ["embedding"],
                "embedding": []  # Final state
            }

            if new_status not in valid_transitions.get(document.status, []):
                errors.append(f"Cannot transition document {document_id} from {document.status} to {new_status}")
                continue

            document.status = new_status
            if new_status == "processed":
                document.processed_at = datetime.utcnow()

            updated_count += 1

        except Exception as e:
            errors.append(f"Error updating document {document_id}: {str(e)}")

    db.commit()

    return {
        "message": f"Successfully updated {updated_count} documents to {new_status}",
        "updated_count": updated_count,
        "errors": errors
    }

@router.post("/{document_id}/extract")
async def extract_document_content(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Extract content from uploaded document (admin/superadmin can process any document)"""
    from ..services.document_processor import DocumentProcessor

    # Get document (admin/superadmin can access any document)
    if current_user.role in ["admin", "super_admin"]:
        document = db.query(Document).filter(Document.id == document_id).first()
    else:
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Check if file exists
    if not os.path.exists(document.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found on disk"
        )

    # Initialize processor
    processor = DocumentProcessor()

    try:
        # Extract document content
        result = await processor.extract_document(document.file_path)

        if result.success:
            # Update document with extracted content
            document.content = result.content
            document.status = "extracted"
            document.processed_at = datetime.utcnow()
            db.commit()

            return {
                "message": "Document extracted successfully",
                "document": document,
                "extraction_method": result.method,
                "processing_time": result.processing_time,
                "content_length": len(result.content)
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Extraction failed: {result.method}"
            )

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during extraction: {str(e)}"
        )

@router.post("/{document_id}/chunk")
async def chunk_document_content(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create chunks from extracted document content (admin/superadmin can process any document)"""
    from ..services.document_chunker import DocumentChunker

    # Get document (admin/superadmin can access any document)
    if current_user.role in ["admin", "super_admin"]:
        document = db.query(Document).filter(Document.id == document_id).first()
    else:
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Check if document has extracted content
    if not document.content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has no extracted content. Please extract first."
        )

    # Initialize chunker
    chunker = DocumentChunker()

    try:
        # Chunk the document content
        result = await chunker.process_document_from_db(db, document_id)

        if result.success:
            # Update document status
            document.status = "chunked"
            document.processed_at = datetime.utcnow()
            db.commit()

            return {
                "message": "Document chunked successfully",
                "document": document,
                "chunks_created": result.chunks_created,
                "processing_time": result.processing_time,
                "metadata": result.metadata
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Chunking failed: {result.metadata.get('error', 'Unknown error')}"
            )

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during chunking: {str(e)}"
        )

@router.post("/{document_id}/embed")
async def embed_document_chunks(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Generate embeddings for document chunks (admin/superadmin can process any document)"""
    from ..services.embedding_service import EmbeddingService

    # Get document (admin/superadmin can access any document)
    if current_user.role in ["admin", "super_admin"]:
        document = db.query(Document).filter(Document.id == document_id).first()
    else:
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == current_user.id
        ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Check if document has chunks
    from ..models import DocumentChunk
    chunk_count = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id
    ).count()

    if chunk_count == 0:
        # Also check if document has content - if not, it needs extraction first
        if not document.content or len(document.content.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document has no extracted content. Please extract content first."
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document has no chunks. Please chunk first."
            )

    # Initialize embedding service
    embedding_service = EmbeddingService()

    try:
        # Process embeddings for chunks in this specific document only
        result = await embedding_service.process_embeddings_for_document(db, document_id)

        if result.success:
            # Update document status to final state
            document.status = "processed"
            document.processed_at = datetime.utcnow()
            db.commit()

            return {
                "message": "Document embeddings generated successfully",
                "document": document,
                "embeddings_created": result.embeddings_created,
                "processing_time": result.processing_time,
                "metadata": result.metadata
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Embedding generation failed: {result.metadata.get('error', 'Unknown error')}"
            )

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during embedding generation: {str(e)}"
        )

async def update_document_status_internal(document_id: int, new_status: str, current_user: User, db: Session):
    """Internal function to update document status"""
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Validate status transition
    valid_transitions = {
        "not processed": ["extracted"],
        "extracted": ["chunked"],
        "chunked": ["embedding"],
        "embedding": []  # Final state
    }

    if new_status not in valid_transitions.get(document.status, []):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot transition from {document.status} to {new_status}"
        )

    document.status = new_status
    if new_status == "processed":
        document.processed_at = datetime.utcnow()

    db.commit()
    db.refresh(document)

    return {
        "message": f"Document status updated to {new_status}",
        "document": document
    }