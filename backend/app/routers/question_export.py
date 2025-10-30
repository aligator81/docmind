"""
Question Export Router for processing questions and generating Excel files.
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
import os
import time
from typing import List, Optional

from ..database import get_db
from ..models import User, QuestionAnswerExport
from ..schemas import QuestionAnswerRequest, QuestionAnswerResponse, QuestionAnswerExportSchema, ProgressResponse
from ..auth import get_current_active_user
from ..services.excel_service import ExcelExportService

router = APIRouter()

@router.post("/process-questions", response_model=QuestionAnswerResponse)
async def process_questions_and_export(
    request: QuestionAnswerRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Process questions and generate Excel export"""
    
    if not request.questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No questions provided"
        )
    
    if len(request.questions) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many questions. Maximum 100 questions per request."
        )
    
    # Create processing session for progress tracking
    excel_service = ExcelExportService()
    session_id = excel_service.start_processing_session(
        current_user.id,
        len(request.questions)
    )
    
    # Start processing in background
    background_tasks.add_task(
        process_questions_background,
        request.questions,
        request.document_ids,
        current_user.id,
        db,
        current_user,
        request.export_name,
        session_id
    )
    
    return QuestionAnswerResponse(
        success=True,
        message="Question processing started. You'll be notified when the Excel file is ready.",
        questions_processed=0,
        answers_generated=0,
        session_id=session_id
    )

async def process_questions_background(
    questions: List[str],
    document_ids: Optional[List[int]],
    user_id: int,
    db: Session,
    current_user,
    export_name: str = None,
    session_id: Optional[int] = None
):
    """Background task to process questions and generate Excel file"""
    
    start_time = time.time()
    excel_service = ExcelExportService()
    
    try:
        print(f"🚀 Starting background processing for {len(questions)} questions")
        
        # Process questions and generate Excel
        filename, processed_count = await excel_service.process_questions_and_generate_excel(
            questions=questions,
            document_ids=document_ids,
            user_id=user_id,
            db=db,
            current_user=current_user,
            session_id=session_id,
            export_name=export_name
        )
        
        # Save export record
        export_record = excel_service.save_export_record(
            db=db,
            user_id=user_id,
            filename=filename,
            questions_count=processed_count,
            document_ids=document_ids
        )
        
        # Mark processing as completed
        if session_id:
            excel_service.complete_processing_session(session_id)
        
        processing_time = time.time() - start_time
        
        print(f"✅ Question processing completed: {processed_count}/{len(questions)} questions processed in {processing_time:.2f}s")
        print(f"📁 Excel file saved: {filename}")
        
    except Exception as e:
        print(f"❌ Error in background question processing: {e}")
        # Mark processing as failed
        if session_id:
            excel_service.complete_processing_session(session_id)

@router.get("/exports", response_model=List[QuestionAnswerExportSchema])
async def get_user_exports(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's question-answer exports"""
    
    excel_service = ExcelExportService()
    exports = excel_service.get_user_exports(db, current_user.id)
    
    return exports

@router.get("/exports/{export_id}/download")
async def download_export_file(
    export_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Download generated Excel file"""
    
    export = db.query(QuestionAnswerExport).filter(
        QuestionAnswerExport.id == export_id,
        QuestionAnswerExport.user_id == current_user.id
    ).first()
    
    if not export:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export not found"
        )
    
    if not os.path.exists(export.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export file not found"
        )
    
    from fastapi.responses import FileResponse
    return FileResponse(
        path=export.file_path,
        filename=export.filename,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@router.delete("/exports/{export_id}")
async def delete_export(
    export_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete an export and its file"""
    
    excel_service = ExcelExportService()
    success = excel_service.delete_export(db, export_id, current_user.id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export not found"
        )
    
    return {"message": "Export deleted successfully"}

@router.get("/exports/{export_id}/info", response_model=QuestionAnswerExportSchema)
async def get_export_info(
    export_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about an export"""
    
    export = db.query(QuestionAnswerExport).filter(
        QuestionAnswerExport.id == export_id,
        QuestionAnswerExport.user_id == current_user.id
    ).first()
    
    if not export:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export not found"
        )
    
    return export

@router.get("/progress/{session_id}", response_model=ProgressResponse)
async def get_processing_progress(
    session_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """Get real-time progress for question processing session"""
    
    excel_service = ExcelExportService()
    progress = excel_service.get_progress(session_id)
    
    if not progress:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processing session not found or expired"
        )
    
    # Verify the session belongs to the current user
    if progress["user_id"] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this processing session"
        )
    
    return ProgressResponse(
        session_id=session_id,
        user_id=progress["user_id"],
        total_questions=progress["total_questions"],
        processed_questions=progress["processed_questions"],
        current_question=progress["current_question"],
        current_question_index=progress["current_question_index"],
        status=progress["status"],
        progress_percentage=progress["progress_percentage"]
    )

@router.get("/health")
async def health_check():
    """Health check for question export service"""
    return {
        "status": "healthy",
        "service": "question_export",
        "timestamp": time.time()
    }