"""
Excel Export Service for generating Excel files from question-answer pairs.
"""
import pandas as pd
import os
import uuid
import asyncio
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from sqlalchemy.orm import Session
from ..models import QuestionAnswerExport
from ..schemas import DocumentChatRequest

# Global progress tracking for real-time updates
_processing_progress: Dict[int, Dict] = {}

class ExcelExportService:
    """Service for generating Excel files from question-answer pairs"""
    
    def __init__(self):
        self.exports_dir = "data/exports"
        os.makedirs(self.exports_dir, exist_ok=True)
    
    async def process_questions_and_generate_excel(
        self,
        questions: List[str],
        document_ids: Optional[List[int]],
        user_id: int,
        db: Session,
        current_user,
        session_id: Optional[int] = None
    ) -> Tuple[str, int]:
        """Process questions and generate Excel file"""
        
        results = []
        processed_count = 0
        
        for i, question in enumerate(questions, 1):
            try:
                print(f"🔄 Processing question {i}/{len(questions)}: {question[:50]}...")
                
                # Update progress if session ID provided
                if session_id:
                    self.update_progress(session_id, question, i, processed_count)
                
                # Use existing chat functionality to get AI answers
                chat_request = DocumentChatRequest(
                    message=question,
                    document_ids=document_ids
                )
                
                # Import chat function here to avoid circular imports
                from ..routers.chat import chat_with_documents
                
                # Get AI response using your existing chat system
                chat_response = await chat_with_documents(
                    request=chat_request,
                    current_user=current_user,
                    db=db
                )
                
                if chat_response.success:
                    results.append({
                        "Question": question,
                        "AI Answer": chat_response.response,
                        "Question Number": i,
                        "Generated At": datetime.now().isoformat(),
                        "Model Used": chat_response.model_used or "Unknown"
                    })
                    processed_count += 1
                    print(f"✅ Processed question {i}/{len(questions)}")
                    
                    # Update progress after successful processing
                    if session_id:
                        self.update_progress(session_id, question, i, processed_count)
                else:
                    results.append({
                        "Question": question,
                        "AI Answer": "Failed to generate answer",
                        "Question Number": i,
                        "Generated At": datetime.now().isoformat(),
                        "Model Used": "Error"
                    })
                    print(f"❌ Failed to process question {i}/{len(questions)}")
                
                # Add small delay to avoid rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"❌ Error processing question {i}: {e}")
                # Add question with error message
                results.append({
                    "Question": question,
                    "AI Answer": f"Error: {str(e)}",
                    "Question Number": i,
                    "Generated At": datetime.now().isoformat(),
                    "Model Used": "Error"
                })
        
        # Generate Excel file
        filename = await self._generate_excel_file(results, user_id)
        return filename, processed_count
    
    async def _generate_excel_file(self, results: List[Dict], user_id: int) -> str:
        """Generate Excel file from results"""
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_id = str(uuid.uuid4())[:8]
        filename = f"qa_export_{user_id}_{timestamp}_{file_id}.xlsx"
        file_path = os.path.join(self.exports_dir, filename)
        
        try:
            # Create Excel file with formatting
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Questions & Answers', index=False)
                
                # Get workbook and worksheet for formatting
                workbook = writer.book
                worksheet = writer.sheets['Questions & Answers']
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)  # Max width 50
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                # Add header formatting
                for cell in worksheet[1]:
                    cell.font = cell.font.copy(bold=True)
                
                # Add summary sheet
                summary_data = {
                    'Metric': ['Total Questions', 'Successfully Processed', 'Processing Date'],
                    'Value': [len(results), len([r for r in results if 'Error' not in r.get('AI Answer', '')]), datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Format summary sheet
                summary_sheet = writer.sheets['Summary']
                for column in summary_sheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 30)
                    summary_sheet.column_dimensions[column_letter].width = adjusted_width
                
                for cell in summary_sheet[1]:
                    cell.font = cell.font.copy(bold=True)
            
            print(f"✅ Excel file generated: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error generating Excel file: {e}")
            raise
    
    def save_export_record(
        self, 
        db: Session, 
        user_id: int, 
        filename: str, 
        questions_count: int,
        document_ids: Optional[List[int]] = None
    ) -> QuestionAnswerExport:
        """Save export record to database"""
        
        file_path = os.path.join(self.exports_dir, filename)
        file_size = os.path.getsize(file_path)
        
        export_record = QuestionAnswerExport(
            user_id=user_id,
            filename=filename,
            file_path=file_path,
            file_size=file_size,
            questions_count=questions_count,
            document_ids=str(document_ids) if document_ids else None,
            status="completed"
        )
        
        db.add(export_record)
        db.commit()
        db.refresh(export_record)
        
        print(f"✅ Export record saved: {export_record.id}")
        return export_record
    
    def get_user_exports(self, db: Session, user_id: int) -> List[QuestionAnswerExport]:
        """Get all exports for a user"""
        return db.query(QuestionAnswerExport).filter(
            QuestionAnswerExport.user_id == user_id
        ).order_by(QuestionAnswerExport.created_at.desc()).all()
    
    def delete_export(self, db: Session, export_id: int, user_id: int) -> bool:
        """Delete an export and its file"""
        export = db.query(QuestionAnswerExport).filter(
            QuestionAnswerExport.id == export_id,
            QuestionAnswerExport.user_id == user_id
        ).first()
        
        if not export:
            return False
        
        # Delete file
        try:
            if os.path.exists(export.file_path):
                os.remove(export.file_path)
        except Exception as e:
            print(f"⚠️ Failed to delete file {export.file_path}: {e}")
        
        # Delete database record
        db.delete(export)
        db.commit()
        
        return True
    
    def start_processing_session(self, user_id: int, total_questions: int) -> int:
        """Start a new processing session and return session ID"""
        session_id = int(datetime.now().timestamp() * 1000)  # Unique session ID
        _processing_progress[session_id] = {
            'user_id': user_id,
            'total_questions': total_questions,
            'processed_questions': 0,
            'current_question': '',
            'status': 'processing',
            'start_time': datetime.now(),
            'current_question_index': 0
        }
        return session_id
    
    def update_progress(self, session_id: int, current_question: str, current_index: int, processed_count: int):
        """Update progress for a processing session"""
        if session_id in _processing_progress:
            _processing_progress[session_id].update({
                'current_question': current_question,
                'current_question_index': current_index,
                'processed_questions': processed_count,
                'status': 'processing'
            })
    
    def complete_processing_session(self, session_id: int):
        """Mark a processing session as completed"""
        if session_id in _processing_progress:
            _processing_progress[session_id]['status'] = 'completed'
            _processing_progress[session_id]['end_time'] = datetime.now()
    
    def get_progress(self, session_id: int) -> Optional[Dict]:
        """Get current progress for a session"""
        return _processing_progress.get(session_id)
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old processing sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in _processing_progress.items():
            if 'start_time' in session_data:
                age = current_time - session_data['start_time']
                if age.total_seconds() > max_age_hours * 3600:
                    expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del _processing_progress[session_id]