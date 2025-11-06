from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from datetime import datetime, timedelta
import os

from ..database import get_db
from ..models import User, Document, DocumentChunk, Embedding, ChatHistory, APISession, CompanyBranding as CompanyBrandingModel, SystemPrompt
from ..schemas import User as UserSchema, UserCreate, SystemStats, APIConfigCreate, APIConfig, PasswordReset, CompanyBranding, CompanyBrandingCreate
from ..auth import get_admin_user, get_super_admin_user, require_permissions, get_password_hash

router = APIRouter()

@router.post("/users", response_model=UserSchema)
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Create new user (admin only)"""
    from ..models import User as UserModel

    # Check if user already exists
    existing_user = db.query(UserModel).filter(
        (UserModel.username == user_data.username) | (UserModel.email == user_data.email)
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )

    # Role-based restrictions for user creation
    # Admin users can only create regular users, not other admins
    if current_user.role == "admin" and user_data.role in ["admin", "super_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin users can only create regular users"
        )

    # Super admin users can create any role
    # Validate role is one of the allowed values
    if user_data.role not in ["user", "admin", "super_admin"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role. Must be 'user', 'admin', or 'super_admin'"
        )

    # Hash password and create user (active by default for admin creation)
    hashed_password = get_password_hash(user_data.password)
    db_user = UserModel(
        username=user_data.username,
        email=user_data.email,
        password_hash=hashed_password,
        role=user_data.role,
        is_active=True  # Admin-created users are active by default
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user

@router.get("/users", response_model=List[UserSchema])
async def list_users(
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """List all users (admin only)"""
    # Admin users should not see super_admin users
    if current_user.role == "admin":
        users = db.query(User).filter(User.role != "super_admin").all()
    else:
        users = db.query(User).all()
    
    # Convert to schema manually to handle validation issues
    user_schemas = []
    for user in users:
        user_data = {
            "id": user.id,
            "username": user.username,
            "email": user.email if user.email and "@" in str(user.email) else None,
            "role": user.role,
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login": user.last_login.isoformat() if user.last_login else None
        }
        user_schemas.append(UserSchema(**user_data))
    
    return user_schemas

@router.get("/users/{user_id}", response_model=UserSchema)
async def get_user(
    user_id: int,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Get specific user details (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user

@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: int,
    role_data: dict,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Update user role (admin only)"""
    role = role_data.get("role")
    if role not in ["admin", "user", "super_admin"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role. Must be 'admin', 'user', or 'super_admin'"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Role-based restrictions for user role updates
    # Admin users can only update roles to 'user', not to 'admin' or 'super_admin'
    if current_user.role == "admin" and role in ["admin", "super_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin users can only assign 'user' role"
        )

    # Prevent admin from changing their own role
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own role"
        )

    user.role = role
    db.commit()

    return {"message": f"User role updated to {role}"}

@router.put("/users/{user_id}/status")
async def update_user_status(
    user_id: int,
    status_data: dict,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Activate/deactivate user (admin only)"""
    is_active = status_data.get("is_active")
    if is_active is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="is_active field is required"
        )
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    user.is_active = is_active
    db.commit()

    status_text = "activated" if is_active else "deactivated"
    return {"message": f"User {status_text}"}

@router.put("/users/{user_id}/password")
async def reset_user_password(
    user_id: int,
    password_data: PasswordReset,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Reset user password (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Hash the new password
    hashed_password = get_password_hash(password_data.new_password)
    user.password_hash = hashed_password
    db.commit()

    return {"message": "User password reset successfully"}

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Delete user (admin only)"""
    # Prevent admin from deleting themselves
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Get all user's documents before deletion
    user_documents = db.query(Document).filter(Document.user_id == user_id).all()

    # Delete uploaded files from filesystem
    for document in user_documents:
        try:
            if os.path.exists(document.file_path):
                os.remove(document.file_path)
                print(f"Deleted file: {document.file_path}")
        except Exception as e:
            print(f"Warning: Failed to delete file {document.file_path}: {e}")

    # Delete user sessions
    db.query(APISession).filter(APISession.user_id == user_id).delete()

    # Delete chat history
    db.query(ChatHistory).filter(ChatHistory.user_id == user_id).delete()

    # Delete user's documents (cascading will handle chunks and embeddings)
    db.query(Document).filter(Document.user_id == user_id).delete()

    # Finally, delete the user
    db.delete(user)
    db.commit()

    return {"message": "User and all associated data deleted successfully"}

@router.get("/stats", response_model=SystemStats)
async def get_system_stats(
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Get system statistics (admin only)"""
    total_users = db.query(User).count()
    total_documents = db.query(Document).count()
    total_chunks = db.query(DocumentChunk).count()
    total_embeddings = db.query(Embedding).count()

    # Count active sessions (API sessions in last 30 minutes)
    thirty_minutes_ago = datetime.utcnow() - timedelta(minutes=30)
    active_sessions = db.query(APISession).filter(
        APISession.expires_at > thirty_minutes_ago
    ).count()

    return SystemStats(
        total_users=total_users,
        total_documents=total_documents,
        total_chunks=total_chunks,
        total_embeddings=total_embeddings,
        active_sessions=active_sessions
    )

@router.get("/documents")
async def list_all_documents(
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """List all documents in system (admin only)"""
    documents = db.query(Document).all()
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "original_filename": doc.original_filename,
            "user_id": doc.user_id,
            "username": db.query(User.username).filter(User.id == doc.user_id).first()[0],
            "status": doc.status,
            "file_size": doc.file_size,
            "created_at": doc.created_at,
            "processed_at": doc.processed_at
        }
        for doc in documents
    ]

@router.delete("/documents/{document_id}")
async def delete_any_document(
    document_id: int,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Delete any document (admin only)"""
    document = db.query(Document).filter(Document.id == document_id).first()
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
        print(f"Warning: Failed to delete file {document.file_path}: {e}")

    # Delete from database (cascading will handle chunks and embeddings)
    db.delete(document)
    db.commit()

    return {"message": "Document deleted successfully"}

@router.delete("/documents")
async def delete_all_documents(
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Delete all documents (admin only)"""
    try:
        # Get all documents before deletion
        documents = db.query(Document).all()
        deleted_count = len(documents)

        if deleted_count == 0:
            return {"message": "No documents found to delete"}

        # Delete uploaded files from filesystem
        deleted_files = 0
        for document in documents:
            try:
                if os.path.exists(document.file_path):
                    os.remove(document.file_path)
                    deleted_files += 1
            except Exception as e:
                print(f"Warning: Failed to delete file {document.file_path}: {e}")

        # Delete all documents (cascading will handle chunks and embeddings)
        db.query(Document).delete()

        # Also delete any orphaned chunks and embeddings
        db.query(DocumentChunk).delete()
        db.query(Embedding).delete()

        db.commit()

        return {
            "message": f"Successfully deleted {deleted_count} documents and {deleted_files} files",
            "deleted_documents": deleted_count,
            "deleted_files": deleted_files
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete documents: {str(e)}"
        )

@router.get("/health")
async def admin_health_check(
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Detailed health check for admin"""
    try:
        # Test database connection
        db.execute("SELECT 1")

        # Get detailed stats
        stats = await get_system_stats(current_user, db)

        # Check API keys
        openai_key = os.getenv("OPENAI_API_KEY")
        mistral_key = os.getenv("MISTRAL_API_KEY")

        return {
            "status": "healthy",
            "database": "connected",
            "api_keys": {
                "openai": "configured" if openai_key else "missing",
                "mistral": "configured" if mistral_key else "missing"
            },
            "system_stats": stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"System unhealthy: {str(e)}"
        )

@router.post("/api-config")
async def configure_api_keys(
    config: APIConfigCreate,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Configure API keys and models (admin only)"""
    if config.provider not in ["openai", "mistral"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid provider. Must be 'openai' or 'mistral'"
        )

    if not config.api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="API key is required"
        )

    # Define available models for validation
    available_models = {
        "openai": {
            "chat": ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
            "embedding": ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"]
        },
        "mistral": {
            "chat": ["mistral-large", "mistral-medium", "mistral-small"],
            "embedding": ["mistral-embed"]
        }
    }

    # Validate chat model if provided
    if config.model and config.model not in available_models[config.provider]["chat"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid chat model for {config.provider}. Available: {', '.join(available_models[config.provider]['chat'])}"
        )

    # Validate embedding model if provided
    if config.embedding_model and config.embedding_model not in available_models[config.provider]["embedding"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid embedding model for {config.provider}. Available: {', '.join(available_models[config.provider]['embedding'])}"
        )

    # Validate API key by making a test request
    try:
        if config.provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=config.api_key)
            # Test with a simple request
            client.models.list()
        elif config.provider == "mistral":
            from mistralai import Mistral
            client = Mistral(api_key=config.api_key)
            # Test with a simple request
            client.models.list()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid API key: {str(e)}"
        )

    # Save configuration to .env file for persistence
    env_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    
    # Read existing .env file if it exists
    env_lines = []
    if os.path.exists(env_file_path):
        with open(env_file_path, 'r') as f:
            env_lines = f.readlines()
    
    # Update or add configuration lines
    config_updates = {
        f"{config.provider.upper()}_API_KEY": config.api_key,
        f"{config.provider.upper()}_CHAT_MODEL": config.model,
        f"{config.provider.upper()}_EMBEDDING_MODEL": config.embedding_model,
        f"{config.provider.upper()}_MAX_TOKENS": str(config.max_tokens) if config.max_tokens else None,
        f"{config.provider.upper()}_TEMPERATURE": str(config.temperature) if config.temperature else None
    }
    
    # Update existing lines or add new ones
    updated_lines = []
    processed_keys = set()
    
    for line in env_lines:
        line_stripped = line.strip()
        if line_stripped and not line_stripped.startswith('#'):
            key = line_stripped.split('=')[0]
            if key in config_updates and config_updates[key] is not None:
                # Update existing line
                updated_lines.append(f"{key}={config_updates[key]}\n")
                processed_keys.add(key)
            else:
                # Keep existing line unchanged
                updated_lines.append(line)
        else:
            # Keep comments and empty lines
            updated_lines.append(line)
    
    # Add any new configuration that wasn't in the file
    for key, value in config_updates.items():
        if key not in processed_keys and value is not None:
            updated_lines.append(f"{key}={value}\n")
    
    # Write updated .env file
    with open(env_file_path, 'w') as f:
        f.writelines(updated_lines)
    
    # Also update environment variables for immediate use
    os.environ[f"{config.provider.upper()}_API_KEY"] = config.api_key
    if config.model:
        os.environ[f"{config.provider.upper()}_CHAT_MODEL"] = config.model
    if config.embedding_model:
        os.environ[f"{config.provider.upper()}_EMBEDDING_MODEL"] = config.embedding_model
    if config.max_tokens:
        os.environ[f"{config.provider.upper()}_MAX_TOKENS"] = str(config.max_tokens)
    if config.temperature:
        os.environ[f"{config.provider.upper()}_TEMPERATURE"] = str(config.temperature)

    return {
        "message": f"{config.provider.title()} API configuration saved successfully",
        "provider": config.provider,
        "is_active": config.is_active,
        "model": config.model,
        "embedding_model": config.embedding_model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature
    }

@router.post("/branding")
async def save_company_branding(
    branding: CompanyBrandingCreate,
    current_user: User = Depends(get_super_admin_user),
    db: Session = Depends(get_db)
):
    """Save company branding (super admin only)"""
    # Check if branding already exists
    existing = db.query(CompanyBrandingModel).first()
    if existing:
        existing.company_name = branding.company_name
        existing.logo_url = branding.logo_url
        db.commit()
        db.refresh(existing)
        return {
            "id": existing.id,
            "company_name": existing.company_name,
            "logo_url": existing.logo_url
        }
    else:
        new_branding = CompanyBrandingModel(
            company_name=branding.company_name,
            logo_url=branding.logo_url
        )
        db.add(new_branding)
        db.commit()
        db.refresh(new_branding)
        return {
            "id": new_branding.id,
            "company_name": new_branding.company_name,
            "logo_url": new_branding.logo_url
        }

@router.get("/branding")
async def get_company_branding(
    current_user: User = Depends(get_super_admin_user),
    db: Session = Depends(get_db)
):
    """Get company branding (super admin only)"""
    branding = db.query(CompanyBrandingModel).first()
    if not branding:
        return {"id": 0, "company_name": "", "logo_url": None}
    return {
        "id": branding.id,
        "company_name": branding.company_name,
        "logo_url": branding.logo_url
    }

@router.get("/branding/public")
async def get_company_branding_public(
    db: Session = Depends(get_db)
):
    """Get company branding (public)"""
    branding = db.query(CompanyBrandingModel).first()
    if not branding:
        # Return default values if not set
        return {"id": 0, "company_name": "📚 Doc Q&A", "logo_url": None}
    return {
        "id": branding.id,
        "company_name": branding.company_name,
        "logo_url": branding.logo_url
    }

@router.get("/system-prompt")
async def get_system_prompt(
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Get system prompt (admin only)"""
    prompt = db.query(SystemPrompt).first()
    if not prompt:
        # Insert default prompt
        default_prompt = """You are a helpful assistant that answers questions based on the provided document context from selected documents.

Use ONLY the information from the context to answer questions. Consider information from ALL provided document sources when forming your response.

Context:
{context}

{references_text}

When answering, please:
1. Be direct and helpful
2. Reference specific documents when relevant (mention which document the information comes from)
3. Include page numbers and section titles when available to help users locate the information
4. Synthesize information from multiple documents when possible
5. Compare and contrast information from different documents when relevant
6. If information conflicts between documents, acknowledge the differences
7. If no relevant information is found in any document, clearly state that
8. When synthesizing from multiple sources, indicate which documents contributed to your answer

For multi-document questions:
- If the question spans multiple documents, synthesize information from all relevant sources
- If documents provide complementary information, combine them logically
- If documents provide conflicting information, present both perspectives
- Always attribute information to specific documents when possible, including page numbers and sections when available

When referencing sources:
- Mention the document name, page number(s), and section title when available
- Example: "According to [Document Name] (Page X, Section Y)..."
- Example: "As mentioned in [Document Name] on page X..."
- Example: "Based on the section '[Section Title]' in [Document Name]..."
- This helps users easily locate the referenced information in their documents"""
        prompt = SystemPrompt(prompt_text=default_prompt)
        db.add(prompt)
        db.commit()
        db.refresh(prompt)
    return {"prompt_text": prompt.prompt_text}

@router.post("/system-prompt")
async def update_system_prompt(
    prompt_data: dict,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Update system prompt (admin only)"""
    prompt_text = prompt_data.get("prompt_text")
    if not prompt_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="prompt_text is required"
        )
    prompt = db.query(SystemPrompt).first()
    if not prompt:
        prompt = SystemPrompt(prompt_text=prompt_text)
        db.add(prompt)
    else:
        prompt.prompt_text = prompt_text
        prompt.updated_at = func.now()
    db.commit()
    db.refresh(prompt)
    return {"message": "System prompt updated", "prompt_text": prompt.prompt_text}