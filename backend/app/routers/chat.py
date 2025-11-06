from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import numpy as np
from datetime import datetime
import hashlib
import asyncio

from ..database import get_db
from ..models import User, Document, DocumentChunk, Embedding, ChatHistory, SystemPrompt
from ..schemas import DocumentChatRequest, ChatResponse
from ..auth import get_current_active_user
from ..config import settings

# Import existing chat logic
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Simple in-memory cache for embeddings (replace with Redis in production)
embedding_cache = {}

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
@router.post("/chat/", response_model=ChatResponse)
@router.post("", response_model=ChatResponse)  # Handle empty path as well
async def chat_with_documents(
    request: DocumentChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Chat with documents using embeddings and LLM"""

    # Check if API keys are configured using settings
    openai_key = settings.openai_api_key
    mistral_key = settings.mistral_api_key

    print(f"🔑 API Key Check - OpenAI: {'✅' if openai_key else '❌'}, Mistral: {'✅' if mistral_key else '❌'}")

    if not openai_key and not mistral_key:
        print("❌ No LLM API keys configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM API keys not configured. Please configure OpenAI or Mistral API keys."
        )

    # Determine LLM provider and model using settings
    if openai_key:
        llm_provider = "openai"
        model_name = settings.openai_chat_model
        print(f"🤖 Using OpenAI provider with model: {model_name}")
    elif mistral_key:
        llm_provider = "mistral"
        model_name = settings.mistral_chat_model
        print(f"🤖 Using Mistral provider with model: {model_name}")
    else:
        print("❌ No LLM provider available despite API key check")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No LLM provider available"
        )

    # Validate document_ids if provided
    if request.document_ids:
        # Ensure document_ids is a list
        if isinstance(request.document_ids, str):
            try:
                import json
                document_ids_list = json.loads(request.document_ids)
            except (json.JSONDecodeError, ValueError):
                document_ids_list = []
        else:
            document_ids_list = request.document_ids

        for doc_id in document_ids_list:
            # Admin and super_admin users can access any document
            if current_user.role in ["admin", "super_admin"]:
                document = db.query(Document).filter(Document.id == doc_id).first()
            else:
                document = db.query(Document).filter(
                    Document.id == doc_id,
                    Document.user_id == current_user.id
                ).first()

            if not document:
                print(f"DEBUG: Document with ID {doc_id} not found for user {current_user.id} (role: {current_user.role})")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Document with ID {doc_id} not found or you don't have access to it. Please refresh the page and select available documents."
                )

            # Check if document is fully processed
            if document.status != "processed":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Document '{document.original_filename}' (ID: {doc_id}) is not fully processed yet. Current status: {document.status}. Please wait for processing to complete."
                )

    # Get relevant context using embeddings
    if request.document_ids:
        if isinstance(request.document_ids, str):
            try:
                import json
                document_ids_list = json.loads(request.document_ids)
            except (json.JSONDecodeError, ValueError):
                document_ids_list = []
        else:
            document_ids_list = request.document_ids
        # Admin and super_admin users can search all documents, regular users only their own
        user_id_filter = None if current_user.role in ["admin", "super_admin"] else current_user.id
        context, references = await get_context_from_db(request.message, db, document_ids_list, user_id_filter)
    else:
        # When no specific documents selected, search only current user's documents (admin/super_admin can search all)
        user_id_filter = None if current_user.role in ["admin", "super_admin"] else current_user.id
        if user_id_filter:
            # Regular users only search their own documents
            user_document_ids = [doc.id for doc in db.query(Document.id).filter(Document.user_id == current_user.id).all()]
            context, references = await get_context_from_db(request.message, db, user_document_ids, current_user.id)
        else:
            # Admin users can search all documents
            context, references = await get_context_from_db(request.message, db, None, None)

    # Always generate response using LLM, even if no context

    # Fetch system prompt
    prompt_record = db.query(SystemPrompt).first()
    if prompt_record:
        system_prompt_template = prompt_record.prompt_text
    else:
        # Fallback default prompt
        system_prompt_template = """You are a helpful assistant that answers questions based on the provided document context from selected documents.

If context is provided, use ONLY the information from the context to answer questions. Consider information from ALL provided document sources when forming your response.

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
7. If no relevant information is found in any document, clearly state that you cannot find information on that topic in the documents, but offer general help if appropriate
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
- This helps users easily locate the referenced information in their documents

If no context is provided or the context is empty, respond as a general helpful assistant and engage in conversation naturally."""

    # Generate response using LLM
    response_text = await generate_llm_response(
        request.message,
        context,
        references,
        llm_provider,
        model_name,
        request.document_ids,
        system_prompt_template
    )

    # Extract context document IDs for response
    context_doc_ids = [ref.get("id", 0) for ref in references] if references else []

    # Prepare detailed references for response
    detailed_references = []
    if references:
        for ref in references:
            # Get document ID from the chunk ID
            chunk = db.query(DocumentChunk).filter(DocumentChunk.id == ref["id"]).first()
            if chunk:
                # Convert page_numbers to string if it's a list
                page_numbers = ref["page_numbers"]
                if isinstance(page_numbers, list):
                    page_numbers = ", ".join(map(str, page_numbers))
                elif page_numbers is None:
                    page_numbers = "N/A"
                
                detailed_references.append({
                    "document_id": chunk.document_id,
                    "filename": ref["filename"],
                    "page_numbers": page_numbers,
                    "section_title": ref["section_title"],
                    "similarity": ref["similarity"]
                })

    # Save chat history
    chat_record = ChatHistory(
        user_id=current_user.id,
        message=request.message,
        response=response_text,
        context_docs=str(context_doc_ids),
        model_used=model_name
    )
    db.add(chat_record)
    db.commit()

    return ChatResponse(
        success=True,
        response=response_text,
        context_docs=context_doc_ids,
        model_used=model_name,
        references=detailed_references
    )

@router.get("/history")
async def get_chat_history(
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's chat history"""
    history = db.query(ChatHistory).filter(
        ChatHistory.user_id == current_user.id
    ).order_by(ChatHistory.created_at.desc()).limit(limit).all()

    return [
        {
            "id": record.id,
            "message": record.message,
            "response": record.response,
            "context_docs": record.context_docs,
            "model_used": record.model_used,
            "created_at": record.created_at
        }
        for record in history
    ]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    try:
        a_array = np.array(a)
        b_array = np.array(b)

        # Calculate dot product
        dot_product = np.dot(a_array, b_array)

        # Calculate magnitudes
        norm_a = np.linalg.norm(a_array)
        norm_b = np.linalg.norm(b_array)

        # Avoid division by zero
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return 0.0

def extract_page_numbers_from_query(query: str) -> List[int]:
    """Extract page numbers from query text"""
    import re
    if not query:
        return []
    
    # Look for page number patterns
    page_patterns = [
        r'page\s+(\d+)',                   # "page 23"
        r'Page\s+(\d+)',                   # "Page 23"
        r'p\.\s*(\d+)',                    # "p. 23"
        r'pp\.\s*(\d+)',                   # "pp. 23"
        r'pg\.\s*(\d+)',                   # "pg. 23"
        r'\(page\s+(\d+)\)',               # (page 23)
        r'\[page\s+(\d+)\]',               # [page 23]
    ]
    
    found_pages = []
    for pattern in page_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            if match.isdigit():
                page_num = int(match)
                if 1 <= page_num <= 10000:  # Reasonable page range
                    found_pages.append(page_num)
    
    return sorted(set(found_pages))

async def get_context_from_db(query: str, db: Session, document_ids: Optional[List[int]] = None, user_id: Optional[int] = None) -> tuple[str, list]:
    """Get relevant context from database using embeddings"""
    try:
        # Get embedding for query
        query_embedding = await get_embedding(query)
        if not query_embedding:
            return "", []

        # Build base query - include section_title and page_numbers
        query_base = db.query(
            DocumentChunk.id,
            DocumentChunk.chunk_text,
            DocumentChunk.section_title,
            DocumentChunk.page_numbers,
            Document.filename,
            Document.original_filename
        ).join(
            Embedding, DocumentChunk.id == Embedding.chunk_id
        ).join(
            Document, DocumentChunk.document_id == Document.id
        ).filter(
            Embedding.embedding_vector.isnot(None)
        )

        # Apply user filtering for non-admin users
        if user_id:
            query_base = query_base.filter(Document.user_id == user_id)

        # If document_ids is specified, filter to only those documents
        if document_ids:
            query_base = query_base.filter(Document.id.in_(document_ids))

        # Limit the search scope for better performance - only get top 100 chunks initially
        print(f"🔍 Searching for relevant context in limited scope...")
        limited_chunks = query_base.limit(100).all()

        if not limited_chunks:
            return "", []

        # Extract page numbers from query for boosting
        query_pages = extract_page_numbers_from_query(query)

        # Calculate similarity scores for limited chunks
        similarities = []
        for chunk_id, chunk_text, section_title, page_numbers, filename, original_filename in limited_chunks:
            # Get the embedding vector for this chunk
            embedding_result = db.query(Embedding).filter(Embedding.chunk_id == chunk_id).first()
            if embedding_result and embedding_result.embedding_vector:
                try:
                    # Convert JSON string to list of floats if needed
                    if isinstance(embedding_result.embedding_vector, str):
                        import json
                        embedding_vector = json.loads(embedding_result.embedding_vector)
                    else:
                        embedding_vector = embedding_result.embedding_vector

                    # Ensure it's a list of floats
                    if embedding_vector and isinstance(embedding_vector, list):
                        embedding_vector = [float(x) for x in embedding_vector]
                        # Calculate cosine similarity
                        similarity = cosine_similarity(query_embedding, embedding_vector)
                        
                        # Boost similarity if query mentions specific pages that match chunk pages
                        if query_pages and page_numbers:
                            # Handle page_numbers as list or string
                            chunk_pages = page_numbers if isinstance(page_numbers, list) else [page_numbers]
                            if any(p in chunk_pages for p in query_pages):
                                similarity += 0.1  # Boost for page match
                        
                        similarities.append((chunk_id, chunk_text, section_title, page_numbers, filename, original_filename, similarity))
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    print(f"Error processing embedding vector for chunk {chunk_id}: {e}")
                    continue

        # Sort by similarity score (highest first) and take top chunks
        similarities.sort(key=lambda x: x[6], reverse=True)
        
        # Use dynamic threshold - take at least top 3 chunks, but filter very low similarity ones
        min_similarity = 0.3  # Lower threshold for better context retrieval
        filtered_similarities = [sim for sim in similarities if sim[6] >= min_similarity]
        
        # If we have filtered results, use them; otherwise use top 3 regardless of similarity
        if filtered_similarities:
            results = filtered_similarities[:5]
        else:
            # If no chunks meet the threshold, take top 3 anyway for context
            results = similarities[:3]
            print(f"⚠️ No high-similarity chunks found, using top {len(results)} chunks anyway")
        
        print(f"✅ Found {len(results)} relevant chunks from {len(limited_chunks)} searched (filtered from {len(similarities)} total)")
        
        # Debug: Show similarity scores
        if results:
            print(f"📊 Similarity scores: {[round(sim[6], 3) for sim in results]}")

        # Build context and references
        context_parts = []
        references = []

        for i, (chunk_id, chunk_text, section_title, page_numbers, filename, original_filename, similarity) in enumerate(results, 1):
            # Add to context with enhanced metadata
            context_parts.append(f"Document: {original_filename}")
            if section_title:
                context_parts.append(f"Section: {section_title}")
            if page_numbers:
                context_parts.append(f"Page(s): {page_numbers}")
            context_parts.append(f"Content: {chunk_text}")
            context_parts.append("---")

            # Add to references with enhanced metadata
            references.append({
                "id": chunk_id,
                "filename": original_filename,
                "page_numbers": page_numbers if page_numbers else "N/A",
                "section_title": section_title if section_title else "",
                "similarity": similarity
            })

        context = "\n".join(context_parts)
        return context, references

    except Exception as e:
        print(f"Error getting context: {e}")
        return "", []

async def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding for text using configured provider - ASYNC VERSION with caching"""
    try:
        # Check cache first
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in embedding_cache:
            print(f"📋 Using cached embedding for query: {text[:50]}...")
            return embedding_cache[cache_key]

        openai_key = os.getenv("OPENAI_API_KEY")
        mistral_key = os.getenv("MISTRAL_API_KEY")

        if openai_key:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=openai_key)
            embedding_model = settings.openai_embedding_model
            print(f"🤖 Getting async embedding from OpenAI ({embedding_model}) for: {text[:50]}...")
            response = await client.embeddings.create(
                model=embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            
            # Cache the result
            embedding_cache[cache_key] = embedding
            print(f"✅ Async embedding generated and cached ({len(embedding)} dimensions)")
            return embedding

        elif mistral_key:
            # Mistral doesn't have official async support yet, use sync with asyncio
            from mistralai import Mistral
            client = Mistral(api_key=mistral_key)
            embedding_model = settings.mistral_embedding_model
            print(f"🤖 Getting embedding from Mistral ({embedding_model}) for: {text[:50]}...")
            
            # Run sync Mistral call in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.embeddings.create(
                    model=embedding_model,
                    inputs=[text]
                )
            )
            embedding = response.data[0].embedding
            
            # Cache the result
            embedding_cache[cache_key] = embedding
            print(f"✅ Mistral embedding generated and cached ({len(embedding)} dimensions)")
            return embedding

        return None

    except Exception as e:
        print(f"❌ Error getting embedding: {e}")
        return None

async def generate_llm_response(
    message: str,
    context: str,
    references: list,
    provider: str,
    model: str,
    document_ids: Optional[List[int]] = None,
    system_prompt_template: str = None
) -> str:
    """Generate response using LLM"""
    try:
        # Format references for the prompt with enhanced metadata
        references_text = ""
        if references:
            references_text = "\n\nSource References:\n"
            for ref in references:
                ref_line = f"• {ref['filename']}"
                if ref['page_numbers'] != "N/A":
                    ref_line += f" (Page(s): {ref['page_numbers']})"
                if ref['section_title']:
                    ref_line += f" - Section: {ref['section_title']}"
                references_text += ref_line + "\n"

        # Create system prompt
        selected_docs_text = ""
        if document_ids:
            # Ensure document_ids is a list
            if isinstance(document_ids, str):
                try:
                    import json
                    document_ids_list = json.loads(document_ids)
                except (json.JSONDecodeError, ValueError):
                    document_ids_list = []
            else:
                document_ids_list = document_ids

            if document_ids_list and len(document_ids_list) > 1:
                selected_docs_text = f"You are answering questions based on {len(document_ids_list)} selected documents. "

        system_prompt = system_prompt_template.format(selected_docs_text=selected_docs_text, context=context, references_text=references_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]

        # Get configured temperature and max_tokens from settings
        if provider == "openai":
            temperature = settings.openai_temperature
            max_tokens = settings.openai_max_tokens
            api_key = settings.openai_api_key
        elif provider == "mistral":
            temperature = settings.mistral_temperature
            max_tokens = settings.mistral_max_tokens
            api_key = settings.mistral_api_key

        if provider == "openai":
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key)
            print(f"🤖 Generating async LLM response with {model} (temp: {temperature}, max_tokens: {max_tokens})...")
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        elif provider == "mistral":
            # Mistral doesn't have official async support yet, use sync with asyncio
            from mistralai import Mistral
            client = Mistral(api_key=api_key)
            print(f"🤖 Generating LLM response with {model} (temp: {temperature}, max_tokens: {max_tokens})...")
            
            # Run sync Mistral call in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            )
            return response.choices[0].message.content

        return "Error: No LLM provider available"

    except Exception as e:
        print(f"Error generating response: {e}")
        return f"I encountered an error while processing your question: {str(e)}"