#!/usr/bin/env python3
"""
Test script to verify chat functionality with configured API keys
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_chat_functionality():
    """Test if chat functionality is properly configured"""
    
    print("🔍 Testing AI Chat Configuration...")
    print("=" * 50)
    
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    mistral_key = os.getenv("MISTRAL_API_KEY")
    
    print(f"📋 API Key Status:")
    print(f"   OpenAI: {'✅ Configured' if openai_key and openai_key != 'your-actual-openai-api-key-here' else '❌ Not Configured'}")
    print(f"   Mistral: {'✅ Configured' if mistral_key and mistral_key != 'your-actual-mistral-api-key-here' else '❌ Not Configured'}")
    
    # Determine available providers
    available_providers = []
    if openai_key and openai_key != 'your-actual-openai-api-key-here':
        available_providers.append("OpenAI")
    if mistral_key and mistral_key != 'your-actual-mistral-api-key-here':
        available_providers.append("Mistral")
    
    if not available_providers:
        print("\n❌ CRITICAL: No valid API keys configured!")
        print("   Please follow the instructions in API_KEY_SETUP_GUIDE.md")
        return False
    
    print(f"\n✅ Available LLM Providers: {', '.join(available_providers)}")
    
    # Test basic API connectivity (optional - requires actual API calls)
    print(f"\n🔗 Testing API Connectivity...")
    
    if "OpenAI" in available_providers:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            # Simple test - list models (free API call)
            models = client.models.list()
            print("   OpenAI: ✅ API connection successful")
        except Exception as e:
            print(f"   OpenAI: ❌ API connection failed - {str(e)}")
    
    if "Mistral" in available_providers:
        try:
            from mistralai import Mistral
            client = Mistral(api_key=mistral_key)
            # Simple test - list models
            models = client.models.list()
            print("   Mistral: ✅ API connection successful")
        except Exception as e:
            print(f"   Mistral: ❌ API connection failed - {str(e)}")
    
    # Check database status
    print(f"\n🗄️ Checking Database Status...")
    try:
        sys.path.append('backend')
        from backend.app.database import SessionLocal
        from backend.app.models import Document, DocumentChunk, Embedding
        
        db = SessionLocal()
        
        # Count documents
        total_docs = db.query(Document).count()
        processed_docs = db.query(Document).filter(Document.status == 'processed').count()
        total_chunks = db.query(DocumentChunk).count()
        total_embeddings = db.query(Embedding).count()
        
        print(f"   Documents: {total_docs} total, {processed_docs} processed")
        print(f"   Chunks: {total_chunks}")
        print(f"   Embeddings: {total_embeddings}")
        
        if processed_docs > 0:
            print("   ✅ Ready for chat - processed documents available")
        else:
            print("   ⚠️ No processed documents - upload and process documents first")
        
        db.close()
        
    except Exception as e:
        print(f"   ❌ Database check failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. Update your .env file with actual API keys")
    print("2. Restart the backend server")
    print("3. Upload and process documents if needed")
    print("4. Test chat functionality in the web interface")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_chat_functionality())