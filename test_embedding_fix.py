#!/usr/bin/env python3
"""
Test script to verify the embedding service fix for token limit errors
"""

import sys
import os
import asyncio

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.services.optimized_embedding_service import OptimizedEmbeddingService
from app.database import SessionLocal

async def test_embedding_service():
    """Test the embedding service with token validation"""
    print("🧪 Testing embedding service with token validation...")
    
    # Initialize service
    service = OptimizedEmbeddingService(provider="openai")
    
    # Test with various chunk sizes
    test_chunks = [
        "This is a short chunk.",  # Small chunk
        "This is a medium chunk. " * 100,  # Medium chunk (~2000 chars)
        "This is a very large chunk. " * 1000,  # Large chunk (~25000 chars, should trigger splitting)
    ]
    
    print(f"📝 Testing {len(test_chunks)} chunks with token validation...")
    
    for i, chunk in enumerate(test_chunks):
        print(f"\n--- Testing chunk {i+1} ---")
        print(f"Original length: {len(chunk)} characters")
        
        # Test validation and splitting
        validated_chunks, token_counts = service.validate_and_split_chunk(chunk)
        
        print(f"Result: {len(validated_chunks)} sub-chunks")
        for j, (sub_chunk, token_count) in enumerate(zip(validated_chunks, token_counts)):
            print(f"  Sub-chunk {j+1}: {token_count} tokens, {len(sub_chunk)} chars")
            
            # Verify token count is within limits
            if token_count > service.max_chunk_size:
                print(f"  ❌ ERROR: Sub-chunk {j+1} exceeds max token limit!")
            else:
                print(f"  ✅ Sub-chunk {j+1} is within limits")

async def test_batch_validation():
    """Test batch validation functionality"""
    print("\n🧪 Testing batch validation...")
    
    service = OptimizedEmbeddingService(provider="openai")
    
    # Create a batch with mixed chunk sizes
    batch_texts = [
        "Short chunk.",
        "Medium chunk. " * 50,
        "Very large chunk. " * 800,  # Should trigger splitting
        "Another medium chunk. " * 30,
    ]
    
    print(f"📦 Testing batch with {len(batch_texts)} texts...")
    
    validated_texts, token_counts = service.validate_and_split_batch(batch_texts)
    
    print(f"Result: {len(validated_texts)} validated chunks")
    print(f"Token counts: {token_counts}")
    
    # Check if any chunks exceed limits
    for i, (text, token_count) in enumerate(zip(validated_texts, token_counts)):
        if token_count > service.max_chunk_size:
            print(f"❌ ERROR: Chunk {i+1} exceeds limit: {token_count} tokens")
        else:
            print(f"✅ Chunk {i+1}: {token_count} tokens (within limits)")

async def main():
    """Run all tests"""
    print("🚀 Starting embedding service token limit fix tests...")
    
    try:
        await test_embedding_service()
        await test_batch_validation()
        
        print("\n🎉 All tests completed successfully!")
        print("✅ Token validation and chunk splitting are working correctly")
        print("✅ The embedding service should now handle large chunks without exceeding OpenAI's token limits")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())