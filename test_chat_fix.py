#!/usr/bin/env python3
"""
Test script to verify the chat functionality is working after the schema fix.
"""

import requests
import json
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_chat_functionality():
    """Test the chat endpoint with the new schema"""
    
    # Test data matching the new DocumentChatRequest schema
    test_data = {
        "message": "What is this document about?",
        "document_ids": [100]  # Use the document ID that was processed earlier
    }
    
    # Get auth token (you'll need to login first)
    login_data = {
        "username": "admin",
        "password": "admin123"
    }
    
    try:
        # Login to get token
        print("🔐 Logging in...")
        login_response = requests.post(
            "http://localhost:8000/api/auth/login",
            data=login_data
        )
        
        if login_response.status_code != 200:
            print(f"❌ Login failed: {login_response.status_code}")
            print(f"Response: {login_response.text}")
            return False
            
        token_data = login_response.json()
        access_token = token_data.get("access_token")
        
        if not access_token:
            print("❌ No access token received")
            return False
            
        print("✅ Login successful")
        
        # Test chat endpoint
        print("💬 Testing chat endpoint...")
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        chat_response = requests.post(
            "http://localhost:8000/api/chat",
            headers=headers,
            json=test_data
        )
        
        print(f"📊 Chat response status: {chat_response.status_code}")
        
        if chat_response.status_code == 200:
            chat_data = chat_response.json()
            print("✅ Chat endpoint working!")
            print(f"🤖 Response: {chat_data.get('response', 'No response')}")
            print(f"📄 Model used: {chat_data.get('model_used', 'Unknown')}")
            print(f"🔗 References: {len(chat_data.get('references', []))}")
            return True
        else:
            print(f"❌ Chat request failed: {chat_response.status_code}")
            print(f"Error details: {chat_response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the backend is running.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Chat Functionality Fix")
    print("=" * 50)
    
    success = test_chat_functionality()
    
    print("=" * 50)
    if success:
        print("🎉 Chat functionality is working correctly!")
    else:
        print("💥 Chat functionality still has issues.")