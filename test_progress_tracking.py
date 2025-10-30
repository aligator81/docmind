#!/usr/bin/env python3
"""
Test script for real-time progress tracking in question-answer export system.
"""

import requests
import json
import time
import sys

# Configuration
BASE_URL = "http://localhost:8000/api"
TEST_QUESTIONS = [
    "What is artificial intelligence?",
    "How does machine learning work?",
    "What are neural networks?",
    "What is natural language processing?",
    "How do transformers work in AI?"
]

def get_auth_token():
    """Get authentication token for testing"""
    try:
        response = requests.post(
            f"{BASE_URL}/auth/login",
            data={
                "username": "testuser",
                "password": "testpassword"
            }
        )
        if response.status_code == 200:
            data = response.json()
            return data["access_token"]
        else:
            print(f"❌ Login failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Login error: {e}")
        return None

def test_progress_tracking():
    """Test the complete progress tracking workflow"""
    
    # Get authentication token
    token = get_auth_token()
    if not token:
        print("❌ Cannot proceed without authentication")
        return False
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    print("🚀 Testing progress tracking system...")
    
    # Step 1: Start question processing
    print("\n📝 Step 1: Starting question processing...")
    payload = {
        "questions": TEST_QUESTIONS,
        "export_name": "Progress Test Export"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/question-export/process-questions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Processing started: {data['message']}")
            
            if "session_id" in data:
                session_id = data["session_id"]
                print(f"📊 Session ID: {session_id}")
                
                # Step 2: Monitor progress
                print("\n📊 Step 2: Monitoring progress...")
                max_attempts = 30  # 30 seconds max
                
                for attempt in range(max_attempts):
                    try:
                        progress_response = requests.get(
                            f"{BASE_URL}/question-export/progress/{session_id}",
                            headers=headers
                        )
                        
                        if progress_response.status_code == 200:
                            progress_data = progress_response.json()
                            
                            print(f"\n📈 Progress Update #{attempt + 1}:")
                            print(f"   Status: {progress_data['status']}")
                            print(f"   Progress: {progress_data['progress_percentage']:.1f}%")
                            print(f"   Processed: {progress_data['processed_questions']}/{progress_data['total_questions']}")
                            print(f"   Current: {progress_data['current_question']}")
                            
                            # Check if processing is complete
                            if progress_data['status'] in ['completed', 'failed']:
                                print(f"\n✅ Processing {progress_data['status']}!")
                                break
                            
                        else:
                            print(f"❌ Failed to get progress: {progress_response.status_code}")
                            break
                            
                    except Exception as e:
                        print(f"❌ Progress check error: {e}")
                        break
                    
                    # Wait before next check
                    time.sleep(1)
                
                # Step 3: Check exports list
                print("\n📁 Step 3: Checking exports list...")
                exports_response = requests.get(
                    f"{BASE_URL}/question-export/exports",
                    headers=headers
                )
                
                if exports_response.status_code == 200:
                    exports = exports_response.json()
                    print(f"✅ Found {len(exports)} exports")
                    
                    if exports:
                        latest_export = exports[0]
                        print(f"📄 Latest export: {latest_export['filename']}")
                        print(f"📊 Questions: {latest_export['questions_count']}")
                        print(f"📅 Created: {latest_export['created_at']}")
                        
                        # Step 4: Test download
                        print("\n📥 Step 4: Testing download...")
                        download_response = requests.get(
                            f"{BASE_URL}/question-export/exports/{latest_export['id']}/download",
                            headers=headers
                        )
                        
                        if download_response.status_code == 200:
                            print(f"✅ Download successful! File size: {len(download_response.content)} bytes")
                        else:
                            print(f"❌ Download failed: {download_response.status_code}")
                    
                else:
                    print(f"❌ Failed to get exports: {exports_response.status_code}")
                
                return True
                
            else:
                print("❌ No session_id in response")
                return False
                
        else:
            print(f"❌ Processing request failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return False

def test_progress_endpoint_directly():
    """Test the progress endpoint with invalid session"""
    print("\n🔍 Testing progress endpoint with invalid session...")
    
    token = get_auth_token()
    if not token:
        return False
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(
            f"{BASE_URL}/question-export/progress/999999",  # Invalid session
            headers=headers
        )
        
        if response.status_code == 404:
            print("✅ Correctly handled invalid session (404)")
        else:
            print(f"❌ Unexpected response for invalid session: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Progress endpoint test error: {e}")

if __name__ == "__main__":
    print("🧪 Question Export Progress Tracking Test")
    print("=" * 50)
    
    # Test the main workflow
    success = test_progress_tracking()
    
    # Test error cases
    test_progress_endpoint_directly()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)