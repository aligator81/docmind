"""
Test script for the question export functionality
"""
import requests
import json
import time

def test_question_export_workflow():
    """Test the complete question export workflow"""
    
    # Base URL
    base_url = "http://localhost:8000/api"
    
    # Login first to get token
    print("🔐 Logging in...")
    login_response = requests.post(
        f"{base_url}/auth/login",
        data={
            "username": "superadmin",
            "password": "super123"
        }
    )
    
    if login_response.status_code != 200:
        print(f"❌ Login failed: {login_response.status_code}")
        print(login_response.text)
        return
    
    login_data = login_response.json()
    token = login_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    print("✅ Login successful")
    
    # Test 1: Process questions
    print("\n📝 Testing question processing...")
    questions = [
        "What is the main topic of the document?",
        "How many sections are there?",
        "What are the key findings?",
        "Who is the target audience?"
    ]
    
    process_response = requests.post(
        f"{base_url}/question-export/process-questions",
        json={
            "questions": questions,
            "export_name": "Test Export"
        },
        headers=headers
    )
    
    if process_response.status_code == 200:
        print("✅ Question processing started successfully")
        process_data = process_response.json()
        print(f"   Message: {process_data['message']}")
    else:
        print(f"❌ Question processing failed: {process_response.status_code}")
        print(process_response.text)
        return
    
    # Test 2: Get user exports
    print("\n📁 Testing exports list...")
    exports_response = requests.get(
        f"{base_url}/question-export/exports",
        headers=headers
    )
    
    if exports_response.status_code == 200:
        exports = exports_response.json()
        print(f"✅ Found {len(exports)} exports")
        for export in exports:
            print(f"   - {export['filename']} ({export['questions_count']} questions)")
    else:
        print(f"❌ Failed to get exports: {exports_response.status_code}")
        print(exports_response.text)
    
    # Test 3: Health check
    print("\n🏥 Testing health check...")
    health_response = requests.get(f"{base_url}/question-export/health")
    if health_response.status_code == 200:
        health_data = health_response.json()
        print(f"✅ Health check: {health_data['status']}")
    else:
        print(f"❌ Health check failed: {health_response.status_code}")
    
    print("\n🎉 Question export workflow test completed!")

if __name__ == "__main__":
    test_question_export_workflow()