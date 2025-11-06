"""
Test script for the progress endpoint functionality
"""
import requests
import json
import time
import asyncio

def test_progress_workflow():
    """Test the complete progress tracking workflow"""
    
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
    
    # Test 1: Process questions and get session ID
    print("\n📝 Testing question processing with progress tracking...")
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
            "export_name": "Progress Test Export"
        },
        headers=headers
    )
    
    if process_response.status_code == 200:
        print("✅ Question processing started successfully")
        process_data = process_response.json()
        print(f"   Message: {process_data['message']}")
        
        if 'session_id' in process_data:
            session_id = process_data['session_id']
            print(f"   Session ID: {session_id}")
            
            # Test 2: Poll progress endpoint
            print(f"\n📊 Testing progress polling for session {session_id}...")
            
            for i in range(10):  # Poll 10 times
                print(f"   Poll {i+1}: ", end="")
                
                progress_response = requests.get(
                    f"{base_url}/question-export/progress/{session_id}",
                    headers=headers
                )
                
                if progress_response.status_code == 200:
                    progress_data = progress_response.json()
                    print(f"✅ Progress: {progress_data['processed_questions']}/{progress_data['total_questions']} "
                          f"({progress_data['progress_percentage']:.1f}%) - {progress_data['status']}")
                    
                    if progress_data['status'] in ['completed', 'failed']:
                        print(f"   Processing {progress_data['status']}!")
                        break
                        
                elif progress_response.status_code == 404:
                    print("❌ Session not found")
                    break
                else:
                    print(f"❌ Progress check failed: {progress_response.status_code}")
                    print(progress_response.text)
                    break
                
                time.sleep(2)  # Wait 2 seconds between polls
                
            else:
                print("   ⏰ Progress polling timeout")
                
        else:
            print("❌ No session ID returned from backend")
            
    else:
        print(f"❌ Question processing failed: {process_response.status_code}")
        print(process_response.text)
        return
    
    # Test 3: Test progress endpoint with invalid session
    print("\n🧪 Testing progress endpoint with invalid session...")
    invalid_session_response = requests.get(
        f"{base_url}/question-export/progress/999999",
        headers=headers
    )
    
    if invalid_session_response.status_code == 404:
        print("✅ Invalid session correctly returns 404")
    else:
        print(f"❌ Expected 404 but got {invalid_session_response.status_code}")
    
    print("\n🎉 Progress workflow test completed!")

def test_excel_service_progress():
    """Test the Excel service progress tracking directly"""
    print("\n🔧 Testing Excel service progress tracking...")
    
    try:
        from app.services.excel_service import ExcelExportService
        
        # Create service instance
        service = ExcelExportService()
        
        # Test 1: Start processing session
        session_id = service.start_processing_session(user_id=1, total_questions=5)
        print(f"✅ Started processing session: {session_id}")
        
        # Test 2: Update progress
        service.update_progress(session_id, "Test question 1", 1, 1)
        progress = service.get_progress(session_id)
        print(f"✅ Progress updated: {progress['processed_questions']}/{progress['total_questions']}")
        
        # Test 3: Complete session
        service.complete_processing_session(session_id)
        progress = service.get_progress(session_id)
        print(f"✅ Session completed: {progress['status']}")
        
        # Test 4: Get non-existent session
        non_existent = service.get_progress(999999)
        print(f"✅ Non-existent session returns: {non_existent}")
        
    except Exception as e:
        print(f"❌ Excel service test failed: {e}")

if __name__ == "__main__":
    test_progress_workflow()
    test_excel_service_progress()