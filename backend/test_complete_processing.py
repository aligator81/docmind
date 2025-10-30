import requests
import json
import time

def test_complete_processing():
    print("Testing complete document processing pipeline...")
    
    try:
        # Login with testuser
        login_data = {'username': 'testuser', 'password': 'testpassword123'}
        login_response = requests.post('http://localhost:8000/api/auth/login', data=login_data)
        print(f'Login status: {login_response.status_code}')
        
        if login_response.status_code != 200:
            print(f'Login failed: {login_response.text}')
            return
        
        token = login_response.json().get('access_token')
        headers = {'Authorization': f'Bearer {token}'}
        print('Successfully logged in as testuser')
        
        # Upload test document
        print('Uploading test document...')
        with open('test_document.md', 'rb') as f:
            files = {'file': ('test_document.md', f, 'text/markdown')}
            upload_response = requests.post('http://localhost:8000/api/documents/upload', files=files, headers=headers)
        
        print(f'Upload status: {upload_response.status_code}')
        if upload_response.status_code != 200:
            print(f'Upload failed: {upload_response.text}')
            return
        
        upload_result = upload_response.json()
        print(f'Upload response: {upload_result}')
        
        # Extract document ID from response
        document_id = upload_result.get('document', {}).get('id')
        
        if not document_id:
            print('Could not extract document ID from upload response')
            return
        
        print(f'Successfully uploaded document with ID: {document_id}')
        
        # Wait a moment for the document to be processed
        time.sleep(2)
        
        # Get documents to verify upload
        docs_response = requests.get('http://localhost:8000/api/documents', headers=headers)
        if docs_response.status_code == 200:
            documents = docs_response.json()
            print(f'Found {len(documents)} documents:')
            for doc in documents:
                print(f'  - {doc["original_filename"]} (ID: {doc["id"]}, Status: {doc["status"]})')
        
        # Test processing the document
        print(f'Starting complete processing for document {document_id}...')
        process_response = requests.post(f'http://localhost:8000/api/processing/{document_id}/process', headers=headers)
        print(f'Process complete status: {process_response.status_code}')
        
        if process_response.status_code == 200:
            process_result = process_response.json()
            print(f'Processing started successfully: {process_result}')
            
            # Monitor processing status
            print('Monitoring processing status...')
            for i in range(10):  # Check status for up to 20 seconds
                status_response = requests.get(f'http://localhost:8000/api/processing/{document_id}/status', headers=headers)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f'Status check {i+1}: {status_data["status"]}')
                    
                    if status_data["status"] == "processed":
                        print('✅ Document processing completed successfully!')
                        print(f'  - Chunks created: {status_data.get("chunks_count", 0)}')
                        print(f'  - Embeddings created: {status_data.get("embeddings_count", 0)}')
                        break
                    elif status_data["status"] == "failed":
                        print('❌ Document processing failed')
                        break
                
                time.sleep(2)  # Wait 2 seconds between checks
            
            # Final status check
            final_status_response = requests.get(f'http://localhost:8000/api/documents/{document_id}/status', headers=headers)
            if final_status_response.status_code == 200:
                final_status = final_status_response.json()
                print(f'Final status: {final_status["status"]}')
                
        else:
            print(f'Processing failed: {process_response.text}')
            
    except Exception as e:
        print(f'Test failed: {e}')

if __name__ == "__main__":
    test_complete_processing()