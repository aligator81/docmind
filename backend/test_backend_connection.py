import requests
import json

def test_backend():
    print("Testing backend connection...")
    
    try:
        # Login with testuser
        login_data = {'username': 'testuser', 'password': 'testpassword123'}
        login_response = requests.post('http://localhost:8000/api/auth/login', data=login_data)
        print(f'Login status: {login_response.status_code}')
        
        if login_response.status_code == 200:
            token = login_response.json().get('access_token')
            print('Successfully logged in as testuser')
        else:
            print(f'Login failed: {login_response.text}')
            return
        print(f'Login status: {login_response.status_code}')
        
        if login_response.status_code == 200:
            token = login_response.json().get('access_token')
            headers = {'Authorization': f'Bearer {token}'}
            
            # Get documents to see what's available
            docs_response = requests.get('http://localhost:8000/api/documents', headers=headers)
            print(f'Documents status: {docs_response.status_code}')
            if docs_response.status_code == 200:
                documents = docs_response.json()
                print(f'Found {len(documents)} documents')
                for doc in documents:
                    print(f'  - {doc["original_filename"]} (ID: {doc["id"]}, Status: {doc["status"]})')
                    
                    # Test processing if document is not processed
                    if doc["status"] == "not processed":
                        print(f'    Testing processing for document {doc["id"]}...')
                        process_response = requests.post(f'http://localhost:8000/api/documents/{doc["id"]}/process-complete', headers=headers)
                        print(f'    Process complete status: {process_response.status_code}')
                        if process_response.status_code != 200:
                            print(f'    Process error: {process_response.text}')
            else:
                print(f'Documents error: {docs_response.text}')
        else:
            print(f'Login failed: {login_response.text}')
            
    except Exception as e:
        print(f'Test failed: {e}')

if __name__ == "__main__":
    test_backend()