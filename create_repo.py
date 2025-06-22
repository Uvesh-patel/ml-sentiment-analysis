import requests
import sys

def create_github_repo(token, repo_name, private=True):
    """
    Create a new GitHub repository using GitHub API
    """
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    data = {
        'name': repo_name,
        'private': private,
        'description': 'Reduced version of Sentiment Analysis ML project submitted for examination'
    }
    
    response = requests.post('https://api.github.com/user/repos', headers=headers, json=data)
    
    if response.status_code == 201:
        repo_info = response.json()
        print(f"Repository created successfully: {repo_info['html_url']}")
        print(f"Git URL: {repo_info['clone_url']}")
        return repo_info['clone_url']
    else:
        print(f"Failed to create repository. Status code: {response.status_code}")
        print(f"Error message: {response.json()}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_repo.py <github_token> <repo_name>")
        sys.exit(1)
    
    token = sys.argv[1]
    repo_name = sys.argv[2]
    
    create_github_repo(token, repo_name)
