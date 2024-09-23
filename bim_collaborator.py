import requests
import streamlit as st

class BIMCollaborator:
    def __init__(self, platform_url, api_key):
        self.platform_url = platform_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def upload_ifc(self, ifc_data, project_id):
        url = f"{self.platform_url}/api/projects/{project_id}/ifc"
        files = {"file": ("model.ifc", ifc_data, "application/ifc")}
        response = requests.post(url, headers=self.headers, files=files)
        return response.json() if response.status_code == 200 else None

    def get_projects(self):
        url = f"{self.platform_url}/api/projects"
        response = requests.get(url, headers=self.headers)
        return response.json() if response.status_code == 200 else []

    def create_issue(self, project_id, issue_data):
        url = f"{self.platform_url}/api/projects/{project_id}/issues"
        response = requests.post(url, headers=self.headers, json=issue_data)
        return response.json() if response.status_code == 201 else None

def setup_bim_collaboration():
    st.sidebar.header("BIM Collaboration Settings")
    platform_url = st.sidebar.text_input("Platform URL")
    api_key = st.sidebar.text_input("API Key", type="password")
    
    if platform_url and api_key:
        return BIMCollaborator(platform_url, api_key)
    return None
