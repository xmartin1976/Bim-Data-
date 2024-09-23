import streamlit as st

def get_file_content_as_string(file_path):
    with open(file_path, "r") as file:
        return file.read()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'ifc'}

@st.cache_data
def load_css():
    return get_file_content_as_string("style.css")
