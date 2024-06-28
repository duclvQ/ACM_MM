import streamlit as st
import pandas as pd
from io import StringIO
import os

UPLOAD_DIR = "./uploads"
# Create a directory to store uploaded files
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    
def upload_text_file():
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "mp3", "wav"])
        
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            st.write("Uploaded file", uploaded_file)
            if uploaded_file.type == "audio/mp3":
                st.write("You've just uploaded an audio file, I will convert it to text")     
            # save the file
            save_dir = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(save_dir, "wb") as f:
                f.write(bytes_data)
            return True
        return False
    
        

        