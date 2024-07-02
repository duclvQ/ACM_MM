import streamlit as st
import pandas as pd
from io import StringIO
import os
from audio2dia import Audio2Dia
from utils import available_mem_GPUs
UPLOAD_DIR = "./uploads"
# Create a directory to store uploaded files
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    
def upload_text_file(uploaded_file):

        if uploaded_file is not None:
            chose_gpu = -1
            for i, mem in enumerate(available_mem_GPUs()):
                if mem >= 3:
                    chose_gpu = i
                    break
            if chose_gpu == -1:
                st.write("No GPU available, run on CPU")
            if chose_gpu != -1:
                device = "cuda"
                device_index = chose_gpu
            else:
                device = "cpu"
                device_index = 0
                
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            # save the file
            save_dir = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(save_dir, "wb") as f:
                f.write(bytes_data)
            print("file type", uploaded_file.type)
            if uploaded_file.type in ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/flac', 'audio/x-wav', 'audio/x-wav']:
                
                with st.spinner("You've just uploaded an audio file.\n I will convert it to text.\nWait for it..."):
                    model_audio = Audio2Dia(name_model='large-v2',
                                    batch_size=16,
                                    compute_type="float16",
                                    device=device,
                                    device_index=device_index)
                    model_audio.generate(os.path.join(UPLOAD_DIR, uploaded_file.name),
                                        os.path.join(UPLOAD_DIR, f"{uploaded_file.name[:-3]}txt"))    
                    st.success('Done!')  
            st.write("Let's chat now!")
            with open(os.path.join(UPLOAD_DIR, f"{uploaded_file.name[:-3]}txt"), "r") as f:
                text = f.read()
            return True, text
        return False, None
    

        

        