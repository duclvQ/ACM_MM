
import re
import nvidia_smi

import json
def get_OpenAI_API_key():
    with open("/mnt/HDD1/duclv/open_ai_key.txt", "r") as f:
        openai_api_key = f.read().strip()
    return openai_api_key

def available_mem_GPUs():
    # Get the available memory for each GPU
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    mem = []
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        mem.append(info.free)
    nvidia_smi.nvmlShutdown()
    # convert to GB
    for i in range(len(mem)):
        mem[i] = mem[i]/1024**3
    return mem

def convert_str_to_json(text):
    return json.loads(text)
def txt_reader(file):
    with open(file, 'r') as f:
        text = f.read()
    return text

def regex_search(user_intent):  
    match = re.search(r'<intent>(.*?)</intent>', user_intent)
    
    if match:
        # The first group contains the text between <answer> and </answer>
        answer = match.group(1)
        user_intent = answer
    return user_intent
