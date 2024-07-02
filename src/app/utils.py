
import re
import nvidia_smi

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
