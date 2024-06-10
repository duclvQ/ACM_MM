
import re




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
