# from openai import OpenAI
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState
from transformers import pipeline
from accelerate import Accelerator
import torch
import sys
import GPUtil
import time 
import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,5,6,7"
distributed_state = PartialState()
accelerator = Accelerator()
CHAT_COMPLETION_MODELS = ["gpt-4o", "gpt-4-turbo-preview", 'gpt-3.5-turbo',  'gpt-4', "gpt-4-32k", "gpt-3.5-turbo-16k"]

class Phi3Model():
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float,
        **kwargs):
        
        _model_name = "microsoft/Phi-3-mini-128k-instruct"
        # _model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.model = None
        if model_name == 'base':
            self.model = AutoModelForCausalLM.from_pretrained(_model_name,  trust_remote_code=True)
        elif model_name == 'optimizer':
            self.model = AutoModelForCausalLM.from_pretrained(_model_name,  trust_remote_code=True, device_map= 'auto')
        else:
            sys.exit()
            raise ValueError(f"Model {model_name} not supported.")
        
        self.start_time = time.time()
        # convert to datetime format
        self.start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))
        
        if model_name == 'base':
            self.model = self.model.to("cuda:4")
        # elif model_name == 'optimizer':
        #     self.model = self.model.to("cuda:5")
        # else:
        #     raise ValueError(f"Model {model_name} not supported.")
        # self.model = None
        # if model_name == 'base':
        #     self.model = pipeline("_", model=_model_name, device=0)
        # elif model_name == 'optimizer':
        
        #     self.model = pipeline("_", model=_model_name, device=1)
        # else:
        #     raise ValueError(f"Model {model_name} not supported.")
            
        self.model.half()
        self.tokenizer = AutoTokenizer.from_pretrained(_model_name, device_map="cuda", torch_dtype="auto", trust_remote_code=True,)
        self.model_name = model_name
        self.temperature = temperature
        
        # if model_name in CHAT_COMPLETION_MODELS: 
        self.batch_forward_func = self.batch_forward_chatcompletion
        self.generate = self.gpt_chat_completion
        # else:
        #     raise ValueError(f"Model {model_name} not supported.")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.generation_args = {}
        if self.model_name == 'optimizer':
            self.generation_args = {
                "max_new_tokens": 1000,
                "return_full_text": False,
                "do_sample":True,
                "temperature": 1.0,
                
            }
        else:
            self.generation_args = {
                "max_new_tokens": 1000,
                "return_full_text": False,
                "do_sample":False,
                "num_beams": 1,
               
            }
    
    def batch_forward_chatcompletion(self, batch_prompts):
        """
        Input a batch of prompts
        """
        responses = []
        for prompt in batch_prompts:
            response = self.gpt_chat_completion(prompt=prompt)
            responses.append(response)
        return responses
    
    # def split_text_into_pieces(self, text,\
    #                        max_tokens=900,\
    #                        overlapPercent=10): \
    #     # Tokenize the text
    #     tokens = self.tokenizer.tokenize(text)

    #     # Calculate the overlap in tokens
    #     overlap_tokens = int(max_tokens * overlapPercent / 100)

    #     # Split the tokens into chunks of size
    #     # max_tokens with overlap
    #     pieces = [tokens[i:i + max_tokens]
    #             for i in range(0, len(tokens),
    #                             max_tokens - overlap_tokens)]

    #     # Convert the token pieces back into text
    #     text_pieces = [self.tokenizer.decode(
    #         self.tokenizer.convert_tokens_to_ids(piece),
    #         skip_special_tokens=True) for piece in pieces]

    #     return text_pieces
    def gpt_chat_completion(self, prompt):
        # print(len(prompt))
        conversation_folder = './logs/conversation/'
        if not os.path.exists(conversation_folder):
            os.makedirs(conversation_folder)
        messages = [{"role": "user", "content": prompt},]
        with open(f'{conversation_folder}_{str(self.start_time)}_{self.model_name}.txt', 'a') as f:
            f.write(f'\n *****model_type: {self.model_name}******* ')
            
            f.write(f"\n**len_input**: {len(prompt.split(' '))} ")
            
            f.write(f'\n**input**: {prompt}')
            # inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        
        # inputs = inputs.to(self.model.device)
        # outputs = self.model.generate(inputs, max_new_tokens=1500,  pad_token_id=self.tokenizer.eos_token_id)
        # text = self.tokenizer.batch_decode(outputs)[0]
        text = self.pipe(messages, **self.generation_args)
        
        output = text[0]['generated_text']
        
        
        with open(f'{conversation_folder}_{str(self.start_time)}_{self.model_name}.txt', 'a') as f:
            f.write(f'**output**: {output}\n ---------------------- \n')
        # print('output:', text)
        # print('len_text',len(text))
        if self.model_name == 'optimizer':
            # print(len('optimizer_response:'), text)
            # print('response:', text)
            gpu_mem_folder = './logs/gpu_mem/'
            if not os.path.exists(gpu_mem_folder):
                os.makedirs(gpu_mem_folder)
            with open(f'{gpu_mem_folder}_{str(self.start_time)}.txt', 'a') as f:
                if self.model_name == 'optimizer':
                    f.write(f'\n ---------------------- \n')
                f.write(f'\n input_len: {len(prompt)}\n')
                # Get the first available GPU
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    if int(gpu.id) > 4:
                        f.write(f"gpu: {gpu.id}, load: {gpu.load * 100}%, free memory: {gpu.memoryFree}MB, used memory: {gpu.memoryUsed}MB\n")
                if self.model_name == 'optimizer':
                    f.write(f'\n ---------------------- \n')
                    
                    
            
        return output
        
        