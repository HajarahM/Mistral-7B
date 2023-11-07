# Install dependencies: !pip install git+https://github.com/huggingface/transformers torch accelerate bitsandbytes langchain
# Import libraries 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Download Model and Transformer and set variables

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", load_in_4bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")