import os
import torch
import streamlit as st
from transformers import AutoTokenizer

try:
    from peft import PeftModel, PeftConfig
except ModuleNotFoundError:
    st.error("🚨 The 'peft' module is missing! Please install it using: `pip install peft`")
    raise SystemExit

# ✅ Load the fine-tuned LoRA model and tokenizer
model_path = "./fine_tuned_model"
map_location = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Check if adapter_config.json exists
adapter_config_path = os.path.join(model_path, "adapter_config.json")
if not os.path.exists(adapter_config_path):
    st.error(f"🚨 Missing 'adapter_config.json' in {model_path}. Ensure the LoRA model is saved correctly.")
    raise SystemExit

# Load LoRA configuration
config = PeftConfig.from_pretrained(model_path)

# Load model with LoRA
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Load LoRA model
model = PeftModel.from_pretrained(base_model, model_path)
model.to(map_location)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate_response(user_input):
    """Generates a response based on user input."""
    input_ids = tokenizer.encode(user_input, return_tensors="pt").to(model.device)
    response_ids = model.generate(input_ids, max_length=100)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

# ✅ Streamlit Web App
st.title("🧠 AI Response Generator")
st.write("Enter a prompt and get a response from the fine-tuned LoRA model.")

user_input = st.text_area("Enter your prompt:", "Type here...")
if st.button("Generate Response"):
    response = generate_response(user_input)
    st.write("**AI Response:**", response)

st.markdown("<p style='text-align:center;'>Powered by Hugging Face, PEFT & Streamlit</p>", unsafe_allow_html=True)
