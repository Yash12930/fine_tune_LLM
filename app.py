import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

# --- 1. App Configuration ---
st.set_page_config(page_title="Conversational AI Assistant", layout="wide")
st.title("Student & Professional Email Assistant 🤖")
st.write("This agent helps you draft replies to messages. Provide the message you received and your goal for the reply.")

# --- 2. Model Loading ---
@st.cache_resource
def load_specialist_model():
    merged_model_hub_id = "Yash12930/tinellama_student_assistant_merged" 

    tokenizer = AutoTokenizer.from_pretrained(merged_model_hub_id)
    model = AutoModelForCausalLM.from_pretrained(
        merged_model_hub_id,
        torch_dtype=torch.float32, # float32 is best for CPU
        trust_remote_code=True,
    ).to("cpu")
    
    print("Pre-merged model loaded successfully on CPU.")
    return model, tokenizer

# Load the model and tokenizer
try:
    model, tokenizer = load_specialist_model()
except Exception as e:
    st.error(f"Error loading the model. Details: {e}")
    st.stop()

# --- 3. The Multi-Agent Workflow ---
def planner(user_request: str, incoming_message: str) -> dict: #
    prompt = (
        f"### Human:\nI received the following message: \"{incoming_message}\". "
        f"Please help me draft a response that is {user_request}.\n\n"
        f"### Assistant:\n"
    ) 
    plan = {
        "task": "generate_reply",
        "prompt": prompt
    } 
    return plan

# --- 4. Main App Interface ---
st.header("Draft a Reply") 
col1, col2 = st.columns(2) 

with col1:
    incoming_message = st.text_area("Incoming Message:", "N/A", height=150) #
with col2:
    user_request = st.text_area("Your Goal for the Reply:", "i missed class on tuesday, help with attendance", height=150) #

if st.button("Generate Reply", type="primary"): 
    if not incoming_message or not user_request:
        st.warning("Please fill in both fields.") 
    else:
        with st.spinner("Agent is thinking and drafting a reply..."): 
            generated_plan = planner(user_request, incoming_message) 
            prompt = generated_plan.get("prompt") 

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) 
            
            inputs = tokenizer(prompt, return_tensors="pt").to("cpu") 

            generation_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            ) 

            thread = Thread(target=model.generate, kwargs=generation_kwargs) 
            thread.start() 

            st.success("Generated Reply:") 
            st.write_stream(streamer) 