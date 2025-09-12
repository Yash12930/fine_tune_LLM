import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 1. App Configuration ---
st.set_page_config(page_title="Conversational AI Assistant", layout="wide")
st.title("Student & Professional Email Assistant 🤖")
st.write("This agent helps you draft replies to messages. Provide the message you received and your goal for the reply.")

# --- 2. Model Loading ---
# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_specialist_model():
    # Define the paths to your models on the Hugging Face Hub
    base_model_hub_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    adapter_hub_id = "Yash12930/tinellama_student_assistant" 
    
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_hub_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_hub_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the LoRA adapters from the Hub
    model = PeftModel.from_pretrained(model, adapter_hub_id)
    # Merge the adapters for faster inference
    model = model.merge_and_unload()
    
    print("Fine-tuned model loaded successfully.")
    return model, tokenizer

# Load the models when the app starts
try:
    model, tokenizer = load_specialist_model()
except Exception as e:
    st.error(f"Error loading the model. Please make sure you have updated the 'adapter_hub_id' in the code. Details: {e}")
    st.stop()


# --- 3. The Multi-Agent Workflow ---

# The specialized tool (your fine-tuned model)
def generate_reply_tool(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# AGENT 1: THE PLANNER
def planner(user_request: str, incoming_message: str) -> dict:
    """
    The "thinking" agent. It takes the user's goal and creates a detailed plan (a prompt).
    """
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

# AGENT 2: THE EXECUTOR
def executor(plan: dict) -> str:
    """
    The "doing" agent. It takes the plan and executes it with the specialized tool.
    """
    if plan["task"] == "generate_reply":
        prompt = plan["prompt"]
        full_response = generate_reply_tool(prompt)
        
        # Clean up the output to get only the assistant's reply
        assistant_part = full_response.split("### Assistant:")[-1]
        cleaned_response = assistant_part.split("### Human:")[0].strip()
        
        return cleaned_response
    else:
        return "Executor doesn't know how to handle this task."


# --- 4. Main App Interface ---
st.header("Draft a Reply")

col1, col2 = st.columns(2)

with col1:
    incoming_message = st.text_area("Incoming Message:", "Hey, I missed the lecture on Tuesday. Can I get your notes?", height=150)

with col2:
    user_request = st.text_area("Your Goal for the Reply:", "A polite message saying I'll send them this evening.", height=150)

if st.button("Generate Reply", type="primary"):
    if not incoming_message or not user_request:
        st.warning("Please fill in both fields.")
    else:
        with st.spinner("Agent is thinking and drafting a reply..."):
            # The agent collaboration is run here
            generated_plan = planner(user_request, incoming_message)
            final_response = executor(generated_plan)
            
            st.success("Generated Reply:")
            st.markdown(f"> {final_response}")

