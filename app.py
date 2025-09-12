import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- 1. App Configuration ---
st.set_page_config(page_title="Conversational AI Assistant", layout="wide")
st.title("Student & Professional Email Assistant 🤖")
st.write("This agent helps you draft replies to messages. Provide the message you received and your goal for the reply.")

# --- 2. Model Loading ---
@st.cache_resource
def load_specialist_model():
    base_model_hub_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_hub_id = "Yash12930/tinellama_student_assistant"

    # Load the tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(base_model_hub_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the base model with CPU-safe settings
    model = AutoModelForCausalLM.from_pretrained(
        base_model_hub_id,
        torch_dtype=torch.float32,  # Use float32 for CPU compatibility
        trust_remote_code=True,
    ) # <-- This closing parenthesis was missing

    # Load and merge the LoRA adapters
    model = PeftModel.from_pretrained(model, adapter_hub_id)
    model = model.merge_and_unload()
    
    # Ensure the final model is explicitly on the CPU
    model = model.to("cpu")
    
    print("Fine-tuned model loaded successfully.")
    return model, tokenizer

# Load the models when the app starts
try:
    model, tokenizer = load_specialist_model()
except Exception as e:
    st.error(f"Error loading the model. Details: {e}")
    st.stop()

# --- 3. The Multi-Agent Workflow ---
def generate_reply_tool(prompt: str) -> str:
    # Ensure inputs are also sent to the CPU
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to("cpu")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    ) # <-- This closing parenthesis was missing
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def planner(user_request: str, incoming_message: str) -> dict:
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

def executor(plan: dict) -> str:
    if plan["task"] == "generate_reply":
        prompt = plan["prompt"]
        full_response = generate_reply_tool(prompt)
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
            generated_plan = planner(user_request, incoming_message)
            final_response = executor(generated_plan)
            st.success("Generated Reply:")
            st.markdown(f"> {final_response}")
