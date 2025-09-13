# AI Agent Architecture Report

## 1. Project Overview
**Agent Name:** Conversational Assistant for Formal Communication  

**Objective:**  
The goal of this project is to build an AI agent that helps users draft polite, effective, and context-aware responses in academic and professional settings. The assistant focuses on formal and semi-formal communication, making it useful for tasks such as replying to emails, requesting extensions, or politely declining invitations.  

---

## 2. Agent Architecture: Single-Agent System
This system is designed as a **single-agent architecture**, chosen for its balance of simplicity, efficiency, and suitability for CPU-only environments.  

**Workflow:**  ```User Input (Message + Request) → Optimized Prompt Template → Fine-Tuned LLM → Generated Response → User```

### Architectural Components

- **User Input Interface**  
  The user provides:  
  - **incoming_message:** the original message they received (or "N/A" if they are starting the conversation).  
  - **user_request:** their goal for the response (e.g., “a polite decline,” “request an extension”).  

- **Prompt Engineering Layer**  
  The model was fine-tuned and is prompted using a consistent `### Human: / ### Assistant: structure`.
  This clear, turn-based format helps the model reliably understand its task and generate the appropriate response.

- **Core Logic (AI Agent)**  
  - **Model:** Fine-tuned **TinyLlama/TinyLlama-1.1B-Chat-v1.0**  
  - **Task:** End-to-end text generation. The model takes the structured input and produces a polished response in one step.  

- **Output Generation**  
  The raw model output is cleaned and decoded so the user only sees the final, professional message.  

---

## 3. Why a Single-Agent System?
I considered a multi-agent design (e.g., separate “planner” and “executor”), but ultimately chose the single-agent approach. The main reasons are:  

- **Efficiency and Deployment Feasibility**  
  Since this project targets CPU-only environments, efficiency is critical. A single-agent system reduces inference calls by half, making it faster and cheaper to run.  

- **Task Simplicity**  
  Drafting formal communication is a relatively linear process. The fine-tuned TinyLlama model demonstrated strong ability to handle both planning and generation within a single step.  

- **Ease of Development and Maintenance**  
  A single-agent system is easier to build, debug, and maintain. This makes it more reliable, especially at the prototype stage.  

---

## 4. Summary
The Conversational Assistant is built on a streamlined **single-agent architecture** that emphasizes clarity, efficiency, and practicality. By leveraging prompt engineering with TinyLlama and a clean input-to-output pipeline, the system can generate polished, context-aware responses suitable for formal communication, while remaining lightweight enough for CPU-based deployment.  
