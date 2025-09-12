# AI Agent Architecture

## 1. Overview
[cite_start]This project is an AI agent that automates the task of drafting replies to messages, acting as a conversational assistant for student-related topics. [cite: 3] [cite_start]It uses a multi-agent Planner-Executor design pattern to structure its reasoning and execution flow. [cite: 14]

## 2. Components
* **Planner Agent:** A Python function (`planner`) that takes a high-level user goal and an incoming message. It "plans" the task by creating a detailed, structured prompt for the specialist tool.
* **Executor Agent:** A Python function (`executor`) that receives the structured plan from the Planner. It "executes" the plan by calling the fine-tuned model.
* **Specialized Tool (Fine-Tuned Model):** A `TinyLlama-1.1B-Chat-v1.0` model fine-tuned with LoRA. [cite_start]This model is specialized in generating coherent, helpful responses for student-related conversations. [cite: 5]

## 3. Interaction Flow
1.  A user provides a high-level goal and an incoming message.
2.  The **Planner Agent** creates a detailed prompt.
3.  The **Executor Agent** receives this prompt and calls the **Fine-Tuned Model**.
4.  The model generates a response, which the Executor returns as the final output.

## 4. Design Choices
* **Model:** `TinyLlama-1.1B-Chat-v1.0` was chosen for its strong performance and small size, which allows for very fast and efficient fine-tuning.
* [cite_start]**Fine-Tuning:** The model was fine-tuned for **task specialization**. [cite: 11] [cite_start]The goal was to adapt its style to be more effective at generating student-related conversational replies than the base model, thereby improving its reliability. [cite: 11]