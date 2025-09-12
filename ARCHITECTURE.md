AI Agent Architecture
1. Overview
This project is an AI agent designed to automate the task of drafting replies to messages, specifically for student and professional academic contexts. It follows a multi-agent Planner-Executor design pattern to structure its reasoning and execution flow.

2. Components
Planner Agent: A Python function (planner) that acts as the "thinking" component. It receives a high-level user goal and an incoming message, and its responsibility is to create a detailed, structured prompt. This represents the "planning" stage.

Executor Agent: A Python function (executor) that acts as the "doing" component. It receives the structured plan from the Planner and executes the task by calling the appropriate specialized tool.

Specialized Tool (Fine-Tuned Model): The core tool is a TinyLlama/TinyLlama-1.1B-Chat-v1.0 model that has been fine-tuned using LoRA. This model is an expert at the specific task of generating coherent, helpful responses for student-related conversations.

3. Interaction Flow
A user provides a high-level goal (e.g., "a polite decline") and an incoming message.

The Planner Agent creates a detailed prompt formatted for the specialized tool.

The Executor Agent receives this prompt and calls the Fine-Tuned Model Tool.

The model generates a response, which the Executor cleans up and returns as the final output.

4. Design Choices
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 was chosen for its excellent balance of performance and efficiency, which allows for very fast fine-tuning on free, consumer-grade hardware like Google Colab.

Fine-Tuning: The model was fine-tuned for task specialization. The goal was to adapt its style to be more effective at generating student-related conversational replies than the base model, thereby improving its reliability and the quality of its outputs for this specific use case.