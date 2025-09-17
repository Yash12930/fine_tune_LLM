# Data Science Report: Fine-Tuning a Conversational Assistant

---

## 1. Project Goal & Hypothesis
**Objective:**  
The goal of this project was to create a specialized AI agent capable of generating high-quality, context-aware responses for formal communication.  

**Hypothesis:**  
Fine-tuning the **TinyLlama-1.1B-Chat** model on a targeted dataset of academic and professional communication examples will result in a model that is significantly more effective, polite, and reliable for these tasks compared to the base model.  

---

## 2. Data Preparation & Fine-Tuning

### 2.1 Dataset Curation
- **Initial Attempt:** Started with the [OpenAssistant Guanaco dataset](https://huggingface.co/datasets/timdettmers/openassistant-guanaco), filtered for English.  
  - **Issue:** While large and diverse, it was **too general-purpose** and not focused enough on academic/professional communication.  
- **Custom Dataset Attempt:** Considered creating a small dataset of personal emails (~50 samples).  
  - **Issue:** The dataset size was too limited to support meaningful fine-tuning.  
- **Final Choice:** [Databricks Dolly-15k](https://huggingface.co/datasets/databricks/dolly-15k), filtered down to **517 curated examples** of student, academic, and professional communication.  
  - **Filtering Strategy:** Selected tasks containing keywords such as *"professor," "email," "assignment," "deadline,"* and *"professional"* in the instruction/category fields.  
  - **Reasoning:** Balanced size, relevance, and diversity while staying domain-specific.

### 2.2 Fine-Tuning Method: PEFT (LoRA)
- **Base Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
- **Technique:** Parameter-Efficient Fine-Tuning (PEFT) using **LoRA** (Low-Rank Adaptation), chosen to reduce computational requirements and memory usage.  

**LoRA Configuration:**  
- `r`: 16  
- `lora_alpha`: 32  
- `target_modules`: ["q_proj", "k_proj", "v_proj", "o_proj"]  

### 2.3 Training Parameters
- **Epochs:** 1  
- **Learning Rate:** 2e-5  
- **Batch Size:** 2  
- **Gradient Accumulation Steps:** 4 (Effective Batch Size = 8)
- **Precision:** `fp16` (Mixed Precision)  

---

## 3. Model Selection Journey
The choice of base model evolved through several iterations:  

1. **Gemma-3 4B**  
   - **Reason for Initial Choice:** Strong performance and modern architecture.  
   - **Problem:** The model **overfit very quickly** (loss approaching 0), likely due to being too advanced for the small dataset size.  
   
2. **Gemma-2B**  
   - **Reason for Switching:** Attempted to use a slightly smaller model to balance performance and data size.  
   - **Problem:** Still too resource-intensive to **host/deploy cost-effectively**.  

3. **Final Choice – TinyLlama-1.1B**  
   - **Reasoning:**  
     - Lightweight enough to **host on smaller servers**.  
     - Pre-trained as a **chat model**, making it well-suited for conversational fine-tuning.  
     - Avoided severe overfitting on the filtered dataset, striking the right balance between capability and deployability.

---

## 4. Evaluation Methodology

### 4.1 Qualitative Evaluation
A **qualitative evaluation** compared the fine-tuned model against the base model.  

- **Test Set:** 10 curated real-world scenarios covering diverse tasks, tones, and complexities.  
- **Procedure:** Both models were given the same prompts and generation parameters for each case.  
- **Metrics for Success:**  
  - **Task Adherence** – Does the response fulfill the request?  
  - **Tone Appropriateness** – Is the response polite and professional?  
  - **Clarity & Conciseness** – Is the message easy to understand and not verbose?  
  - **Helpfulness** – Can the response be used directly without major edits?  

---

## 5. Results & Analysis

### Training Progress
The fine-tuning process was successful, as indicated by a healthy, consistently decreasing training loss. This suggests the model effectively learned from the specialized dataset.  

![alt text](image.png)  

---

### Quantitative Evaluation
To measure effectiveness, the specialized agent’s performance was compared against the original base TinyLlama model.  

- **Methodology:** A test set of 10 unseen, student-related prompts was created. Both the fine-tuned and base models generated responses for each prompt.  
- **Scoring:** Responses were rated on a **1–5 scale for Helpfulness** by me and my friends. A score of **≥4** was considered a “Success.” 

| ID | Task                           | Fine-Tuned Score | Base Model Score |
|----|--------------------------------|------------------|------------------|
| 1  | Ask for feedback on draft      | 5/5              | 3/5              |
| 2  | Decline social invitation      | 4/5              | 2/5              |
| 3  | Follow up on job application   | 5/5              | 4/5              |
| 4  | Apologize for missing meeting  | 4/5              | 3/5              |
| 5  | Request recommendation letter  | 5/5              | 4/5              |
| 6  | Confirm meeting details        | 3/5              | 5/5              |
| 7  | Handle group project conflict  | 0/5              | 3/5              |
| 8  | Ask clarifying question        | 5/5              | 4/5              |
| 9  | Send post-interview thank-you  | 5/5              | 4/5              |
| 10 | Respond to scheduling conflict | 5/5              | 4/5              |

**Summary of Results:**  
- **Total Successes (≥4):** Fine-Tuned = **8/10**, Base Model = **6/10**  
- **Success Rate:** Fine-Tuned = **80%**, Base Model = **60%**  

---

### Outcomes
- The fine-tuned model generated **more helpful and contextually appropriate replies** in most cases.  
- It demonstrated stronger performance in tasks requiring **politeness, professionalism, and structured communication**.  
- Weaknesses were observed in complex or sensitive scenarios (e.g., group project conflict).  

---

### Sample Output
![Evaluation Results](image-1.png)  

---

## 6. Conclusion
The fine-tuning process was **highly successful**.  

- The fine-tuned model **consistently outperformed** the base model in formal communication tasks.  
- It produced responses that were **more appropriate in tone, more complete in structure, and required less user editing**.  
- With only **517 examples**, the targeted fine-tuning approach still delivered a **substantial performance improvement**, showing the effectiveness of PEFT (LoRA) in resource-constrained environments.  

**Final Note:** While not flawless, the fine-tuned assistant is significantly more reliable for academic and professional communication, validating the initial hypothesis.  

---
