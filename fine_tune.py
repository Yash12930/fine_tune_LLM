import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

def main():
    """
    Main function to run the fine-tuning process.
    """
    # --- 1. Configuration ---
    # Ensure your Hugging Face token is set as an environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")

    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    final_model_path = "./tiny-conversational-assistant-final"

    # --- 2. Load and Filter Dataset ---
    print("Loading and filtering the Dolly-15k dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train", token=hf_token)

    keywords = [
        "email", "letter", "note", "message", "subject:", "draft a reply", "write a response", "respond to",
        "professor", "teacher", "TA", "university", "academic", "college", "lecture", "course", "exam", "thesis",
        "professional", "colleague", "manager", "interview", "application", "job", "internship", "recommendation",
        "assignment", "deadline", "reschedule", "request for", "follow up", "feedback", "inquiry", "apologize", "decline"
    ]

    def is_relevant(example):
        text_to_check = (example['instruction'] + " " + example['category']).lower()
        return any(keyword in text_to_check for keyword in keywords)

    filtered_dataset = dataset.filter(is_relevant)

    if len(filtered_dataset) > 1000:
        filtered_dataset = filtered_dataset.shuffle(seed=42).select(range(1000))

    print(f"✅ Created a targeted dataset with {len(filtered_dataset)} relevant examples.")

    # --- 3. Format and Tokenize Dataset ---
    def format_dolly_prompt(example):
        instruction = example["instruction"]
        context = example["context"]
        response = example["response"]
        if context:
            return f"### Instruction:\\n{instruction}\\n\\n### Context:\\n{context}\\n\\n### Response:\\n{response}"
        else:
            return f"### Instruction:\\n{instruction}\\n\\n### Response:\\n{response}"

    def create_text_column(example):
        return {"text": format_dolly_prompt(example)}

    processed_dataset = filtered_dataset.map(create_text_column, remove_columns=list(filtered_dataset.features))

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)

    tokenized_dataset = processed_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print("✅ Dataset formatted and tokenized.")

    # --- 4. Load Model with 4-bit Quantization ---
    print("Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"
    )
    print("✅ Base model loaded.")

    # --- 5. Configure LoRA ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = get_peft_model(base_model, lora_config)
    print("✅ PEFT model configured with LoRA.")

    # --- 6. Set Up Trainer ---
    training_args = TrainingArguments(
        output_dir="./tiny-conversational-assistant-training",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=1,
        save_strategy="epoch",
        logging_steps=10,
        fp16=True,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # --- 7. Start Fine-Tuning ---
    print("\\nStarting fine-tuning...")
    trainer.train()
    print("✅ Fine-tuning finished!")

    # --- 8. Save the Final Model ---
    trainer.save_model(final_model_path)
    print(f"✅ Model adapters saved to {final_model_path}")
    print("You can now zip this folder for easy distribution if needed.")


if __name__ == "__main__":
    main()