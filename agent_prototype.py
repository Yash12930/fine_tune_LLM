import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import HfApi


class LoRAModelManager:
    def __init__(self, base_model_id, adapter_id, merged_repo_id, hf_token):
        self.base_model_id = base_model_id
        self.adapter_id = adapter_id
        self.merged_repo_id = merged_repo_id
        self.hf_token = hf_token

        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable not found.")

    def load_fine_tuned_model(self):
        """Loads the base model and applies a LoRA adapter from the Hub."""
        print(f"Loading base model: {self.base_model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"Loading and merging adapter: {self.adapter_id}")
        model = PeftModel.from_pretrained(model, self.adapter_id)
        model = model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        tokenizer.pad_token = tokenizer.eos_token

        print("✅ Fine-tuned model loaded successfully.")
        return model, tokenizer

    def run_inference(self, model, tokenizer, user_request, incoming_message="N/A"):
        """Generates a response from a model based on a prompt."""
        if incoming_message == "N/A":
            prompt = f"### Human:\nPlease help me draft a message that is {user_request}.\n\n### Assistant:\n"
        else:
            prompt = (
                f"### Human:\nI received the following message: \"{incoming_message}\". "
                f"Please help me draft a response that is {user_request}.\n\n"
                f"### Assistant:\n"
            )

        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        cleaned_response = response.split("### Assistant:")[-1].strip()
        return cleaned_response.split("### Human:")[0].strip()

    def merge_and_upload_model(self):
        """Merges a LoRA adapter and uploads the full model to the Hub."""
        print(f"Loading base model: {self.base_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        print(f"Loading and merging adapter: {self.adapter_id}")
        merged_model = PeftModel.from_pretrained(base_model, self.adapter_id)
        merged_model = merged_model.merge_and_unload()

        local_save_path = "./merged_model"
        print(f"Saving merged model locally to '{local_save_path}'...")
        merged_model.save_pretrained(local_save_path)
        tokenizer.save_pretrained(local_save_path)

        print(f"\nUploading to new Hub repository: {self.merged_repo_id}")
        api = HfApi(token=self.hf_token)
        api.create_repo(repo_id=self.merged_repo_id, exist_ok=True)
        api.upload_folder(
            folder_path=local_save_path,
            repo_id=self.merged_repo_id,
            repo_type="model",
        )
        print(f"✅ Upload complete! Your merged model is at: https://huggingface.co/{self.merged_repo_id}")

    def run(self):
        """Main execution block."""
        # --- Option 1: Test the fine-tuned model (loads adapter from Hub) ---
        print("--- Running Option 1: Testing Model with Hub Adapter ---")
        ft_model, tokenizer = self.load_fine_tuned_model()

        test_request = "a polite text declining an invitation because I have to study."
        test_incoming = "Hey, we're all going out for drinks on Friday, you should come!"
        response = self.run_inference(ft_model, tokenizer, test_request, test_incoming)

        print(f"\nTest Request: {test_request}")
        print(f"Generated Response:\n{response}\n")

        # --- Option 2: Merge the adapter and upload a full model ---
        print("\n--- Running Option 2: Merging and Uploading Model ---")
        self.merge_and_upload_model()


# --- Entry Point ---
if __name__ == "__main__":
    BASE_MODEL_HUB_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ADAPTER_HUB_ID = "Yash12930/tinellama_student_assistant"
    MERGED_MODEL_REPO_ID = "Yash12930/tinellama_student_assistant_merged"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    manager = LoRAModelManager(BASE_MODEL_HUB_ID, ADAPTER_HUB_ID, MERGED_MODEL_REPO_ID, HF_TOKEN)
    manager.run()
