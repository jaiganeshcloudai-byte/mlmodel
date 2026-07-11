# ==============================================================================
# FINAL ASSIGNMENT DELIVERABLE: PRODUCTION INFERENCE ENGINE
# ==============================================================================
# File Path Location: src/inference.py
# Objective: Load the final preference-aligned DPO model and provide a clean 
#            entry point to handle incoming customer support inquiries.

import os
import torch
from unsloth import FastLanguageModel

def generate_answer(question_text: str) -> str:
    """
    Loads the final DPO-aligned model, wraps the user's question into the exact 
    prompt template used during training, and performs accelerated inference decoding.
    """
    max_seq_length = 2048
    
    # Point to the drive path containing your final DPO weights configuration
    model_path = "/content/drive/MyDrive/domain-ai-assistant-finetuning/models/final_dpo_model"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find the final DPO model at '{model_path}'. "
            "Please verify that Stage 3 has completed and saved successfully."
        )

    # 1. Load the production-aligned weights natively inside Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        dtype = None,            # Auto-detects based on system GPU architecture
        load_in_4bit = True,     # Keeps memory footprint safe on a single T4 GPU
    )
    
    # 2. Shift active network parameters into fast inference decoding mode
    FastLanguageModel.for_inference(model)
    
    # 3. Construct the exact prompt structure used during instruction tuning and alignment
    prompt = f"""You are an expert customer support assistant. Provide clear, accurate, and structured answers.

### Question:
{question_text}

### Response:
"""
    
    # 4. Tokenize inputs and map them directly to the active GPU space
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # 5. Generate completion tokens with optimized decoding parameters
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        use_cache=True,
        temperature=0.3,  # Lower temperature guarantees highly focused, deterministic text
        top_p=0.9
    )
    
    # 6. Decode tokens and isolate the newly generated assistant response payload
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Clean up formatting artifact boundaries to isolate the response text cleanly
    response_segment = decoded_output.split("### Response:\n")[-1]
    return response_segment.strip()

if __name__ == "__main__":
    # Sample domain-specific tracking and refund query
    sample_question = "What is the policy for processing a refund if my package was delayed for weeks and missed my event?"
    
    print("Executing AI Customer Support Generation Engine...")
    print("=" * 60)
    
    try:
        final_answer = generate_answer(sample_question)
        print(f"User Question:\n-> {sample_question}\n")
        print(f"Final Assistant Answer:\n-> {final_answer}")
    except Exception as e:
        print(f"🚨 Inference Generation Error: {str(e)}")

    print("=" * 60)
