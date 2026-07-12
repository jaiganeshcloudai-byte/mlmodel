# ==============================================================================
# FINAL ASSIGNMENT DELIVERABLE: PRODUCTION INFERENCE ENGINE (FIXED)
# ==============================================================================
import os
import torch
from unsloth import FastLanguageModel

def generate_answer(question_text: str) -> str:
    max_seq_length = 2048
    model_path = "/content/drive/MyDrive/domain-ai-assistant-finetuning/models/final_dpo_model"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model path alignment layer: {model_path}")

    # 1. Load the model and force it to recognize the adapter layers
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )
    
    # 2. Shift active network parameters into fast inference decoding mode
    FastLanguageModel.for_inference(model)
    
    # 3. Construct the prompt template matching Stage 2 and Stage 3 exactly
    prompt = f"""You are an expert customer support assistant. Provide clear, accurate, and structured answers.

### Question:
{question_text}

### Response:
"""
    
    # 4. Tokenize and map to GPU
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # 5. Optimized generation parameters to stop the looping behavior entirely
    outputs = model.generate(
        **inputs, 
        max_new_tokens=200, 
        use_cache=True,
        temperature=0.5,           # Balanced creativity and structure
        top_p=0.9,
        repetition_penalty=1.2     # CRITICAL: Penalizes the model from repeating words/phrases
    )
    
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return decoded_output.split("### Response:\n")[-1].strip()

if __name__ == "__main__":
    sample_question = "What is the policy for processing a refund if my package was delayed for weeks and missed my event?"
    print("Executing AI Customer Support Generation Engine...\n" + "="*60)
    try:
        final_answer = generate_answer(sample_question)
        print(f"User Question:\n-> {sample_question}\n\nFinal Assistant Answer:\n-> {final_answer}")
    except Exception as e:
        print(f"🚨 Inference Generation Error: {str(e)}")
    print("="*60)