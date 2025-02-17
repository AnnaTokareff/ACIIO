from typing import List, Dict, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch

from huggingface_hub import login
import logging

logging.basicConfig(level=logging.INFO)


login(token="")

def setup_simple_prompt() -> str:
    return "Summarize the dialogue. Ensure the summary starts with 'Summary:' and it is short (not longer than 20 tokens).\n\n"

def preprocess_context(context: str) -> str:
    lines = context.split("\n")
    processed_lines = [line.split(":", 1)[-1].strip() for line in lines]
    return " ".join(processed_lines)

def make_query(context: str, prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context}
    ]

# def extract_summary(response: str) -> str:
#     summary_part = response.strip()
#     return summary_part

def extract_summary(response: str) -> str:
    response = response.strip()
    start_index = response.find("Summary:")
    if start_index != -1:
        # Извлекаем текст после "Summary:" и удаляем лишние пробелы или переносы строк
        summary = response[start_index + len("Summary:"):].strip()
        return summary
    else:
        return response.strip()
        
    
def save_results(results: List[Dict[str, Union[str, List[Dict[str, str]]]]], output_path: str) -> None:
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)


def generate_response(context: str, prompt: str, model, tokenizer, max_new_tokens: int = 60, temperature: float = 0.2, top_p: float = 0.6) -> str:
    try:
        logging.info("Preprocessing context and creating query")
        context = preprocess_context(context)
        messages = make_query(context, prompt)
        
        logging.info("Applying chat template")
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        attention_mask = torch.ones_like(input_ids)
        
        logging.info("Generating response")
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        
        logging.info("Decoding response")
        response = outputs[0][input_ids.shape[-1]:]
        decoded_response = tokenizer.decode(response, skip_special_tokens=True)
        summary = extract_summary(decoded_response)
        
        logging.info(f"Generated summary: {summary[:50]}...")
        return summary
    
    except Exception as e:
        logging.error(f"Error in generate_response: {str(e)}")
        return ""
