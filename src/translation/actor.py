from typing import List, Dict, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer
import re
import json
import torch

from huggingface_hub import login
import logging

logging.basicConfig(level=logging.INFO)

# Выполнение команды для логина
login(token="")
torch.cuda.empty_cache()

def setup_translation_prompt(source_lang: str, target_lang: str) -> str:
    return f'Translate the lyrics from {source_lang} to {target_lang}. Directly start from "Translation:".\n\n'

def make_query(context: str, prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context}
    ]

def extract_translation(response: str) -> str:
    return response.strip()

def preprocess_context(context: str) -> str:
    return context.strip()


def generate_translation(
    source_text: str, 
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int = 1000,
    temperature: float = 0.5,
    top_p: float = 0.7
) -> str:
    """Generate translation with proper text extraction."""
    try:
        logging.info("Preprocessing context and creating query")
        context = preprocess_context(source_text)
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
        translation = extract_translation(decoded_response)
        translation = postprocess_translation(translation)
        
        logging.info(f"Generated translation: {translation[:100]}...")
        return translation
    
    except Exception as e:
        logging.error(f"Error in generate_translation: {str(e)}")
        return ""
    
def postprocess_translation(text: str) -> str:
    """Improved translation post-processing."""
    try:
        # Удаляем любые инструкции или подсказки
        text = re.sub(r'Translation\s*:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Text to translate\s*:', '', text, flags=re.IGNORECASE)
        
        # Нормализация пробелов и пунктуации
        text = re.sub(r'\s+', ' ', text)  # Заменяем множественные пробелы
        text = re.sub(r'\s*([,.!?])', r'\1', text)  # Убираем пробелы перед знаками препинания
        text = re.sub(r'([,.!?])(?!["\'\]\})])', r'\1 ', text)  # Добавляем пробел после знаков препинания
        
        # Восстановление переносов строк для структуры песни
        text = text.replace('. ', '.\n')
        text = text.replace('! ', '!\n')
        text = text.replace('? ', '?\n')
        
        # Удаление пустых строк
        text = '\n'.join(line for line in text.split('\n') if line.strip())

        return text  # Возвращаем обработанный текст

    except Exception as e:
        logging.error(f"Ошибка в postprocess_translation: {str(e)}")
        return text  # Возвращаем текст без изменений в случае ошибки
