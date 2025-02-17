import torch
import pronouncing
import logging
from typing import Dict, Optional, List
import re

import sacrebleu
from bert_score import BERTScorer
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from actor import postprocess_translation

from huggingface_hub import login
login(token="")

torch.cuda.empty_cache()


def setup_lyrics_feedback_prompt(source_text: str, translation: str, initial_prompt: str) -> str:
    return f"""
    Look at the original lyrics in English and its translation to French and evaluate the translation based on the following criteria\
    and provide a score from 1 to 5 for each criterion. \
  
    1. Semantic Accuracy (1-5): How well does the translation preserve the original meaning and emotions?
    2. Musicality (1-5): Does the translation maintain the rhythm, flow, and musicality?
    3. Poetic Quality (1-5): Does the translation retain metaphors, and poetic devices?
    4. Cultural Adaptation (1-5): Is the translation culturally appropriate?
    5. Emotional Impact (1-5): Does the translation maintain the emotional impact of the original?
    6. Naturalness (1-5): How natural does the translation sound in the target language?
    7. Formatting (1-5): Preservation of formatting and structure?
    8. Language diversity (1-5): How creative and rich the language of translation?
    
    After evaluating each criterion, provide a total score (sum of all criteria) and based on the given scores provide concise feedback \
    on how to improve the INITIAL PROMPT to create the best summary. \
    Please limit the feedback to a MAXIMUM of 20 tokens.
    
    Initial Prompt: {initial_prompt}

    Original Lyrics:
    {source_text}

    Translation:
    {translation}
    
    Make sure to thoroughly look at the original lyrics and the generated translation to give relevant scores. Remember that \
    the translation is supposed to be in FRENCH. Be strict with that!
    
    Provide scores as follows:
    
    Semantic Accuracy (X): [explanation]
    Musicality (X): [explanation]
    Poetic Quality (X): [explanation]
    Cultural Adaptation (X): [explanation]
    Emotional Impact (X): [explanation]
    Naturalness (X): [explanation]
    Formatting (X): [explanation]
    Language diversity (X): [explanation]

    Total Score: XX/40

    Feedback: [how to improve prompt to make better translation]
    """
    
def extract_scores(feedback: str) -> Dict[str, int]:
    criteria = [
        "Semantic Accuracy", "Musicality", "Poetic Quality", "Cultural Adaptation",
        "Emotional Impact", "Naturalness", "Formatting", "Language diversity"
    ]
    
    scores = {}
    
    # Use more robust regex pattern
    for criterion in criteria:
        pattern = rf"{criterion}\s*\((\d+)\)"
        match = re.search(pattern, feedback, re.IGNORECASE)
        # Default to 1 if no score found to avoid None values
        scores[criterion] = int(match.group(1)) if match else 1
    
    # Calculate total score only from found values
    total = sum(scores.values())
    scores["Total"] = total
    
    return scores

def extract_feedback(feedback: str) -> str:
    feedback_pattern = r"Feedback:\s*([^\n]+)"
    match = re.search(feedback_pattern, feedback, re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    logging.warning("Could not find feedback in the response. Using default feedback.")
    return "Unable to extract specific feedback from the evaluation."


def compute_translation_metrics(predicted_translation: str, reference_translations: List[str], target_lang: str) -> Dict[str, float]:
    """Compute BLEU and BERTScore metrics with proper error handling."""
    try:
        # Проверяем входные данные
        if not predicted_translation or not reference_translations:
            return {'bleu': 0.0, 'bertscore': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}}
        
        # Защита от пустых референсных переводов
        reference_translations = [ref for ref in reference_translations if ref]
        if not reference_translations:
            return {'bleu': 0.0, 'bertscore': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}}

        # Применяем постобработку к переводу
        processed_translation = postprocess_translation(predicted_translation)
        processed_references = [postprocess_translation(ref) for ref in reference_translations]
        
        try:
            bleu = sacrebleu.corpus_bleu([processed_translation], [processed_references])
            bleu_score = bleu.score / 100
        except Exception as e:
            logging.warning(f"Error computing BLEU score: {e}")
            bleu_score = 0.0
        
        try:
            bert_scorer = BERTScorer(lang=target_lang, rescale_with_baseline=True)
            P, R, F1 = bert_scorer.score([processed_translation] * len(processed_references), processed_references)
            bertscore = {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item()
            }
        except Exception as e:
            logging.warning(f"Error computing BERTScore: {e}")
            bertscore = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        return {'bleu': bleu_score, 'bertscore': bertscore}
    
    except Exception as e:
        logging.error(f"Error computing translation metrics: {str(e)}")
        return {'bleu': 0.0, 'bertscore': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}}

def compute_rhythm_similarity(text1: str, text2: str) -> float:
    """Compute rhythm similarity with proper error handling."""
    try:
        if not text1 or not text2:
            return 0.0

        def get_rhythm_pattern(text):
            try:
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                return [len(line.split()) for line in lines]
            except Exception as e:
                logging.warning(f"Error getting rhythm pattern: {e}")
                return []
    
        pattern1 = get_rhythm_pattern(text1)
        pattern2 = get_rhythm_pattern(text2)
        
        # Проверяем на пустые паттерны
        if not pattern1 or not pattern2:
            return 0.0
        
        max_len = max(len(pattern1), len(pattern2))
        if max_len == 0:
            return 0.0
        
        # Безопасное вычисление максимального значения
        max_words = max(max(pattern1 or [0]), max(pattern2 or [0]), 1)
        
        # Выравниваем длины паттернов
        pattern1.extend([0] * (max_len - len(pattern1)))
        pattern2.extend([0] * (max_len - len(pattern2)))
        
        differences = sum(abs(a - b) for a, b in zip(pattern1, pattern2))
        similarity = 1.0 - (differences / (max_len * max_words))
        
        return max(0.0, min(1.0, similarity))  # Ограничиваем результат диапазоном [0, 1]
    
    except Exception as e:
        logging.error(f"Error computing rhythm similarity: {str(e)}")
        return 0.0

def compute_rhyme_similarity(text1: str, text2: str) -> float:
    """Compute rhyme similarity with proper error handling."""
    try:
        if not text1 or not text2:
            return 0.0

        def clean_text(text: str) -> str:
            try:
                return re.sub(r'[^\w\s]', '', text.lower())
            except Exception as e:
                logging.warning(f"Error cleaning text: {e}")
                return text.lower()
    
        def get_last_rhyme(line: str) -> str:
            try:
                words = line.split()
                if not words:
                    return ''
                    
                last_word = words[-1]
                phones = pronouncing.phones_for_word(last_word)
                if phones:
                    rhymes = pronouncing.rhyming_part(phones[0])
                    return rhymes or last_word[-2:]
                return last_word[-2:]
            except Exception as e:
                logging.warning(f"Error getting last rhyme: {e}")
                return ''
    
        def get_rhyme_pattern(text: str) -> List[str]:
            try:
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                if not lines:
                    return []
                    
                cleaned_lines = [clean_text(line) for line in lines]
                return [get_last_rhyme(line) for line in cleaned_lines if line]
            except Exception as e:
                logging.warning(f"Error getting rhyme pattern: {e}")
                return []
    
        # Get rhyme patterns
        pattern1 = get_rhyme_pattern(text1)
        pattern2 = get_rhyme_pattern(text2)
        
        # Проверяем на пустые паттерны
        if not pattern1 or not pattern2:
            return 0.0
        
        # Сравниваем паттерны
        min_len = min(len(pattern1), len(pattern2))
        if min_len == 0:
            return 0.0
            
        matches = sum(1 for a, b in zip(pattern1[:min_len], pattern2[:min_len]) 
                     if a and b and a == b)
        similarity = matches / min_len
        
        return round(max(0.0, min(1.0, similarity)), 3)  # Округляем и ограничиваем диапазоном [0, 1]
        
    except Exception as e:
        logging.error(f"Error computing rhyme similarity: {str(e)}")
        return 0.0


def generate_feedback(source_text: str, translation: str, initial_prompt: str,
                     reference_translations: List[str], source_lang: str, target_lang: str,
                     model, tokenizer, max_tokens: int = 600) -> Dict:
    try:
        # Set up tokenizer configuration
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        logging.info("Setting up feedback prompt")
        messages = [
            {"role": "system", "content": "You are an expert translation evaluator. Your task is to provide detailed, structured feedback on translated lyrics, including numerical scores for specific criteria."},
            {"role": "user", "content": setup_lyrics_feedback_prompt(source_text, translation, initial_prompt)}
        ]        
        
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        attention_mask = torch.ones_like(input_ids)

        logging.info("Generating feedback")
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.6,
        )

        torch.cuda.empty_cache()
        response = outputs[0][input_ids.shape[-1]:]
        raw_feedback = tokenizer.decode(response, skip_special_tokens=True)
        detailed_feedback = extract_feedback(raw_feedback)
        scores = extract_scores(raw_feedback)
        
        # Compute standard metrics with error handling
        try:
            metrics = compute_translation_metrics(translation, reference_translations, target_lang)
        except Exception as metrics_error:
            logging.error(f"Error computing metrics: {str(metrics_error)}")
            metrics = {
                "bleu": 0.0,
                "bertscore": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            }
        
        # Compute rhythm and rhyme similarities
        try:
            reference_translation = reference_translations[0] if reference_translations else ""
            rhythm_similarity = compute_rhythm_similarity(translation, reference_translation)
            rhyme_similarity = compute_rhyme_similarity(translation, reference_translation)
        except Exception as lyrics_metrics_error:
            logging.error(f"Error computing lyrics metrics: {str(lyrics_metrics_error)}")
            rhythm_similarity = 0.0
            rhyme_similarity = 0.0

        return {
            "source_text": source_text,  # Add source text separately
            "translation": translation,   # Store only the translation
            "annotations": scores,
            "total_score": scores["Total"],
            "feedback": detailed_feedback,
            "automatic_metrics": metrics,
            "rhythm_similarity": rhythm_similarity,
            "rhyme_pattern_similarity": rhyme_similarity
        }
        
    except Exception as e:
        logging.error(f"Error in generate_feedback: {str(e)}", exc_info=True)
        default_scores = {criterion: 1 for criterion in [
            "Semantic Accuracy", "Musicality", "Poetic Quality", "Cultural Adaptation",
            "Emotional Impact", "Naturalness", "Formatting", "Language diversity"
        ]}
        default_scores["Total"] = sum(default_scores.values())
        
        return {
            "source_text": source_text,  # Include source text even in error case
            "translation": translation,   # Include translation even in error case
            "annotations": default_scores,
            "total_score": default_scores["Total"],
            "feedback": "Error during feedback generation.",
            "automatic_metrics": {"bleu": 0.0, "bertscore": {"precision": 0.0, "recall": 0.0, "f1": 0.0}},
            "rhythm_similarity": 0.0,
            "rhyme_pattern_similarity": 0.0
        }
