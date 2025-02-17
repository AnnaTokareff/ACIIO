import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Optional

import logging

logging.basicConfig(level=logging.INFO)
# Выполнение команды для логина
from huggingface_hub import login
login(token="")

def setup_feedback_prompt(dialogue: str, summary: str, prompt: str) -> str:
    return f"""Look at the dialogue and summary and evaluate the summary based on the following criteria\
        and provide a score from 1 to 5 for each criterion. Add a brief explanation for each score to clarify the reasoning. \
        Be strict in your evaluation!

    1. Accuracy (1-5): How well does the summary capture the key events and facts of the dialogue?
    2. Conciseness (1-5): How well does the summary avoid unnecessary information?
    3. Coherence (1-5): Is the summary logically structured and easy to follow?
    4. Completeness (1-5): Are all important aspects of the dialogue covered?
    5. Readability (1-5): Is the summary easy and pleasant to read?
    6. Relevance (1-5): Does the summary stay on topic with the main theme of the dialogue?
    7. Informativeness (1-5): Does the summary provide sufficient information from the dialogue?
    8. Engagement (1-5): Is the summary engaging and able to hold the reader's interest?

    After evaluating each criterion, provide a total score (sum of all criteria) and based on the given scores provide concise feedback \
    on how to improve the INITIAL PROMPT to create the best summary. \
    Please limit the feedback to a MAXIMUM of 15-20 tokens. 

    Dialogue: {dialogue}

    Summary: {summary}

    Initial Prompt: {prompt}
    
    Make sure to thoroughly look at the original dialogue and the generated summary. If the summary doesn't make any\
    sense, then the scores should be relevant.

    Your evaluation:
    [Provide scores for each criterion, Total Score, and Feedback here]
    
    Example Output:
        Accuracy (4): good
        Conciseness (3): ok
        Coherence (5): perfect
        Completeness (4): good
        Readability (5): well read
        Relevance (4): relevant
        Informativeness (4): quite informative
        Engagement (3): lacks engagement

        Total Score: 32/40

        Feedback: Focus on improving the conciseness by adding restrictions on the prompt length.
    """""

def extract_scores(feedback: str) -> Dict[str, Optional[int]]:
    criteria = ["Accuracy", "Conciseness", "Coherence", "Completeness", "Readability", "Relevance", "Informativeness", "Engagement"]
    scores = {}

    for criterion in criteria:
        pattern = rf"{criterion}\s*\(?(\d+)\)?:"
        match = re.search(pattern, feedback, re.IGNORECASE)
        if match:
            scores[criterion] = int(match.group(1))
        else:
            logging.warning(f"Could not find score for {criterion} in feedback.")
            scores[criterion] = None
    
    total_score_pattern = r"Total Score:?\s*(\d+)(?:/40)?"
    total_score_match = re.search(total_score_pattern, feedback, re.IGNORECASE)
    if total_score_match:
        scores["Total"] = int(total_score_match.group(1))
    else:
        logging.warning("Total Score not found in feedback.")
        scores["Total"] = None
    
    return scores

def extract_feedback(feedback: str) -> str:
    feedback_pattern = r"Feedback:?\s*(.*?)(?:\n|$)"
    match = re.search(feedback_pattern, feedback, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        logging.warning("Could not find feedback in the response.")
        return ""

def generate_feedback(dialogue: str, summary: str, initial_prompt: str, model, tokenizer, max_tokens: int = 500) -> Dict:
    try:
        logging.info("Setting up feedback prompt")
        messages = [
            {"role": "system", "content": "You are an expert dialogue summarization evaluator. Your task is to provide detailed, structured feedback on summaries, including numerical scores for specific criteria."},
            {"role": "user", "content": setup_feedback_prompt(dialogue, summary, initial_prompt)}
        ]

        logging.info("Applying chat template")
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
            temperature=0.3,
            top_p=0.5,
        )
        
        logging.info("Decoding feedback")
        response = outputs[0][input_ids.shape[-1]:]
        feedback = tokenizer.decode(response, skip_special_tokens=True)
        
        logging.info(f"Raw feedback generated by the model: {feedback}")

        logging.info("Extracting scores and feedback text")
        scores = extract_scores(feedback)
        feedback_text = extract_feedback(feedback)

        if not scores or all(score is None for score in scores.values()):
            logging.warning("No valid scores extracted. Using default scores.")
            scores = {criterion: 1 for criterion in ["Accuracy", "Conciseness", "Coherence", "Completeness", "Readability", "Relevance", "Informativeness", "Engagement"]}
            scores["Total"] = 8

        if not feedback_text:
            logging.warning("No feedback text extracted. Using default feedback.")
            feedback_text = "The input was too short or nonsensical to provide a meaningful evaluation."

        return {
            "annotations": scores,
            "total_score": scores["Total"],
            "feedback": feedback_text
        }
    except Exception as e:
        logging.error(f"Error in generate_feedback: {str(e)}")
        return {
            "annotations": {criterion: 1 for criterion in ["Accuracy", "Conciseness", "Coherence", "Completeness", "Readability", "Relevance", "Informativeness", "Engagement"]},
            "total_score": 8,
            "feedback": "An error occurred during feedback generation."
        }

def compute_semantic_similarity(predicted_summary: str, reference_summary: str, model) -> float:
    embeddings1 = model.encode(predicted_summary, convert_to_tensor=True)
    embeddings2 = model.encode(reference_summary, convert_to_tensor=True)
    
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()
    return similarity
