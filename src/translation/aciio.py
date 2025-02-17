import os
import re
import json
import logging
import numpy as np
from typing import List, Dict, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from datetime import datetime
import sacrebleu
import matplotlib.pyplot as plt
import pronouncing


import nltk
nltk.download('punkt_tab')

from collections import defaultdict
from actor import generate_translation, setup_translation_prompt, postprocess_translation
from critic import generate_feedback, compute_translation_metrics
logging.basicConfig(level=logging.INFO)


def load_data(file_path: str) -> List[Dict]:
    """Load dataset from a JSONL file with proper ID handling."""
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            example = json.loads(line)
            # Добавляем явный ID, если его нет
            example['id'] = i
            data.append(example)
    return data

def save_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def initialize_model(model_name: str) -> Dict[str, Union[AutoModelForCausalLM, AutoTokenizer]]:
    """Initialize model with float16 for balance of speed and memory."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Компромисс между скоростью и памятью
            device_map="auto",
            use_cache=True,  # Включаем кэширование
            low_cpu_mem_usage=True
        )
        model.eval()  # Переключаем в режим инференса
        return {'model': model, 'tokenizer': tokenizer, 'name': model_name}
    except Exception as e:
        logging.error(f"Error loading model {model_name}: {e}")
        raise
    

    
def update_prompt_with_llm(initial_prompt: str, feedback_text: str, model, tokenizer) -> str:
    """Update the translation prompt based on feedback using an LLM."""
    try:
        prompt_update_input = f"""You are a skilled translation prompt engineer. \
            Your task is to improve the current  PROMPT for translation based on the given feedback.

Current Prompt: {initial_prompt}

Feedback: {feedback_text}

REQUIREMENTS:
1. Begin the prompt with "Translate".
2. Focus on the feedback for specific improvements.
3. Make the prompt clear and concise (limit to max 15 tokens).
4. If the current prompt already performs well, make minimal or no changes.
5. Make sure to end the prompt with "Translation:".

Write only the new prompt. Do not include explanations or additional text.

New prompt:"""

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

        inputs = tokenizer(
            prompt_update_input,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=800
        ).to(model.device)

        attention_mask = inputs.input_ids.ne(tokenizer.pad_token_id).long()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        logging.info("Generating updated prompt")

        outputs = model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.5,
            top_p=0.6,
            do_sample=True
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        marker = "New prompt:"
        if marker in generated_text:
            new_prompt = generated_text[generated_text.find(marker) + len(marker):].strip()
        elif "Translate" in generated_text:
            new_prompt = generated_text.find("Translate")
        else:
            new_prompt = generated_text[len(prompt_update_input):].strip()

        # Validate the prompt
        if not new_prompt:
            logging.warning("Empty prompt generated, using default")
            return initial_prompt

        logging.info(f"Generated new prompt: {new_prompt}")
        return new_prompt

    except Exception as e:
        logging.error(f"Error in update_prompt_with_llm: {str(e)}")
        return initial_prompt

def find_best_translation(feedback_results: List[Dict]) -> Dict:
    """Find best translation based on original metrics without normalization."""
    if not feedback_results:
        raise ValueError("No feedback results provided")
        
    best_score = -1
    best_result = None
    
    # Original maximum scores
    max_annotation_score = 40.0  # 8 criteria * 5 points each
    max_automatic_score = 2.0    # BLEU (1.0) + BERTScore (1.0)
    
    # Weights for different components
    weight_annotations = 0.6
    weight_automatic = 0.3
    weight_rhythm_rhyme = 0.1

    for result in feedback_results:
        try:
            # Get annotation scores in original scale (0-5 per criterion)
            scores = result.get('scores', {})
            if 'Total' in scores:
                annotation_score = scores['Total']  # Should be 0-40
            else:
                # Sum of individual criteria scores (each 0-5)
                annotation_score = sum([value for key, value in scores.items() 
                                    if isinstance(value, (int, float)) and key != 'Total'])

            # Keep original scores but weight them appropriately
            weighted_annotation = (annotation_score / max_annotation_score) * weight_annotations

            # Automatic metrics (already 0-1)
            automatic_metrics = result.get('automatic_metrics', {})
            bleu = automatic_metrics.get('bleu', 0)
            bertscore = automatic_metrics.get('bertscore', {}).get('f1', 0)
            weighted_automatic = ((bleu + bertscore) / max_automatic_score) * weight_automatic

            # Rhythm and rhyme (already 0-1)
            rhythm_score = float(result.get('rhythm_similarity', 0))
            rhyme_score = float(result.get('rhyme_pattern_similarity', 0))
            weighted_rhythm_rhyme = ((rhythm_score + rhyme_score) / 2) * weight_rhythm_rhyme

            total_score = weighted_annotation + weighted_automatic + weighted_rhythm_rhyme

            if total_score > best_score:
                best_score = total_score
                best_result = {
                    'translation': result.get('translation', ''),
                    'feedback': result.get('feedback', ''),
                    'score': total_score,
                    'metrics': {
                        'annotations': scores,  # Original scores (0-5 per criterion)
                        'automatic_metrics': automatic_metrics,  # Original BLEU/BERTScore
                        'rhythm_similarity': rhythm_score,  # Original 0-1
                        'rhyme_pattern_similarity': rhyme_score  # Original 0-1
                    }
                }
        except Exception as e:
            logging.warning(f"Error processing result: {str(e)}")
            continue

    if best_result is None:
        raise ValueError("No valid results found after processing all feedback")

    return best_result

def aggregate_scores(feedback_results: List[Dict]) -> Dict[str, float]:
    """Calculate aggregate scores using original metric scales with proper BERTScore handling."""
    if not feedback_results:
        return {
            'total_score': 0.0,
            'annotations': {
                'Semantic Accuracy': 0.0,
                'Musicality': 0.0,
                'Poetic Quality': 0.0,
                'Cultural Adaptation': 0.0,
                'Emotional Impact': 0.0,
                'Naturalness': 0.0,
                'Formatting': 0.0,
                'Language diversity': 0.0
            },
            'automatic_metrics': {
                'bleu': 0.0,
                'bertscore': {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                }
            },
            'rhythm_similarity': 0.0,
            'rhyme_pattern_similarity': 0.0
        }

    total_results = len(feedback_results)
    
    # Accumulators for metrics in original scales
    totals = {
        'annotations': defaultdict(float),      # Will store sum of 0-5 scores per criterion
        'total_annotation_score': 0.0,          # Will store sum of 0-40 total scores
        'bleu': 0.0,                           # Sum of BLEU scores (0-1)
        'bertscore_precision': 0.0,            # Sum of BERTScore precision
        'bertscore_recall': 0.0,               # Sum of BERTScore recall
        'bertscore_f1': 0.0,                   # Sum of BERTScore F1
        'rhythm': 0.0,                         # Sum of rhythm scores (0-1)
        'rhyme': 0.0                           # Sum of rhyme scores (0-1)
    }

    for result in feedback_results:
        # Original annotation scores (0-5 per criterion)
        scores = result.get('scores', {})
        for criterion, score in scores.items():
            if criterion != 'Total' and isinstance(score, (int, float)):
                totals['annotations'][criterion] += score
        
        # Total annotation score (0-40)
        if 'Total' in scores:
            totals['total_annotation_score'] += scores['Total']
        else:
            totals['total_annotation_score'] += sum(score for score in scores.values() 
                                                  if isinstance(score, (int, float)))

        # Original automatic metrics (0-1)
        auto_metrics = result.get('automatic_metrics', {})
        totals['bleu'] += auto_metrics.get('bleu', 0.0)
        
        # Handle complete BERTScore structure
        bertscore = auto_metrics.get('bertscore', {})
        totals['bertscore_precision'] += bertscore.get('precision', 0.0)
        totals['bertscore_recall'] += bertscore.get('recall', 0.0)
        totals['bertscore_f1'] += bertscore.get('f1', 0.0)
        
        # Original rhythm and rhyme scores (0-1)
        totals['rhythm'] += float(result.get('rhythm_similarity', 0.0))
        totals['rhyme'] += float(result.get('rhyme_pattern_similarity', 0.0))

    # Calculate averages in original scales
    return {
        'total_score': totals['total_annotation_score'] / total_results,  # Average 0-40
        'annotations': {
            criterion: score / total_results  # Average 0-5 per criterion
            for criterion, score in totals['annotations'].items()
        },
        'automatic_metrics': {
            'bleu': totals['bleu'] / total_results,  # Average 0-1
            'bertscore': {
                'precision': totals['bertscore_precision'] / total_results,
                'recall': totals['bertscore_recall'] / total_results,
                'f1': totals['bertscore_f1'] / total_results
            }
        },
        'rhythm_similarity': totals['rhythm'] / total_results,  # Average 0-1
        'rhyme_pattern_similarity': totals['rhyme'] / total_results  # Average 0-1
    }
    
    
def save_and_visualize_metrics(output_path: str, iteration: int, 
                             iteration_scores: Dict, 
                             all_iterations_history: List[Dict]) -> None:
    """Save and visualize metrics in their original scales."""
    metrics_dir = os.path.join(output_path, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    current_metrics = {
        'iteration': iteration,
        'timestamp': datetime.now().isoformat(),
        'scores': iteration_scores
    }
    all_iterations_history.append(current_metrics)
    
    # Plot metrics progress
    plt.figure(figsize=(12, 8))
    iterations = [m['iteration'] for m in all_iterations_history]
    
    # Define metrics with their original scales
    metrics_config = {
        'Total Score': {
            'data': [m['scores']['total_score'] for m in all_iterations_history],
            'color': '#1164B4',
            'style': '-',
            'scale': (0, 40)  # Original 0-40 scale
        },
        'BLEU Score': {
            'data': [m['scores']['automatic_metrics']['bleu'] for m in all_iterations_history],
            'color': '#FFA500',
            'style': ':',
            'scale': (0, 1)  # Original 0-1 scale
        },
        'BERTScore': {
            'data': [m['scores']['automatic_metrics']['bertscore']['f1'] for m in all_iterations_history],
            'color': '#FFD700',
            'style': ':',
            'scale': (0, 1)  # Original 0-1 scale
        },
        'Rhythm': {
            'data': [m['scores']['rhythm_similarity'] for m in all_iterations_history],
            'color': '#50C878',
            'style': '--',
            'scale': (0, 1)  # Original 0-1 scale
        }
    }
    
    # Create two y-axes for different scales
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    
    # Plot metrics on appropriate axes
    for name, config in metrics_config.items():
        if config['scale'][1] == 40:  # Total score on primary axis
            ax1.plot(iterations, config['data'], 
                    label=name, color=config['color'], 
                    linestyle=config['style'], linewidth=2,
                    marker='o', markersize=6)
        else:  # Other metrics on secondary axis
            ax2.plot(iterations, config['data'], 
                    label=name, color=config['color'], 
                    linestyle=config['style'], linewidth=2,
                    marker='o', markersize=6)
    
    # Customize axes
    ax1.set_ylim(0, 40)
    ax2.set_ylim(0, 1)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Total Score (0-40)', fontsize=12)
    ax2.set_ylabel('Other Metrics (0-1)', fontsize=12)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              bbox_to_anchor=(1.2, 1), loc='upper left')
    
    plt.title('Progress of Metrics Across Iterations', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'metrics_progress.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics history with original scales
    metrics_history = {
        'iterations': all_iterations_history,
        'metadata': {
            'scales': {
                'total_score': '0-40 (sum of criteria)',
                'individual_criteria': '0-5',
                'automatic_metrics': '0-1',
                'rhythm_rhyme': '0-1'
            }
        }
    }
    
    with open(os.path.join(metrics_dir, 'metrics_history.json'), 'w') as f:
        json.dump(metrics_history, f, indent=4)

        
def save_iteration_file(data: Union[Dict, List], file_path: str):
    """Save data to file for current iteration, overwriting any existing content."""
    try:
        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Записываем данные в файл (перезаписываем существующий)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        logging.info(f"Successfully saved data to {file_path}")
        
    except Exception as e:
        logging.error(f"Error saving file {file_path}: {str(e)}")
        raise



def initialize_json_file(file_path: str, initial_content: Union[List, Dict]) -> None:
    """Initialize a JSON file with given content."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(initial_content, f, indent=4)
    except Exception as e:
        logging.error(f"Error initializing file {file_path}: {e}")
        raise


def run_pace(test_data_path: str, output_path: str, actor_model_name: str, critic_model_name: str,
             num_actors: int, num_critics: int, num_iterations: int, target_total_score: float, patience: int):
    """Run PACE process with proper iteration resume."""
    logging.info("Starting PACE process")
    os.makedirs(output_path, exist_ok=True)

    # Определяем начальную итерацию
    start_iteration = 0
    for i in range(num_iterations):
        iteration_dir = os.path.join(output_path, f'iteration_{i + 1}')
        interim_results_file = os.path.join(iteration_dir, 'interim_results.json')
        full_results_file = os.path.join(iteration_dir, 'full_results.json')
        
        if os.path.exists(full_results_file):
            # Эта итерация завершена, проверяем следующую
            start_iteration = i + 1
            logging.info(f"Found completed iteration {i + 1}")
            continue
        
        if os.path.exists(interim_results_file):
            # Нашли незавершенную итерацию
            start_iteration = i
            logging.info(f"Found incomplete iteration {i + 1}, will resume from here")
            break

    logging.info(f"Starting from iteration {start_iteration + 1}")

    test_data = load_data(test_data_path)
    logging.info(f"Loaded {len(test_data)} examples from {test_data_path}")
    
    actor_model = initialize_model(actor_model_name)
    critic_model = initialize_model(critic_model_name)

    iteration_prompts = {}
    all_iterations_history = []
    best_translations = {}

    # Загружаем результаты предыдущих итераций
    for i in range(start_iteration):
        prev_iteration_dir = os.path.join(output_path, f'iteration_{i + 1}')
        full_results_file = os.path.join(prev_iteration_dir, 'full_results.json')
        if os.path.exists(full_results_file):
            try:
                with open(full_results_file, 'r', encoding='utf-8') as f:
                    prev_data = json.load(f)
                    iteration_prompts.update(prev_data.get('prompts', {}))
                    if 'best_translations' in prev_data:
                        best_translations.update(prev_data['best_translations'])
                logging.info(f"Loaded data from completed iteration {i + 1}")
            except Exception as e:
                logging.error(f"Error loading previous iteration {i + 1}: {e}")

    # Начинаем с нужной итерации
    for iteration in range(start_iteration, num_iterations):
        iteration_dir = os.path.join(output_path, f'iteration_{iteration + 1}')
        os.makedirs(iteration_dir, exist_ok=True)
        logging.info(f"\n{'='*50}\nStarting iteration {iteration + 1}/{num_iterations}\n{'='*50}")
        
        feedback_results = []
        current_iteration_prompts = {}
        
        interim_results_file = os.path.join(iteration_dir, 'interim_results.json')
        
        # Загружаем существующие результаты и отслеживаем прогресс для каждого актора/критика
        processed_combinations = set()
        if os.path.exists(interim_results_file):
            try:
                with open(interim_results_file, 'r', encoding='utf-8') as f:
                    interim_data = json.load(f)
                    feedback_results = interim_data.get('feedback_results', [])
                    current_iteration_prompts = interim_data.get('prompts', {})
                    best_translations = interim_data.get('best_translations', {})
                    
                    # Создаем множество уже обработанных комбинаций
                    for result in feedback_results:
                        combination = (
                            str(result['example_id']),
                            str(result['actor_id']),
                            str(result['critic_id'])
                        )
                        processed_combinations.add(combination)
                        
                logging.info(f"Loaded {len(feedback_results)} existing results")
            except Exception as e:
                logging.error(f"Error loading interim results: {e}")
                interim_data = {'feedback_results': [], 'prompts': {}, 'best_translations': {}}
        else:
            interim_data = {'feedback_results': [], 'prompts': {}, 'best_translations': {}}

        for idx, example in enumerate(test_data):
            example_id = str(idx)
            
            # Проверяем, все ли комбинации актор/критик обработаны для этого примера
            example_fully_processed = all(
                (example_id, str(a), str(c)) in processed_combinations
                for a in range(num_actors)
                for c in range(num_critics)
            )
            
            if example_fully_processed:
                logging.info(f"Example {example_id} fully processed with all actor/critic combinations")
                continue
                
            logging.info(f"Processing example {example_id}")
            
            if iteration == 0 or example_id not in iteration_prompts:
                current_prompt = setup_translation_prompt('English', 'French')
            else:
                current_prompt = iteration_prompts[example_id]
            
            current_iteration_prompts[example_id] = current_prompt

            original_text = example['original_version']
            reference_translation = example['french_version']
            
            example_feedback = []
            
            for actor_id in range(num_actors):
                for critic_id in range(num_critics):
                    # Проверяем, была ли эта комбинация уже обработана
                    combination = (example_id, str(actor_id), str(critic_id))
                    if combination in processed_combinations:
                        logging.info(f"Skipping processed combination: example {example_id}, actor {actor_id}, critic {critic_id}")
                        continue
                        
                    logging.info(f"Processing: example {example_id}, actor {actor_id}, critic {critic_id}")
                    
                    try:
                        translation = generate_translation(
                            original_text, 
                            current_prompt,
                            actor_model['model'],
                            actor_model['tokenizer']
                        )

                        if translation and translation != original_text:
                            feedback = generate_feedback(
                                source_text=original_text,
                                translation=translation,
                                initial_prompt=current_prompt,
                                reference_translations=[reference_translation],
                                source_lang='en',
                                target_lang='fr',
                                model=critic_model['model'],
                                tokenizer=critic_model['tokenizer']
                            )

                            feedback_result = {
                                'example_id': example_id,
                                'actor_id': actor_id,
                                'critic_id': critic_id,
                                'original': original_text,
                                'translation': translation,
                                'reference': reference_translation,
                                'scores': feedback['annotations'] if 'annotations' in feedback else feedback['scores'],
                                'feedback': feedback['feedback'],
                                'automatic_metrics': feedback['automatic_metrics'],
                                'rhythm_similarity': feedback['rhythm_similarity'],
                                'rhyme_pattern_similarity': feedback['rhyme_pattern_similarity']
                            }
                            
                            feedback_results.append(feedback_result)
                            example_feedback.append(feedback_result)
                            processed_combinations.add(combination)

                            # Сохраняем после каждой комбинации актор/критик
                            interim_data = {
                                'iteration_number': iteration + 1,
                                'feedback_results': feedback_results,
                                'prompts': current_iteration_prompts,
                                'best_translations': best_translations
                            }
                            
                            with open(interim_results_file, 'w', encoding='utf-8') as f:
                                json.dump(interim_data, f, indent=4, ensure_ascii=False)

                    except Exception as e:
                        logging.error(f"Error processing combination: example {example_id}, actor {actor_id}, critic {critic_id}: {e}")
                        continue

            # Обрабатываем результаты примера если все комбинации актор/критик завершены
            if example_feedback:
                try:
                    best_result = find_best_translation(example_feedback)
                    best_translations[example_id] = best_result
                    
                    if iteration < num_iterations - 1:
                        new_prompt = update_prompt_with_llm(
                            initial_prompt=current_prompt,
                            feedback_text=best_result['feedback'],
                            model=actor_model['model'],
                            tokenizer=actor_model['tokenizer']
                        )
                        
                        if new_prompt and len(new_prompt.strip()) > 10:
                            iteration_prompts[example_id] = new_prompt
                        else:
                            iteration_prompts[example_id] = current_prompt
                    
                    # Обновляем метрики после завершения примера
                    iteration_scores = aggregate_scores(feedback_results)
                    save_and_visualize_metrics(
                        output_path=output_path,
                        iteration=iteration + 1,
                        iteration_scores=iteration_scores,
                        all_iterations_history=all_iterations_history
                    )
                    
                except Exception as e:
                    logging.error(f"Error processing results for example {example_id}: {e}")
                    continue

        # Сохраняем финальные результаты итерации
        if feedback_results:
            try:
                final_iteration_data = {
                    'iteration_number': iteration + 1,
                    'feedback_results': feedback_results,
                    'prompts': current_iteration_prompts,
                    'best_translations': best_translations,
                    'scores': aggregate_scores(feedback_results)
                }
                
                with open(os.path.join(iteration_dir, 'full_results.json'), 'w', encoding='utf-8') as f:
                    json.dump(final_iteration_data, f, indent=4, ensure_ascii=False)
                
                if os.path.exists(interim_results_file):
                    os.remove(interim_results_file)
                    
            except Exception as e:
                logging.error(f"Error saving final iteration results: {e}")

    # Сохраняем финальное резюме
    try:
        final_summary = {
            'total_iterations': iteration + 1,
            'final_scores': iteration_scores if feedback_results else None,
            'best_translations': best_translations,
            'configuration': {
                'actor_model': actor_model_name,
                'critic_model': critic_model_name,
                'num_actors': num_actors,
                'num_critics': num_critics,
                'target_score': target_total_score
            }
        }
        
        with open(os.path.join(output_path, 'final_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=4, ensure_ascii=False)
            
    except Exception as e:
        logging.error(f"Error saving final summary: {e}")

    logging.info("ACIIO process completed")



if __name__ == "__main__":
    test_data_path = 'trans_test.jsonl'
    output_path = 'output'
    actor_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    critic_model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    num_actors = 1
    num_critics = 1
    num_iterations = 6
    target_total_score = 30.0
    patience = 2

    run_pace(
        test_data_path=test_data_path,
        output_path=output_path,
        actor_model_name=actor_model_name,
        critic_model_name=critic_model_name,
        num_actors=num_actors,
        num_critics=num_critics,
        num_iterations=num_iterations,
        target_total_score=target_total_score,
        patience=patience
    )
