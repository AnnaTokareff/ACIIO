import os
import json
import logging
from typing import List, Dict, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from actor import generate_response, setup_simple_prompt
from critic import generate_feedback, compute_semantic_similarity

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def update_json(new_data: Union[dict, list], file_path: str):
    """Update JSON file with new data, handling both lists and dicts."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = [] if isinstance(new_data, dict) else {}
        else:
            data = [] if isinstance(new_data, dict) else {}

        if isinstance(data, list):
            data.append(new_data)
        elif isinstance(data, dict):
            data.update(new_data)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
    except Exception as e:
        logging.error(f"Error updating JSON file {file_path}: {e}")

def load_data(file_path: str) -> List[Dict]:
    """Load dataset from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            example = json.loads(line)
            example['id'] = i
            data.append(example)
    logging.info(f"Loaded {len(data)} examples from {file_path}")
    return data

def initialize_model(model_name: str) -> Dict[str, Union[AutoModelForCausalLM, AutoTokenizer]]:
    """Initialize the model and tokenizer."""
    try:
        logging.info(f"Initializing model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        return {'model': model, 'tokenizer': tokenizer, 'name': model_name}
    except Exception as e:
        logging.error(f"Error loading model {model_name}: {e}")
        raise


def validate_and_clean_prompt(new_prompt: str, initial_prompt: str) -> str:
    """Validate and clean the generated prompt."""
    if not new_prompt or len(new_prompt) < 10:
        logging.warning("Generated prompt too short or empty, keeping initial prompt")
        return initial_prompt
    return new_prompt


def find_best_summary(feedback_results: List[Dict]) -> Dict:
    """Find best summary by averaging critics scores for each unique summary."""
    if not feedback_results:
        raise ValueError("No feedback results provided")
    
    # Группируем результаты по суммаризациям (actor_id)
    summaries = {}
    for result in feedback_results:
        actor_id = result['actor_id']
        if actor_id not in summaries:
            summaries[actor_id] = {
                'response': result['response'],
                'semantic_similarities': result['semantic_similarities'],
                'critic_scores': [],
                'annotations': [],
                'feedbacks': []
            }
        summaries[actor_id]['critic_scores'].append(result['annotations']['Total'])
        summaries[actor_id]['annotations'].append(result['annotations'])
        summaries[actor_id]['feedbacks'].append(result['feedback'])

    # Находим лучшую суммаризацию
    best_score = -1
    best_summary = None

    for actor_id, data in summaries.items():
        try:
            # Усредняем оценки критиков
            avg_critic_score = sum(data['critic_scores']) / len(data['critic_scores'])
            
            # Берем лучшее семантическое сходство
            best_similarity = max(data['semantic_similarities'])
            
            # Считаем общий скор (60% семантика, 40% оценки критиков)
            total_score = (best_similarity * 0.4) + (avg_critic_score/40 * 0.6)
            
            if total_score > best_score:
                best_score = total_score
                best_summary = {
                    'summary': data['response'],
                    'score': total_score,
                    'metrics': {
                        'semantic_similarity': best_similarity,
                        'avg_critic_score': avg_critic_score,
                        'raw_critic_scores': data['critic_scores'],
                        'detailed_scores': {  # Усредненные оценки по каждому критерию
                            criterion: sum(a.get(criterion, 0) for a in data['annotations']) / len(data['annotations'])
                            for criterion in ['Accuracy', 'Conciseness', 'Coherence', 'Completeness', 
                                           'Readability', 'Relevance', 'Informativeness', 'Engagement']
                        }
                    },
                    'feedback': max(data['feedbacks'], key=len)  # берем самый информативный фидбек
                }
            
        except Exception as e:
            logging.warning(f"Error processing summary from actor {actor_id}: {e}")
            continue
    
    if not best_summary:
        raise ValueError("No valid summary found")
    
    logging.info(f"""
        Selected best summary:
        Score: {best_summary['score']:.3f}
        Semantic similarity: {best_summary['metrics']['semantic_similarity']:.3f}
        Average critic score: {best_summary['metrics']['avg_critic_score']:.3f}
            """)
    
    return best_summary

def aggregate_scores(feedback_results: List[Dict], test_data: List[Dict]) -> Dict[str, Union[float, Dict]]:
    """Calculate comprehensive average scores across all feedback results."""
    if not feedback_results:
        return {
            'avg_semantic_similarity': 0.0,
            'avg_critic_score': 0.0,
            'total_score': 0.0,
            'num_examples': 0,
            'criterion_averages': {},
            'example_scores': {},
            'best_example_scores': {},
            'avg_best_semantic_similarity': 0.0,
            'avg_best_critic_score': 0.0,
            'best_total_score': 0.0
        }

    example_scores = {}
    best_example_scores = {}
    total_similarity = 0.0
    total_critic_score = 0.0
    total_best_similarity = 0.0
    total_best_critic_score = 0.0
    criterion_sums = {
        'Accuracy': 0.0, 'Conciseness': 0.0, 'Coherence': 0.0,
        'Completeness': 0.0, 'Readability': 0.0, 'Relevance': 0.0,
        'Informativeness': 0.0, 'Engagement': 0.0
    }
    
    for example in test_data:
        example_id = str(example['id'])
        example_results = [r for r in feedback_results if r['example_id'] == example_id]
        
        if example_results:
            example_similarity = sum(r['semantic_similarities'][0] for r in example_results) / len(example_results)
            example_critic_score = sum(r['annotations']['Total'] for r in example_results) / len(example_results)
            
            example_scores[example_id] = {
                'semantic_similarity': example_similarity,
                'critic_score': example_critic_score,
                'total_score': (example_similarity * 0.4) + (example_critic_score / 40 * 0.6)
            }
            
            total_similarity += example_similarity
            total_critic_score += example_critic_score
            
            best_result = find_best_summary(example_results)
            best_example_scores[example_id] = {
                'semantic_similarity': best_result['metrics']['semantic_similarity'],
                'critic_score': best_result['metrics']['avg_critic_score'],
                'total_score': best_result['score']
            }
            
            total_best_similarity += best_result['metrics']['semantic_similarity']
            total_best_critic_score += best_result['metrics']['avg_critic_score']
            
            for criterion in criterion_sums.keys():
                criterion_sums[criterion] += sum(r['annotations'].get(criterion, 0) for r in example_results) / len(example_results)
    
    num_examples = len(example_scores)
    avg_similarity = total_similarity / num_examples if num_examples > 0 else 0.0
    avg_critic_score = total_critic_score / num_examples if num_examples > 0 else 0.0
    avg_best_similarity = total_best_similarity / num_examples if num_examples > 0 else 0.0
    avg_best_critic_score = total_best_critic_score / num_examples if num_examples > 0 else 0.0
    criterion_averages = {criterion: score / num_examples if num_examples > 0 else 0.0 for criterion, score in criterion_sums.items()}
    total_score = (avg_similarity * 0.4) + (avg_critic_score / 40 * 0.6)
    best_total_score = (avg_best_similarity * 0.4) + (avg_best_critic_score / 40 * 0.6)

    return {
        'avg_semantic_similarity': avg_similarity,
        'avg_critic_score': avg_critic_score,
        'total_score': total_score,
        'num_examples': num_examples,
        'criterion_averages': criterion_averages,
        'example_scores': example_scores,
        'best_example_scores': best_example_scores,
        'avg_best_semantic_similarity': avg_best_similarity,
        'avg_best_critic_score': avg_best_critic_score,
        'best_total_score': best_total_score
    }

def update_prompt_with_llm(initial_prompt: str, feedback_text: str, model, tokenizer) -> str:
    """Update the summarization prompt based on critic's prompt improvement feedback."""
    try:
        prompt_update_input = f"""You are an expert prompt engineer. \
        Your task is to improve the current summarization prompt based on the critic's feedback.

Current Prompt: {initial_prompt}

Critic's Feedback: {feedback_text}

REQUIREMENTS:
1. Start from "Summarize the dialogue".
2. Keep new prompt VERY simple and concise (max. 15-20 words).
2. Focus on critic's feedback fpr prompt improvement suggestions.
4. If the current prompt is good enough, do not change it.
5. Restrict the summary in length up to 25-30 words.


Write only the new prompt. Do not explain or justify changes.

New prompt:"""

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        inputs = tokenizer(
            prompt_update_input,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1056
        ).to(model.device)

        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.input_ids.ne(tokenizer.pad_token_id).long(),
            max_new_tokens=40,
            temperature=0.2,
            top_p=0.6,
            do_sample=True
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Проверяем наличие маркера "New prompt:"
        if "New prompt:" in generated_text:
            new_prompt_start = generated_text.find("New prompt:") + len("New prompt:")
        elif "Summarize the dialogue" in generated_text:
            new_prompt_start = generated_text.find("Summarize the dialogue")
        else:
            logging.warning("Expected start of new prompt not found. Using initial prompt.")
            return initial_prompt  # Возвращаем начальный промпт, если начало не найдено

        new_prompt = generated_text[new_prompt_start:].strip()
        new_prompt = validate_and_clean_prompt(new_prompt, initial_prompt)
        logging.info(f"Generated new prompt: {new_prompt}")
        
        return validate_and_clean_prompt(new_prompt, initial_prompt)

    except Exception as e:
        logging.error(f"Error in update_prompt_with_llm: {str(e)}")
        return initial_prompt

def save_iteration_results(iteration: int, 
                           iteration_dir: str,
                           prompts: Dict,
                           feedback_results: List[Dict],
                           best_state: Dict,
                           test_data: List[Dict]) -> None:
    """Save detailed results for the current iteration."""
    # Сохраняем промпты
    prompts_file = os.path.join(iteration_dir, 'prompts.json')
    with open(prompts_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=4, ensure_ascii=False)
    
    # Сохраняем общую статистику итерации
    summary_file = os.path.join(iteration_dir, 'iteration_summary.json')
    iteration_scores = aggregate_scores(feedback_results, test_data)
    summary = {
        'iteration': iteration,
        'timestamp': datetime.now().isoformat(),
        'num_examples': iteration_scores['num_examples'],
        'scores': {
            'avg_semantic_similarity': iteration_scores['avg_semantic_similarity'],
            'avg_critic_score': iteration_scores['avg_critic_score'],
            'total_score': iteration_scores['total_score'],
            'avg_best_semantic_similarity': iteration_scores['avg_best_semantic_similarity'],
            'avg_best_critic_score': iteration_scores['avg_best_critic_score'],
            'best_total_score': iteration_scores['best_total_score'],
            'criterion_averages': iteration_scores['criterion_averages']
        },
        'best_state': {
            'scores': best_state['scores'],
            'iteration': best_state['iteration']
        }
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    # Сохраняем детальные оценки для каждого примера
    example_scores_file = os.path.join(iteration_dir, 'example_scores.json')
    with open(example_scores_file, 'w', encoding='utf-8') as f:
        json.dump(iteration_scores['example_scores'], f, indent=4, ensure_ascii=False)

    best_example_scores_file = os.path.join(iteration_dir, 'best_example_scores.json')
    with open(best_example_scores_file, 'w', encoding='utf-8') as f:
        json.dump(iteration_scores['best_example_scores'], f, indent=4, ensure_ascii=False)
 
 
def visualize_metrics_by_type(metrics_dir: str, all_iterations_history: List[Dict]):
    """Create three separate plots: one for semantic similarity, one for critic scores, and one for total scores."""
    
    # Получаем все example_ids
    example_ids = set()
    for metrics in all_iterations_history:
        example_ids.update(metrics['scores']['per_example_metrics'].keys())
    
    # Определяем метрики для визуализации
    metric_groups = {
        'semantic_similarity': {
            'title': 'Semantic Similarity Progress',
            'metrics': ['avg_semantic_similarity', 'best_semantic_similarity'],
            'ylabel': 'Similarity Score'
        },
        'critic_score': {
            'title': 'Critic Score Progress',
            'metrics': ['avg_critic_score', 'best_critic_score'],
            'ylabel': 'Critic Score (normalized)'
        },
        'total_score': {
            'title': 'Total Score Progress',
            'metrics': ['total_score', 'best_total_score'],
            'ylabel': 'Total Score'
        }
    }

    # Создаем график для каждой группы метрик
    for metric_type, metric_info in metric_groups.items():
        plt.style.use('seaborn')
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        ax.grid(False)

        iterations = range(1, len(all_iterations_history) + 1)
        
        # Генерируем разные цвета для разных примеров
        colors = plt.cm.rainbow(np.linspace(0, 1, len(example_ids)))
        
        for example_id, color in zip(example_ids, colors):
            # Собираем данные для примера
            avg_values = []
            best_values = []
            
            for history in all_iterations_history:
                example_metrics = history['scores']['per_example_metrics'].get(example_id, {})
                if metric_type == 'critic_score':
                    # Нормализуем critic score
                    avg_values.append(example_metrics.get(metric_info['metrics'][0], 0) / 40)
                    best_values.append(example_metrics.get(metric_info['metrics'][1], 0) / 40)
                else:
                    avg_values.append(example_metrics.get(metric_info['metrics'][0], 0))
                    best_values.append(example_metrics.get(metric_info['metrics'][1], 0))
            
            # Строим линии для среднего и лучшего значения
            plt.plot(iterations, avg_values, 
                    label=f'Avg Example {example_id}',
                    color=color,
                    linestyle='-',
                    linewidth=2,
                    marker='o',
                    markersize=6)
            
            plt.plot(iterations, best_values,
                    label=f'Best Example {example_id}',
                    color=color,
                    linestyle='--',
                    linewidth=2,
                    marker='s',
                    markersize=6)

        plt.title(metric_info['title'], size=14, pad=20)
        plt.xlabel('Iteration', size=12, labelpad=10)
        plt.ylabel(metric_info['ylabel'], size=12, labelpad=10)
        
        # Настраиваем легенду
        plt.legend(bbox_to_anchor=(1.05, 1),
                  loc='upper left',
                  borderaxespad=0,
                  frameon=False,
                  fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, f'{metric_type}_all_examples.png'),
                    dpi=300,
                    bbox_inches='tight')
        plt.close()
               
def save_and_visualize_metrics(output_path: str, iteration: int, 
                             iteration_scores: Dict, 
                             all_iterations_history: List[Dict]) -> None:
    """Save and visualize metrics for the current iteration and overall progress."""
    metrics_dir = os.path.join(output_path, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    current_metrics = {
        'iteration': iteration,
        'timestamp': datetime.now().isoformat(),
        'scores': iteration_scores,
    }
    all_iterations_history.append(current_metrics)
    
    # 1. График прогресса основных метрик
    plt.figure(figsize=(12, 8))
    iterations = range(1, iteration + 1)
    
    # Определяем порядок метрик
    metrics_order = [
        'Semantic Similarity',
        'Critic Score',
        'Total Score',
        'Best Semantic Similarity',
        'Best Critic Score',
        'Best Total Score'
    ]
    
    metrics_data = {
    'Semantic Similarity': [m['scores']['avg_semantic_similarity'] for m in all_iterations_history],
    'Critic Score': [m['scores']['avg_critic_score']/40 for m in all_iterations_history],
    'Total Score': [m['scores']['total_score'] for m in all_iterations_history],
    'Best Semantic Similarity': [m['scores']['avg_best_semantic_similarity'] for m in all_iterations_history],
    'Best Critic Score': [max(0, m['scores']['avg_best_critic_score']/40) for m in all_iterations_history],
    'Best Total Score': [m['scores']['best_total_score'] for m in all_iterations_history]
    }
    
    # Определяем цвета и стили
    colors = {
        'Semantic Similarity': '#2196F3',
        'Critic Score': '#663399',
        'Total Score': '#1164B4',
        'Best Semantic Similarity': '#50C878',
        'Best Critic Score': '#EE82EE',
        'Best Total Score': '#7FC7FF'
    }
    
    line_styles = {
        'Semantic Similarity': '-',
        'Critic Score': '-',
        'Total Score': '-',
        'Best Semantic Similarity': '--',
        'Best Critic Score': '--',
        'Best Total Score': '--'
    }

    # Строим график
    ax = plt.gca()
    ax.grid(False)
    plt.xticks(range(1, 7))
    
    for metric_name in metrics_order:
        if metric_name in metrics_data:
            plt.plot(iterations, 
                    metrics_data[metric_name],
                    label=metric_name,
                    color=colors[metric_name],
                    linestyle=line_styles[metric_name],
                    linewidth=2.5,
                    marker='o',
                    markersize=8,
                    markeredgecolor='white',
                    markeredgewidth=2)

    plt.title('Progress of Main Metrics Across Iterations', size=14, pad=20)
    plt.xlabel('Iteration', size=12, labelpad=10)
    plt.ylabel('Score', size=12, labelpad=10)
    plt.yticks([0.40, 0.50, 0.60, 0.70, 0.80])
    
    # Создаем легенду
    lines = ax.get_lines()
    labels = [line.get_label() for line in lines]
    
    # Добавляем разделители в легенду
    legend_elements = [(lines[i], labels[i]) for i in range(len(lines))]
    plt.legend([elem[0] for elem in legend_elements],
              [elem[1] for elem in legend_elements],
              bbox_to_anchor=(1.05, 1),
              loc='upper left',
              borderaxespad=0,
              frameon=False,
              fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'main_metrics_progress.png'),
                dpi=300,
                bbox_inches='tight')
    plt.close()
    
    # 2. Тепловая карта критериев
    criterion_scores = {
        'Accuracy': [], 'Conciseness': [], 'Coherence': [],
        'Completeness': [], 'Readability': [], 'Relevance': [],
        'Informativeness': [], 'Engagement': []
    }
    
    for metrics in all_iterations_history:
        for criterion in criterion_scores:
            criterion_scores[criterion].append(
                metrics['scores']['criterion_averages'][criterion]
            )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        data=[[scores[i] for scores in criterion_scores.values()] 
              for i in range(iteration)],
        xticklabels=list(criterion_scores.keys()),
        yticklabels=range(1, iteration + 1),
        annot=True,
        fmt='.2f',
        cmap='YlOrRd'
    )
    plt.title('Criterion Scores Heatmap')
    plt.xlabel('Criteria')
    plt.ylabel('Iteration')
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'criterion_heatmap.png'))
    plt.close()


def run_pace(test_data_path: str, 
             output_path: str, 
             actor_model_name: str,
             critic_model_name: str,
             num_actors: int,
             num_critics: int,
             num_iterations: int,
             similarity_model) -> None:
    """Main PACE loop for dialogue summarization."""
    logging.info("Starting PACE process")
    os.makedirs(output_path, exist_ok=True)

    # Инициализация
    test_data = load_data(test_data_path)
    actor_model = initialize_model(actor_model_name)
    critic_model = initialize_model(critic_model_name)

    # Отслеживание состояния
    iteration_prompts = {}
    best_state = {
        'prompts': {},
        'scores': {},
        'summaries': {},
        'iteration': 0
    }
    
    # История для метрик
    all_iterations_history = []

    for iteration in range(num_iterations):
        logging.info(f"\n{'='*50}\nStarting iteration {iteration + 1}/{num_iterations}\n{'='*50}")
        
        # Создаем директории для текущей итерации
        iteration_dir = os.path.join(output_path, f'iteration_{iteration + 1}')
        os.makedirs(iteration_dir, exist_ok=True)
        
        feedback_file = os.path.join(iteration_dir, 'feedback.json')
        summaries_file = os.path.join(iteration_dir, 'summaries.json')
        prompts_file = os.path.join(iteration_dir, 'prompts.json')
        
        iteration_feedback_results = []

        # Обработка каждого примера
        for example in test_data:
            example_id = str(example['id'])
            logging.info(f"\nProcessing example {example_id}")
            
                    # Получаем или инициализируем промпт
            if iteration == 0:
                current_prompt = setup_simple_prompt()
                logging.info(f"Using initial prompt for example {example_id}: {current_prompt}")
            else:
                current_prompt = iteration_prompts.get(example_id, setup_simple_prompt())
                logging.info(f"Using updated prompt for example {example_id}")

            # Сохраняем текущий промпт
            iteration_prompts[example_id] = current_prompt
            update_json({example_id: current_prompt}, prompts_file)
                                
            # Генерация и оценка для каждого актора
            example_feedback_results = []
            for actor_id in range(num_actors):
                logging.info(f"Actor {actor_id + 1}/{num_actors} generating summary")
                
                # Генерация суммаризации
                summary = generate_response(
                    example['dialogue'],
                    current_prompt,
                    actor_model['model'],
                    actor_model['tokenizer']
                )

                if summary:
                    # Сохраняем суммаризацию
                    summary_data = {
                        'example_id': example_id,
                        'actor_id': actor_id,
                        'summary': summary
                    }
                    update_json(summary_data, summaries_file)
                    
                    # Получение оценок от критиков
                    for critic_id in range(num_critics):
                        logging.info(f"Critic {critic_id + 1}/{num_critics} evaluating summary")
                        
                        feedback = generate_feedback(
                            example['dialogue'],
                            summary,
                            current_prompt,
                            critic_model['model'],
                            critic_model['tokenizer']
                        )

                        # Подсчет семантического сходства
                        similarities = [
                            compute_semantic_similarity(summary, ref, similarity_model)
                            for ref in example['summaries']
                        ]

                        feedback_result = {
                            'example_id': example_id,
                            'actor_id': actor_id,
                            'critic_id': critic_id,
                            'dialogue': example['dialogue'],
                            'response': summary,
                            'reference_summaries': example['summaries'],
                            'semantic_similarities': similarities,
                            'annotations': feedback['annotations'],
                            'feedback': feedback['feedback']
                        }
                        
                        example_feedback_results.append(feedback_result)
                        update_json(feedback_result, feedback_file)

            # Обработка результатов текущего примера
            if example_feedback_results:
                # Выбор лучшей суммаризации
                best_result = find_best_summary(example_feedback_results)
                current_score = best_result['score']
                
                # Обновление лучшего состояния если результат улучшился
                if current_score > best_state['scores'].get(example_id, -1):
                    best_state['scores'][example_id] = current_score
                    best_state['prompts'][example_id] = current_prompt
                    best_state['summaries'][example_id] = best_result['summary']
                    best_state['iteration'] = iteration
                    
                    # Генерация нового промпта для следующей итерации (начиная со второй)
                if iteration > 0 and iteration < num_iterations - 1:  # Добавлено условие iteration > 0
                    new_prompt = update_prompt_with_llm(
                        initial_prompt=current_prompt,
                        feedback_text=best_result['feedback'],
                        model=actor_model['model'],
                        tokenizer=actor_model['tokenizer']
                    )
                    if new_prompt and len(new_prompt.strip()) > 10:
                        iteration_prompts[example_id] = new_prompt
                        logging.info(f"Updated prompt for example {example_id}")
                    else:
                        iteration_prompts[example_id] = current_prompt
                        logging.warning(f"Invalid new prompt, keeping current for example {example_id}")
                else:
                    # Возврат к лучшему промпту если результат ухудшился
                    best_prompt = best_state['prompts'].get(example_id)
                    if best_prompt:
                        iteration_prompts[example_id] = best_prompt
                        logging.info(f"Reverting to best prompt for example {example_id}")
                    else:
                        iteration_prompts[example_id] = current_prompt
                
                # Добавляем результаты примера к результатам итерации
                iteration_feedback_results.extend(example_feedback_results)

        # Сохранение и визуализация результатов итерации
        if iteration_feedback_results:
            iteration_scores = aggregate_scores(iteration_feedback_results, test_data)
            
            # Сохраняем метрики
            save_and_visualize_metrics(
                output_path=output_path,
                iteration=iteration + 1,
                iteration_scores=iteration_scores,
                all_iterations_history=all_iterations_history
            )
            
            # Сохраняем детальные результаты
            save_iteration_results(
            iteration=iteration + 1,
            iteration_dir=iteration_dir,
            prompts=iteration_prompts,
            feedback_results=iteration_feedback_results,
            best_state=best_state,
            test_data=test_data
        )

    logging.info("ACIIO process completed")

if __name__ == "__main__":
    test_data_path = 'dialog_test.jsonl'
    output_path = 'output'
    actor_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    critic_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    from sentence_transformers import SentenceTransformer
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    run_pace(
        test_data_path=test_data_path,
        output_path=output_path,
        actor_model_name=actor_model_name,
        critic_model_name=critic_model_name,
        num_actors=1,
        num_critics=1,
        num_iterations=6,
        similarity_model=similarity_model
    )

