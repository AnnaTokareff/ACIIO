# **Self-Prompt Refinement via Actor-Critic Paradigm**  
**Automated Instruction Optimization for Large Language Models**  

## **Overview**  
This project introduces **Actor-Critic Iterative Instruction Optimization (ACIIO)**, an approach that enhances **prompt quality and output performance** using two LLMs:  
- **Actor** – Generates outputs.  
- **Critic** – Evaluates and refines prompts iteratively.  

Tested on **three NLP tasks**:  
- **Dialogue Summarization** – Evaluates accuracy, conciseness, coherence, etc.  
- **Lyrics Translation** – Assesses semantic accuracy, musicality, poetic quality, BLEU, BERTScore, and Rhythm Similarity.  
- **Code Generation** – Analyzes correctness, efficiency, readability, and complexity.  

## **Future Directions**  
- Fine-tuning critic scoring for **better adaptability**.  
- Exploring actor-critic methods for **creative AI applications**.  
- Investigating **prompt optimization** for structured tasks.  

## **Acknowledgments**  
Inspired by [**Self-Refine**](https://github.com/madaan/self-refine.git) and [**PACE**](https://arxiv.org/html/2308.10088v2), but developed **entirely from scratch** with new methodologies and architectures.

```sql
@misc{madaan2023selfrefine,
      title={Self-Refine: Iterative Refinement with Self-Feedback}, 
      author={Aman Madaan and Niket Tandon and Prakhar Gupta and Skyler Hallinan and Luyu Gao and Sarah Wiegreffe and Uri Alon and Nouha Dziri and Shrimai Prabhumoye and Yiming Yang and Sean Welleck and Bodhisattwa Prasad Majumder and Shashank Gupta and Amir Yazdanbakhsh and Peter Clark},
      year={2023},
      eprint={2303.17651},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```sql
@misc{dong2024paceimprovingpromptactorcritic,
      title={PACE: Improving Prompt with Actor-Critic Editing for Large Language Model}, 
      author={Yihong Dong and Kangcheng Luo and Xue Jiang and Zhi Jin and Ge Li},
      year={2024},
      eprint={2308.10088},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2308.10088}, 
}
```
