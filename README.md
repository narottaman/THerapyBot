# AI-Powered Counseling Therapy: Personalized Conversational Models

This repository presents a state-of-the-art **AI-powered framework** for personalized counseling therapy using advanced **Natural Language Processing (NLP)** techniques. The project aims to improve accessibility, scalability, and effectiveness of mental health services by leveraging transformer-based models such as **BART**, **GPT-2**, **T5**, **LLaMA-3B**, and **DistilGPT-2**.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Goals and Objectives](#project-goals-and-objectives)
3. [Datasets](#datasets)
4. [Models Used](#models-used)
5. [Results and Analysis](#results-and-analysis)
6. [How to Run](#how-to-run)
7. [Achievements](#achievements)
8. [Conclusion](#conclusion)
9. [License](#license)

---

## Overview

The global demand for mental health services highlights the urgent need for scalable, personalized, and effective solutions. This project introduces an **AI-driven framework** designed to generate empathetic and contextually relevant responses for therapeutic conversations. The system uses a curated dataset of **622,000 therapist-client dialogues**, segmented into three therapeutic stages:
- **Exploration**: Understanding client concerns.
- **Comforting**: Providing empathetic support.
- **Action**: Suggesting actionable steps.

---

## Project Goals and Objectives

### Goals
- Transform mental health care by developing AI models that complement human therapists.
- Extend the reach of therapeutic services to underserved populations.

### Objectives
1. **Enhance Therapeutic Interaction**: Equip the AI with nuanced emotional understanding to replicate human-like empathy.
2. **Improve Accessibility**: Provide high-quality therapy to diverse populations at scale.
3. **Reduce Therapeutic Burnout**: Allow therapists to focus on critical cases by automating routine sessions.

---

## Datasets

This project utilized a diverse array of datasets to train the models:

1. **[Nart-100k-Synthetic](https://huggingface.co/datasets/jerryjalapeno/nart-100k-synthetic)**:
   - Synthetic conversations simulating various psychological conditions.

2. **[Mental Health Counseling Conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)**:
   - Real-world counseling texts capturing client-therapist interactions.

3. **[Counsel Chat](https://huggingface.co/datasets/nbertagnolli/counsel-chat)**:
   - Transcripts of live counseling sessions with dynamic interactions.

4. **[Synthetic Therapy Conversations](https://huggingface.co/datasets/Mr-Bhaskar/Synthetic_Therapy_Conversations)**:
   - Simulated therapeutic dialogues focusing on mental health scenarios.

5. **[Therapy-Alpaca](https://huggingface.co/datasets/adarshxs/Therapy-Alpaca)**:
   - Dialogues showcasing supportive and empathetic responses.

6. **[Mental Health Therapy](https://huggingface.co/datasets/fadodr/mental_health_therapy)**:
   - In-depth therapeutic records covering various mental health treatments.

7. **[ESConv](https://huggingface.co/datasets/thu-coai/esconv)**:
   - Dialogues categorized by therapeutic strategies.

---

## Models Used

1. **BART (Bidirectional and Auto-Regressive Transformers)**:
   - Superior performance in empathetic and contextually relevant dialogue generation.
   - Key Features: Bidirectional encoder for context understanding and autoregressive decoder for coherent text generation.

2. **GPT-2**:
   - Known for its fluent text generation capabilities.
   - Struggles with maintaining coherence in longer dialogues.

3. **T5 (Text-to-Text Transfer Transformer)**:
   - Treats all NLP tasks as text-to-text, offering versatility across applications.

4. **LLaMA-3B**:
   - Multilingual capabilities but requires fine-tuning for contextual understanding.

5. **DistilGPT-2**:
   - Lightweight model optimized for speed and efficiency, ideal for real-time applications.

---

## Results and Analysis

### Key Findings

1. **BART**:
   - **Perplexity**: 14.85 (lowest among models, indicating better language modeling).
   - **BLEU-2 Score**: 0.0864 (highest n-gram overlap with reference texts).
   - **ROUGE-L Score**: 0.2478 (best in generating structurally relevant responses).

2. **T5**:
   - Demonstrated strong semantic similarity but lacked BART's structural coherence.

3. **GPT-2**:
   - Effective in generating fluent text but struggled with longer dialogue coherence.

4. **LLaMA-3B**:
   - Underperformed despite a large parameter count due to limited fine-tuning.

5. **DistilGPT-2**:
   - Efficient and lightweight but lacked depth in nuanced therapeutic responses.

---

### Performance Metrics

| Model         | Perplexity | BLEU-2 Score | ROUGE-L Score | BOW Embedding Score |
|---------------|------------|--------------|---------------|----------------------|
| **BART**      | **14.85**  | **0.0864**   | **0.2478**    | 0.6788               |
| T5            | 3.40       | 3.33×10⁻¹⁵⁵  | 0.1698        | **0.7394**           |
| GPT-2         | 204.38     | 2.64×10⁻¹⁵⁵  | 0.1734        | 0.7092               |
| LLaMA-3B      | 917718.94  | 1.87×10⁻¹⁵⁵  | 0.1732        | 0.6817               |
| DistilGPT-2   | 222.65     | 2.97×10⁻¹⁵⁵  | 0.1757        | 0.7048               |

---

## How to Run

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/personalized-ai-therapy.git
   cd personalized-ai-therapy
