import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import json
import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments
)
from torch.utils.data import Dataset
import wandb
from huggingface_hub import login

import os
os.environ["TMPDIR"] = "/scratch/ngangada/tmp"

# Automatically log in to Hugging Face with a token
huggingface_token = "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"
login(token=huggingface_token, add_to_git_credential=False)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize WandB and log in with the API key
wandb.login(key="c58ae33b91d55a794e239c1d62a67ab9eb7ee4a8")

# Define custom tokens for labeling stages
special_tokens_dict = {'additional_special_tokens': ['<Exploration>', '<Comforting>', '<Action>']}

# Load the ESConv dataset and label stages
ds = load_dataset("thu-coai/esconv")

# Define strategy-to-stage mapping for the ESConv dataset
strategy_to_stage = {
    "Question": "Exploration",
    "Affirmation and Reassurance": "Comforting",
    "Self-disclosure": "Comforting",
    "Providing Suggestions": "Action",
    "Restatement or Paraphrasing": "Exploration",
    "Reflection of feelings": "Comforting",
    "Others": "Exploration"
}

# Process the ESConv dataset to extract client-therapist pairs and assign stages
client_texts, therapist_texts, stages = [], [], []
for entry in ds['train']:
    content = json.loads(entry['text'])
    if 'dialog' in content:
        client_message = None
        for message in content['dialog']:
            if message['speaker'] == "usr":
                client_message = message['text']
            elif message['speaker'] == "sys" and client_message:
                strategy = message.get("strategy", "Others")
                stage = strategy_to_stage.get(strategy, "Exploration")
                client_texts.append(client_message)
                therapist_texts.append(message['text'])
                stages.append(stage)
                client_message = None

df = pd.DataFrame({"Client": client_texts, "Therapist": therapist_texts, "Stage": stages})
df['Client'] = df['Client'].fillna('').astype(str)
df['Therapist'] = df['Therapist'].fillna('').astype(str)
df['Stage'] = df['Stage'].fillna('').astype(str)

# Convert data to lists for training
client_texts = df['Client'].tolist()
therapist_texts = df['Therapist'].tolist()
stage_texts = df['Stage'].tolist()

# Tokenizers for T5, GPT-2, BART, LLaMA, Athene, and Gemma
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_tokenizer.add_special_tokens(special_tokens_dict)
t5_tokenizer.pad_token = t5_tokenizer.eos_token

gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_tokenizer.add_special_tokens(special_tokens_dict)
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_tokenizer.add_special_tokens(special_tokens_dict)
bart_tokenizer.pad_token = bart_tokenizer.eos_token

llama_3b_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
llama_2_7b_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
athene_tokenizer = AutoTokenizer.from_pretrained("Nexusflow/Athene-V2-Chat")
gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

# Models for T5, GPT-2, BART, LLaMA, Athene, and Gemma
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
t5_model.resize_token_embeddings(len(t5_tokenizer))

gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt_model.resize_token_embeddings(len(gpt_tokenizer))

bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
bart_model.resize_token_embeddings(len(bart_tokenizer))

llama_3b_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct").to(device)
llama_2_7b_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(device)
athene_model = AutoModelForCausalLM.from_pretrained("Nexusflow/Athene-V2-Chat").to(device)
gemma_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it").to(device)

# Custom Dataset class
class ShardedConversationDataset(Dataset):
    def __init__(self, client_texts, therapist_texts, stage_texts, tokenizer, shard_size=10000, max_length=512):
        self.client_texts = client_texts
        self.therapist_texts = therapist_texts
        self.stage_texts = stage_texts
        self.tokenizer = tokenizer
        self.shard_size = shard_size
        self.max_length = max_length

    def load_shard(self, shard_index):
        start_idx = shard_index * self.shard_size
        end_idx = min((shard_index + 1) * self.shard_size, len(self.client_texts))
        inputs = [
            f"Stage: {stage} | Client: {client}"
            for stage, client in zip(self.stage_texts[start_idx:end_idx], self.client_texts[start_idx:end_idx])
        ]
        self.client_tokens = self.tokenizer(inputs, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        self.therapist_tokens = self.tokenizer(self.therapist_texts[start_idx:end_idx], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        self.therapist_tokens['labels'] = self.therapist_tokens['input_ids'].clone()
        self.therapist_tokens['labels'][self.therapist_tokens['input_ids'] == self.tokenizer.pad_token_id] = -100

    def __len__(self):
        return len(self.client_tokens['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.client_tokens['input_ids'][idx],
            'attention_mask': self.client_tokens['attention_mask'][idx],
            'labels': self.therapist_tokens['labels'][idx]
        }

# Train loop
models_and_tokenizers = [
    {"model": t5_model, "tokenizer": t5_tokenizer, "output_dir": "./results/t5_small"},
    {"model": gpt_model, "tokenizer": gpt_tokenizer, "output_dir": "./results/gpt2"},
    {"model": bart_model, "tokenizer": bart_tokenizer, "output_dir": "./results/bart_base"},
    {"model": llama_3b_model, "tokenizer": llama_3b_tokenizer, "output_dir": "./results/llama_3b"},
    {"model": llama_2_7b_model, "tokenizer": llama_2_7b_tokenizer, "output_dir": "./results/llama_2_7b"},
    {"model": athene_model, "tokenizer": athene_tokenizer, "output_dir": "./results/athene"},
    {"model": gemma_model, "tokenizer": gemma_tokenizer, "output_dir": "./results/gemma"}
]

for setup in models_and_tokenizers:
    model = setup["model"]
    tokenizer = setup["tokenizer"]
    output_dir = setup["output_dir"]

    sharded_dataset = ShardedConversationDataset(client_texts, therapist_texts, stage_texts, tokenizer)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        save_steps=1000,
        save_total_limit=2,
        logging_dir='./logs',
        report_to="wandb"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=sharded_dataset, eval_dataset=sharded_dataset)
    trainer.train()
    model.save_pretrained(f"{output_dir}_modified_final")
    tokenizer.save_pretrained(f"{output_dir}_modified_tokenizer")