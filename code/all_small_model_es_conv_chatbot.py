import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import json
import wandb
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    Trainer, TrainingArguments
)
from torch.utils.data import Dataset, DataLoader
import torch

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
    "Others": "Exploration"  # Default to Exploration if unsure
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
                client_message = None  # Reset client message after pairing

# Create a DataFrame to structure the dataset
df = pd.DataFrame({"Client": client_texts, "Therapist": therapist_texts, "Stage": stages})
df['Client'] = df['Client'].fillna('').astype(str)
df['Therapist'] = df['Therapist'].fillna('').astype(str)
df['Stage'] = df['Stage'].fillna('').astype(str)

# Convert data to lists for training
client_texts = df['Client'].tolist()
therapist_texts = df['Therapist'].tolist()
stage_texts = df['Stage'].tolist()

# Initialize and customize tokenizers with custom tokens
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_tokenizer.add_special_tokens(special_tokens_dict)
t5_tokenizer.pad_token = t5_tokenizer.eos_token  # Set padding token

gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_tokenizer.add_special_tokens(special_tokens_dict)
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token  # Set padding token

bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_tokenizer.add_special_tokens(special_tokens_dict)
bart_tokenizer.pad_token = bart_tokenizer.eos_token  # Set padding token

# Initialize and customize models with resized embeddings
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
t5_model.resize_token_embeddings(len(t5_tokenizer))

gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt_model.resize_token_embeddings(len(gpt_tokenizer))

bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
bart_model.resize_token_embeddings(len(bart_tokenizer))

# Custom Dataset class with sharding for memory-efficient loading
class ShardedConversationDataset(Dataset):
    def __init__(self, client_texts, therapist_texts, stage_texts, tokenizer, shard_size=10000, max_length=512):
        self.client_texts = client_texts
        self.therapist_texts = therapist_texts
        self.stage_texts = stage_texts
        self.tokenizer = tokenizer
        self.shard_size = shard_size
        self.max_length = max_length

    def load_shard(self, shard_index):
        # Calculate start and end indices for the shard
        self.start_idx = shard_index * self.shard_size
        end_idx = min((shard_index + 1) * self.shard_size, len(self.client_texts))

        # Combine Stage and Client text
        inputs = [
            f"Stage: {stage} | Client: {client}"
            for stage, client in zip(self.stage_texts[self.start_idx:end_idx], self.client_texts[self.start_idx:end_idx])
        ]

        # Tokenize inputs and therapist responses
        self.client_tokens = self.tokenizer(inputs, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        self.therapist_tokens = self.tokenizer(self.therapist_texts[self.start_idx:end_idx], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        # Set labels and mask padding tokens
        self.therapist_tokens['labels'] = self.therapist_tokens['input_ids'].clone()
        self.therapist_tokens['labels'][self.therapist_tokens['input_ids'] == self.tokenizer.pad_token_id] = -100

    def __len__(self):
        return len(self.client_tokens['input_ids'])

    def __getitem__(self, idx):
        # Do not move data to device here
        return {
            'input_ids': self.client_tokens['input_ids'][idx],
            'attention_mask': self.client_tokens['attention_mask'][idx],
            'labels': self.therapist_tokens['labels'][idx]
        }

# Train each model using customized tokenizer and model with TrainingArguments
shard_size = 10000
models_and_tokenizers = [
    {"model": t5_model, "tokenizer": t5_tokenizer, "output_dir": "./results/t5_small"},
    {"model": gpt_model, "tokenizer": gpt_tokenizer, "output_dir": "./results/gpt2"},
    {"model": bart_model, "tokenizer": bart_tokenizer, "output_dir": "./results/bart_base"}
]

# Training loop for each model
for setup in models_and_tokenizers:
    model = setup["model"]
    tokenizer = setup["tokenizer"]
    output_dir = setup["output_dir"]

    # Initialize the sharded dataset with the model's tokenizer
    sharded_dataset = ShardedConversationDataset(client_texts, therapist_texts, stage_texts, tokenizer, shard_size=shard_size)

    # Determine the number of shards
    total_shards = len(client_texts) // shard_size + (len(client_texts) % shard_size != 0)
    for shard_index in range(total_shards):
        print(f"Loading shard {shard_index + 1}/{total_shards} for model {model.__class__.__name__}")

        sharded_dataset.load_shard(shard_index)
        dataloader = DataLoader(sharded_dataset, batch_size=8, shuffle=True)

        # Define training arguments with wandb logging
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            save_steps=1000,
            save_total_limit=2,
            logging_dir='./logs',
            report_to="wandb"  # Log to WandB
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=sharded_dataset,
            eval_dataset=sharded_dataset
        )

        # Train on the current shard
        trainer.train()
        model.save_pretrained(f"{output_dir}shard{shard_index + 1}")

    # Save the final model and tokenizer
    model.save_pretrained(f"{output_dir}_final")
    tokenizer.save_pretrained(f"{output_dir}_tokenizer")