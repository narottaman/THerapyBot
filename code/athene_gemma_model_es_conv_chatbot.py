import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import json
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from huggingface_hub import login
import os

os.environ["TMPDIR"] = "/scratch/ngangada/tmp"
os.environ["HF_HOME"] = "/scratch/ngangada/huggingface_cache"

def login_huggingface(token):
    """Log in to Hugging Face."""
    login(token=token, add_to_git_credential=False)

def process_esconv_dataset():
    """Load and preprocess the ESConv dataset."""
    ds = load_dataset("thu-coai/esconv")
    strategy_to_stage = {
        "Question": "Exploration",
        "Affirmation and Reassurance": "Comforting",
        "Self-disclosure": "Comforting",
        "Providing Suggestions": "Action",
        "Restatement or Paraphrasing": "Exploration",
        "Reflection of feelings": "Comforting",
        "Others": "Exploration"
    }

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

    return pd.DataFrame({"Client": client_texts, "Therapist": therapist_texts, "Stage": stages})

class ShardedConversationDataset(Dataset):
    """Custom dataset for sharded loading."""
    def __init__(self, client_texts, therapist_texts, stage_texts, tokenizer, max_length=512):
        self.client_texts = client_texts
        self.therapist_texts = therapist_texts
        self.stage_texts = stage_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-tokenize all data during initialization
        self.client_tokens = self.tokenizer(
            [f"Stage: {stage} | Client: {client}" for stage, client in zip(self.stage_texts, self.client_texts)],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        self.therapist_tokens = self.tokenizer(
            self.therapist_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        self.therapist_tokens['labels'] = self.therapist_tokens['input_ids'].clone()
        self.therapist_tokens['labels'][self.therapist_tokens['input_ids'] == self.tokenizer.pad_token_id] = -100

    def __len__(self):
        return len(self.client_texts)

    def __getitem__(self, idx):
        return {
            'input_ids': self.client_tokens['input_ids'][idx],
            'attention_mask': self.client_tokens['attention_mask'][idx],
            'labels': self.therapist_tokens['labels'][idx]
        }


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Hugging Face login
huggingface_token = "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"
login_huggingface(huggingface_token)

# Preprocess dataset
df = process_esconv_dataset()
client_texts = df['Client'].tolist()
therapist_texts = df['Therapist'].tolist()
stage_texts = df['Stage'].tolist()

# Tokenizers
athene_tokenizer = AutoTokenizer.from_pretrained("Nexusflow/Athene-V2-Chat")
if athene_tokenizer.pad_token is None:
    athene_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
if gemma_tokenizer.pad_token is None:
    gemma_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Train Loop
models_and_tokenizers = [
    {"model_path": "Nexusflow/Athene-V2-Chat", "tokenizer": athene_tokenizer, "output_dir": "./results/athene"},
    {"model_path": "google/gemma-2-2b-it", "tokenizer": gemma_tokenizer, "output_dir": "./results/gemma"}
]

for setup in models_and_tokenizers:
    model_path = setup["model_path"]
    tokenizer = setup["tokenizer"]
    output_dir = setup["output_dir"]

    # Initialize the dataset
    dataset = ShardedConversationDataset(client_texts, therapist_texts, stage_texts, tokenizer)

    # Load the model onto GPU just before training
    print(f"Loading model {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.gradient_checkpointing_enable()  # Enable gradient checkpointing
    model.resize_token_embeddings(len(tokenizer))
    model = model.to("cuda")

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        save_steps=1000,
        save_total_limit=2,
        logging_dir='./logs',
        report_to="wandb",
        fp16=True
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, eval_dataset=dataset)
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained(f"{output_dir}_final")
    tokenizer.save_pretrained(f"{output_dir}_tokenizer")

    # Release GPU memory after training
    del model
    torch.cuda.empty_cache()