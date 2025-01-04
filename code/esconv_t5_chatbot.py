import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import json

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the ESConv dataset
ds = load_dataset("thu-coai/esconv")

# Define strategy-to-stage mapping
strategy_to_stage = {
    "Question": "Exploration",
    "Affirmation and Reassurance": "Comforting",
    "Self-disclosure": "Comforting",
    "Providing Suggestions": "Action",
    "Restatement or Paraphrasing": "Exploration",
    "Reflection of feelings": "Comforting",
    "Others": "Exploration"  # Default to Exploration if unsure
}

# Initialize lists to store client, therapist pairs, and stages
client_texts = []
therapist_texts = []
stages = []  # Separate list to store stages

# Process each conversation in the ESConv dataset
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

# Create a DataFrame with client, therapist pairs, and stage as separate columns
df = pd.DataFrame({
    "Client": client_texts,
    "Therapist": therapist_texts,
    "Stage": stages  # Separate Stage column
})

# Ensure all entries are strings and remove NaN values
df['Client'] = df['Client'].astype(str).fillna('')
df['Therapist'] = df['Therapist'].astype(str).fillna('')
df['Stage'] = df['Stage'].astype(str).fillna('')

# Convert DataFrame columns to lists for sharded loading
client_texts = df['Client'].tolist()
therapist_texts = df['Therapist'].tolist()
stage_texts = df['Stage'].tolist()

# Initialize the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Define the shard size (how many samples to load per shard)
shard_size = 10000

# Custom Dataset class with sharding for efficient memory use
class ShardedConversationDataset(Dataset):
    def __init__(self, client_texts, therapist_texts, stage_texts, tokenizer, shard_size=10000, max_length=512):
        self.client_texts = client_texts
        self.therapist_texts = therapist_texts
        self.stage_texts = stage_texts
        self.tokenizer = tokenizer
        self.shard_size = shard_size
        self.max_length = max_length

    def load_shard(self, shard_index):
        # Calculate start and end indices for the current shard
        start_idx = shard_index * self.shard_size
        end_idx = min((shard_index + 1) * self.shard_size, len(self.client_texts))

        # Prepare inputs by combining Stage and Client text
        inputs = [
            f"Stage: {stage} | Client: {client}"
            for stage, client in zip(self.stage_texts[start_idx:end_idx], self.client_texts[start_idx:end_idx])
        ]

        # Tokenize the current shard and move to GPU
        self.client_tokens = self.tokenizer(inputs, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)
        self.therapist_tokens = self.tokenizer(self.therapist_texts[start_idx:end_idx], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)

        # Set labels and mask padding tokens
        self.therapist_tokens['labels'] = self.therapist_tokens['input_ids'].clone()
        self.therapist_tokens['labels'][self.therapist_tokens['input_ids'] == tokenizer.pad_token_id] = -100

    def __len__(self):
        return len(self.client_tokens['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.client_tokens['input_ids'][idx],
            'attention_mask': self.client_tokens['attention_mask'][idx],
            'labels': self.therapist_tokens['labels'][idx]
        }

# Initialize the sharded dataset
sharded_dataset = ShardedConversationDataset(client_texts, therapist_texts, stage_texts, tokenizer, shard_size=shard_size)

# Training loop to iterate over shards
total_shards = len(client_texts) // shard_size + (len(client_texts) % shard_size != 0)
for shard_index in range(total_shards):
    print(f"Loading shard {shard_index + 1}/{total_shards}")
    sharded_dataset.load_shard(shard_index)

    # Initialize model and move to GPU
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_steps=1000,
        save_total_limit=2,
        logging_dir='./logs',
        fp16=True  # Enable mixed precision if GPU supports it
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
    model.save_pretrained(f"./results/model_shard_{shard_index + 1}")

# Save the final model and tokenizer
model.save_pretrained("/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/t5_therapist_model_esconv_stages")
tokenizer.save_pretrained("/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/t5_therapist_tokenizer_esconv_stages")

# Load model and tokenizer for inference
model = T5ForConditionalGeneration.from_pretrained("/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/t5_therapist_model_esconv_stages").to(device)
tokenizer = T5Tokenizer.from_pretrained("/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/t5_therapist_tokenizer_esconv_stages")

def generate_response(client_input):
    input_text = "Client: " + client_input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    output_ids = model.generate(input_ids, max_length=50)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Test response
print("Therapist Response:", generate_response("I'm feeling really anxious about my work."))