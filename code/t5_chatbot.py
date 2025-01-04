import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Hugging Face datasets and convert them to DataFrames
hf_dataset = load_dataset("Mr-Bhaskar/Synthetic_Therapy_Conversations")
hf_df = pd.DataFrame(hf_dataset['train']).rename(columns={'human': 'Client', 'ai': 'Therapist'})

alpaca_ds = load_dataset("adarshxs/Therapy-Alpaca")
alpaca_df = pd.DataFrame(alpaca_ds['train']).rename(columns={'input': 'Client', 'output': 'Therapist'})

mh_th_ds = load_dataset("fadodr/mental_health_therapy")
mh_th_df = pd.DataFrame(mh_th_ds['train']).rename(columns={'input': 'Client', 'output': 'Therapist'})

# Define paths for additional CSV files
file_paths = [
    "/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/Psych_data_cleaned.csv",
    "/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/cleaned_train.csv",
    "/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/client_therapist_conversations_part_1.csv"
]

# Load each CSV file into a DataFrame and append to a list
dataframes = [pd.read_csv(file_path) for file_path in file_paths]
dataframes.extend([hf_df, alpaca_df, mh_th_df])

# Concatenate all DataFrames into a single DataFrame
df = pd.concat(dataframes, ignore_index=True)
df['Client'] = df['Client'].astype(str).fillna('')
df['Therapist'] = df['Therapist'].astype(str).fillna('')

# Prepare text data
client_texts = df['Client'].tolist()
therapist_texts = df['Therapist'].tolist()

# Initialize the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Define the shard size
shard_size = 10000

# Custom Dataset class with sharding for memory-efficient loading
class ShardedConversationDataset(Dataset):
    def __init__(self, client_texts, therapist_texts, tokenizer, shard_size=10000, max_length=512):
        self.client_texts = client_texts
        self.therapist_texts = therapist_texts
        self.tokenizer = tokenizer
        self.shard_size = shard_size
        self.max_length = max_length

    def load_shard(self, shard_index):
        start_idx = shard_index * self.shard_size
        end_idx = min((shard_index + 1) * self.shard_size, len(self.client_texts))

        inputs = self.tokenizer(
            self.client_texts[start_idx:end_idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(device)
        
        outputs = self.tokenizer(
            self.therapist_texts[start_idx:end_idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(device)
        
        outputs['labels'] = outputs['input_ids'].clone()
        outputs['labels'][outputs['input_ids'] == tokenizer.pad_token_id] = -100
        
        self.client_tokens = inputs
        self.therapist_tokens = outputs

    def __len__(self):
        return len(self.client_tokens['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.client_tokens['input_ids'][idx],
            'attention_mask': self.client_tokens['attention_mask'][idx],
            'labels': self.therapist_tokens['labels'][idx]
        }

# Initialize the sharded dataset
sharded_dataset = ShardedConversationDataset(client_texts, therapist_texts, tokenizer, shard_size=shard_size)

# Training loop to iterate over shards
total_shards = len(client_texts) // shard_size + (len(client_texts) % shard_size != 0)
for shard_index in range(total_shards):
    print(f"Loading shard {shard_index + 1}/{total_shards}")
    sharded_dataset.load_shard(shard_index)
    dataloader = DataLoader(sharded_dataset, batch_size=8, shuffle=True)

    # Initialize the T5 model and move to device
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
        fp16=True if torch.cuda.is_available() else False  # Enable mixed precision on GPUs
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

# Save the final model and tokenizer after all shards are processed
model.save_pretrained("/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/t5_therapist_model")
tokenizer.save_pretrained("/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/t5_therapist_tokenizer")

# Generate a response for a new client input
model = T5ForConditionalGeneration.from_pretrained("/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/t5_therapist_model").to(device)
tokenizer = T5Tokenizer.from_pretrained("/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/t5_therapist_tokenizer")

def generate_response(client_input):
    input_text = "Client: " + client_input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    output_ids = model.generate(input_ids, max_length=50)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Test the model with a new client input
print("Therapist Response:", generate_response("I'm feeling really anxious about my work."))