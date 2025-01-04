import warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Check if a GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Sharded Dataset Class
class ShardedConversationDataset(Dataset):
    def __init__(self, client_texts, therapist_texts, tokenizer, shard_size=1000, max_length=1024):
        self.client_texts = client_texts
        self.therapist_texts = therapist_texts
        self.tokenizer = tokenizer
        self.shard_size = shard_size
        self.max_length = max_length

    def load_shard(self, shard_index):
        # Calculate the start and end indices for the current shard
        self.start_idx = shard_index * self.shard_size
        end_idx = min((shard_index + 1) * self.shard_size, len(self.client_texts))

        # Tokenize the shard of the dataset
        self.client_tokens = self.tokenizer(self.client_texts[self.start_idx:end_idx],
                                            padding='max_length',
                                            truncation=True,
                                            max_length=self.max_length,
                                            return_tensors='pt')

        self.therapist_tokens = self.tokenizer(self.therapist_texts[self.start_idx:end_idx],
                                               padding='max_length',
                                               truncation=True,
                                               max_length=self.max_length,
                                               return_tensors='pt')

        # Set labels and mask padding tokens
        self.therapist_tokens['labels'] = self.therapist_tokens['input_ids'].clone()
        self.therapist_tokens['labels'][self.therapist_tokens['input_ids'] == self.tokenizer.pad_token_id] = -100

    def __len__(self):
        return len(self.client_tokens['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.client_tokens['input_ids'][idx].to(device),
            'attention_mask': self.client_tokens['attention_mask'][idx].to(device),
            'labels': self.therapist_tokens['labels'][idx].to(device)
        }

# Load and Prepare the Data
hf_dataset = load_dataset("Mr-Bhaskar/Synthetic_Therapy_Conversations")
hf_df = pd.DataFrame(hf_dataset['train']).rename(columns={'human': 'Client', 'ai': 'Therapist'})

alpaca_ds = load_dataset("adarshxs/Therapy-Alpaca")
alpaca_df = pd.DataFrame(alpaca_ds['train']).rename(columns={'input': 'Client', 'output': 'Therapist'})

mh_th_ds = load_dataset("fadodr/mental_health_therapy")
mh_th_df = pd.DataFrame(mh_th_ds['train']).rename(columns={'input': 'Client', 'output': 'Therapist'})

# Define the file paths
file_paths = [
    "/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/Psych_data_cleaned.csv",
    "/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/cleaned_train.csv",
    "/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/client_therapist_conversations_part_1.csv",
    "/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/client_therapist_conversations_part_2.csv"
]

dataframes = [pd.read_csv(file_path) for file_path in file_paths]
dataframes.append(hf_df)
dataframes.append(alpaca_df)
dataframes.append(mh_th_df)

df = pd.concat(dataframes, ignore_index=True)
df['Client'] = df['Client'].astype(str).fillna('')
df['Therapist'] = df['Therapist'].astype(str).fillna('')

client_texts = df['Client'].tolist()
therapist_texts = df['Therapist'].tolist()

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Shard size
shard_size = 10000
sharded_dataset = ShardedConversationDataset(client_texts, therapist_texts, tokenizer, shard_size=shard_size)

# Training loop with GPU support
total_shards = len(client_texts) // shard_size + (len(client_texts) % shard_size != 0)
for shard_index in range(total_shards):
    print(f"Loading shard {shard_index + 1}/{total_shards}")
    sharded_dataset.load_shard(shard_index)

    dataloader = DataLoader(sharded_dataset, batch_size=8, shuffle=True)
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_steps=1000,
        save_total_limit=2,
        logging_dir='./logs',
        fp16=True if torch.cuda.is_available() else False  # Enable mixed-precision on GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sharded_dataset,
        eval_dataset=sharded_dataset
    )

    trainer.train()
    model.save_pretrained(f"./results/model_shard_{shard_index + 1}")

# Save the final model and tokenizer
model.save_pretrained("/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/gpt2_big_therapist_model")
tokenizer.save_pretrained("/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/gpt2_big_therapist_tokenizer")

# Response generation on GPU
model.to(device)

def generate_response(text):
    input_tokens = tokenizer(text, return_tensors='pt').to(device)
    output_tokens = model.generate(input_tokens['input_ids'], max_length=50)
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# Test responses
print("Therapist Response:", generate_response("I'm feeling really anxious about my work."))
print("Therapist Response:", generate_response("I'm feeling depressed"))
print("Therapist Response:", generate_response("I want to die!"))