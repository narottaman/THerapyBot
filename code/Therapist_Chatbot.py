import torch
from torch.utils.data import Dataset, DataLoader
import wandb

wandb.login(key="c58ae33b91d55a794e239c1d62a67ab9eb7ee4a8")

class ShardedConversationDataset(Dataset):
    def __init__(self, client_texts, therapist_texts, tokenizer, shard_size=1000, max_length=1024):
        self.client_texts = client_texts
        self.therapist_texts = therapist_texts
        self.tokenizer = tokenizer
        self.shard_size = shard_size
        self.max_length = max_length
        self.start_idx = 0  # Initialize the starting index for the shard

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
        # Return the number of samples in the shard
        return len(self.client_tokens['input_ids'])

    def __getitem__(self, idx):
        # Return a single sample from the current shard
        return {
            'input_ids': self.client_tokens['input_ids'][idx],
            'attention_mask': self.client_tokens['attention_mask'][idx],
            'labels': self.therapist_tokens['labels'][idx]
        }


import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader

# Load the dataset
file_path = "/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/client_therapist_conversations.csv"
#scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/client_therapist_conversations.csv
df = pd.read_csv(file_path)

# Ensure that all entries are strings (remove NaNs or non-string data)
df['Client'] = df['Client'].astype(str).fillna('')
df['Therapist'] = df['Therapist'].astype(str).fillna('')
# Assuming df['Client'] and df['Therapist'] contain the full dataset
client_texts = df['Client'].tolist()
therapist_texts = df['Therapist'].tolist()

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Shard size: define how many samples to load in each shard (e.g., 1000 at a time)
shard_size = 1000

# Initialize the sharded dataset
sharded_dataset = ShardedConversationDataset(client_texts, therapist_texts, tokenizer, shard_size=shard_size)

# Training loop to iterate over shards
total_shards = len(client_texts) // shard_size + (len(client_texts) % shard_size != 0)
for shard_index in range(total_shards):
    print(f"Loading shard {shard_index + 1}/{total_shards}")

    # Load the current shard into memory
    sharded_dataset.load_shard(shard_index)

    # Create DataLoader for the current shard
    dataloader = DataLoader(sharded_dataset, batch_size=8, shuffle=True)

    # Initialize model
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))  # Update model embeddings for the new pad_token

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,  # Set the number of epochs you want for each shard
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs'
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sharded_dataset,
        eval_dataset=sharded_dataset  # You can use a different dataset for evaluation
    )

    # Train on the current shard
    trainer.train()

    # Save the model after training each shard (optional)
    model.save_pretrained(f"./results/model_shard_{shard_index + 1}")

model.save_pretrained("/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/gpt2_big_therapist_model")
tokenizer.save_pretrained("/scratch/ngangada/SML_GRP_Prj_Therapist_ChatBot/gpt2_big_therapist_tokenizer")
# Check if a GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device (GPU or CPU)
model.to(device)

# Generate a response for a new client input
new_client_text = "I'm feeling really anxious about my work."
input_tokens = tokenizer(new_client_text, return_tensors='pt')

# Move the input tokens to the same device as the model
input_tokens = {key: val.to(device) for key, val in input_tokens.items()}

# Generate response
output_tokens = model.generate(input_tokens['input_ids'], max_length=50)

# Decode the output to text
response_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print("Therapist Response:", response_text)
