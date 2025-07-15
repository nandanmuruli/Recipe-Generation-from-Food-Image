import pandas as pd
import os
import re
import ast
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence # For padding sequences
import random
import math
import time
import traceback # For better error reporting

# --- Import model components ---
# Assuming seq2seq_model.py is in the 'models' directory relative to train.py
# Make sure your models/seq2seq_model.py file is saved with the latest class definitions
from models.seq2seq_model import EncoderRNN, DecoderRNN, Seq2Seq

# --- Data Preprocessing and Loading (Copied for self-containment) ---
# In a larger project, you would modularize this into separate functions/files
# and import them.

data_file_path = os.path.join(os.getcwd(), 'data', 'dataset', 'full_dataset.csv')

print(f"Attempting to load data for training from: {data_file_path}")

try:
    df = pd.read_csv(data_file_path)
    essential_columns = ['title', 'ingredients', 'directions']
    df_cleaned = df.dropna(subset=essential_columns).reset_index(drop=True)
    print(f"Dataset loaded and cleaned. Shape: {df_cleaned.shape}")

    def clean_and_tokenize(text):
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        return tokens

    df_cleaned['title_tokens'] = df_cleaned['title'].apply(clean_and_tokenize)

    def parse_and_tokenize_ingredients(ingredients_str):
        if not isinstance(ingredients_str, str) or not ingredients_str.strip():
            return []
        try:
            ingredients_list = ast.literal_eval(ingredients_str)
            if not isinstance(ingredients_list, list):
                ingredients_list = [ingredients_list]
            all_tokens = []
            for item in ingredients_list:
                all_tokens.extend(clean_and_tokenize(str(item)))
            return all_tokens
        except (ValueError, SyntaxError):
            return clean_and_tokenize(ingredients_str)

    df_cleaned['ingredients_tokens'] = df_cleaned['ingredients'].apply(parse_and_tokenize_ingredients)

    def parse_and_tokenize_directions(directions_str):
        if not isinstance(directions_str, str) or not directions_str.strip():
            return []
        try:
            directions_list = ast.literal_eval(directions_str)
            if not isinstance(directions_list, list):
                directions_list = [directions_list]
            all_tokens = []
            for step in directions_list:
                all_tokens.extend(clean_and_tokenize(str(step)))
            return all_tokens
        except (ValueError, SyntaxError):
            return clean_and_tokenize(directions_str)

    df_cleaned['directions_tokens'] = df_cleaned['directions'].apply(parse_and_tokenize_directions)

    all_tokens = []
    for col in ['title_tokens', 'ingredients_tokens', 'directions_tokens']:
        for tokens_list in df_cleaned[col]:
            all_tokens.extend(tokens_list)

    token_counts = Counter(all_tokens)

    special_tokens = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,
        '<EOS>': 3
    }

    word_to_id = special_tokens.copy()
    next_id = len(special_tokens)

    # --- NEW: VOCABULARY FILTERING ---
    VOCAB_SIZE_LIMIT = 10000 # Keep only the top 50,000 most frequent words + special tokens. You can adjust this.
    filtered_tokens = [word for word, count in token_counts.most_common(VOCAB_SIZE_LIMIT - len(special_tokens))]

    for word in filtered_tokens:
        if word not in word_to_id: # Ensure we don't overwrite special tokens
            word_to_id[word] = next_id
            next_id += 1
    # All other words not in filtered_tokens will be mapped to <UNK> (ID 1)
    # --- END VOCABULARY FILTERING ---

    id_to_word = {id: word for word, id in word_to_id.items()}

    def numericalize_tokens(tokens, vocab, unk_id):
        return [vocab.get(token, unk_id) for token in tokens]

    df_cleaned['title_ids'] = df_cleaned['title_tokens'].apply(
        lambda x: numericalize_tokens(x, word_to_id, special_tokens['<UNK>'])
    )
    df_cleaned['ingredients_ids'] = df_cleaned['ingredients_tokens'].apply(
        lambda x: numericalize_tokens(x, word_to_id, special_tokens['<UNK>'])
    )
    df_cleaned['directions_ids'] = df_cleaned['directions_tokens'].apply(
        lambda x: numericalize_tokens(x, word_to_id, special_tokens['<UNK>'])
    )

    # --- RecipeDataset and collate_fn ---
    class RecipeDataset(Dataset):
        def __init__(self, dataframe, bos_token_id, eos_token_id):
            self.dataframe = dataframe
            self.bos_token_id = bos_token_id
            self.eos_token_id = eos_token_id
            self.ingredients_ids_list = dataframe['ingredients_ids'].tolist()
            self.directions_ids_list = dataframe['directions_ids'].tolist()

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            source_ids = torch.tensor(self.ingredients_ids_list[idx], dtype=torch.long)
            current_directions_ids = self.directions_ids_list[idx]
            if not isinstance(current_directions_ids, list):
                current_directions_ids = []
            target_ids = torch.tensor(
                [self.bos_token_id] + current_directions_ids + [self.eos_token_id],
                dtype=torch.long
            )
            return source_ids, target_ids

    def collate_fn(batch):
        source_ids_batch = [item[0] for item in batch]
        target_ids_batch = [item[1] for item in batch]

        padded_source_ids = pad_sequence(source_ids_batch, batch_first=True, padding_value=special_tokens['<PAD>'])
        padded_target_ids = pad_sequence(target_ids_batch, batch_first=True, padding_value=special_tokens['<PAD>'])

        return padded_source_ids, padded_target_ids

    # --- MODEL HYPERPARAMETERS ---
    INPUT_DIM = len(word_to_id) # Now reflects filtered vocabulary size
    OUTPUT_DIM = len(word_to_id) # Now reflects filtered vocabulary size
    HIDDEN_DIM = 64 # Kept at 64, can be increased slightly if memory allows after vocab filtering
    DROPOUT_RATE = 0.5
    LEARNING_RATE = 0.001
    N_EPOCHS = 3 # Set to 3 for quick test, increase later
    CLIP = 1.0

    BATCH_SIZE = 8 # Kept at 8, can try increasing after confirming it runs

    # --- DEVICE SETUP ---
    # We will still try MPS first, as vocab filtering should help a lot
    #device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cpu')  # FORCED TO CPU TO AVOID MPS MEMORY ERROR
    print(f"\nModel will run on: {device}")

    # --- MODEL INSTANTIATION ---
    encoder = EncoderRNN(INPUT_DIM, HIDDEN_DIM, DROPOUT_RATE).to(device)
    decoder = DecoderRNN(OUTPUT_DIM, HIDDEN_DIM, DROPOUT_RATE).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # --- LOSS FUNCTION AND OPTIMIZER ---
    criterion = nn.CrossEntropyLoss(ignore_index=special_tokens['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- DATASET AND DATALOADER SETUP ---
    recipe_dataset = RecipeDataset(
        dataframe=df_cleaned,
        bos_token_id=special_tokens['<BOS>'],
        eos_token_id=special_tokens['<EOS>']
    )

    # SUBSET SIZE for faster testing
    subset_size = 50000 # Kept at 50000 samples for a very quick test run. You can increase this.
    if len(recipe_dataset) > subset_size:
        indices = torch.randperm(len(recipe_dataset))[:subset_size]
        subset_dataset = torch.utils.data.Subset(recipe_dataset, indices)
        print(f"Using a subset of {len(subset_dataset)} samples for faster testing.")
        train_dataloader = DataLoader(
            subset_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
    else:
        train_dataloader = DataLoader(
            recipe_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )

    print(f"Number of samples in full dataset: {len(recipe_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches per epoch (training on subset): {len(train_dataloader)}")


    # --- TRAINING LOOP ---
    def train_epoch(model, dataloader, optimizer, criterion, clip, device):
        model.train()
        epoch_loss = 0

        for batch_idx, (src, trg) in enumerate(dataloader):
            src, trg = src.to(device), trg.to(device)

            optimizer.zero_grad()
            output = model(src, trg)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)} Loss: {loss.item():.4f}")

        return epoch_loss / len(dataloader)

    print("\n--- Starting Training ---")
    best_loss = float('inf')
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, CLIP, device)
        end_time = time.time()

        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'seq2seq_recipe_model.pt')
            print(f"Model saved to seq2seq_recipe_model.pt (Epoch {epoch+1})")

    print("\n--- Training Complete ---")


except FileNotFoundError:
    print(f"Error: The file was not found at {data_file_path}")
    print("Please ensure 'full_dataset.csv' is in your 'data/dataset' folder.")
except Exception as e:
    print(f"An unexpected error occurred during training setup or execution: {e}")
    traceback.print_exc()