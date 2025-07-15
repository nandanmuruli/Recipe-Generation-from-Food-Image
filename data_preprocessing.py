import pandas as pd
import os
import re
import ast
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence # For padding sequences

data_file_path = os.path.join(os.getcwd(), 'data', 'dataset', 'full_dataset.csv')

print(f"Attempting to load data from: {data_file_path}")

try:
    df = pd.read_csv(data_file_path)

    print(f"\nDataset loaded successfully! Shape BEFORE cleaning: {df.shape}")

    essential_columns = ['title', 'ingredients', 'directions']
    df_cleaned = df.dropna(subset=essential_columns).reset_index(drop=True)
    rows_dropped = df.shape[0] - df_cleaned.shape[0]
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows due to missing essential data.")
    print(f"Shape AFTER cleaning missing values: {df_cleaned.shape}")

    # Function to clean and tokenize text
    def clean_and_tokenize(text):
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        return tokens

    df_cleaned['title_tokens'] = df_cleaned['title'].apply(clean_and_tokenize)

    def parse_and_tokenize_ingredients(ingredients_str):
        if not isinstance(ingredients_str, str):
            return []
        try:
            ingredients_list = ast.literal_eval(ingredients_str)
            all_tokens = []
            for item in ingredients_list:
                all_tokens.extend(clean_and_tokenize(item))
            return all_tokens
        except (ValueError, SyntaxError):
            return clean_and_tokenize(ingredients_str)

    df_cleaned['ingredients_tokens'] = df_cleaned['ingredients'].apply(parse_and_tokenize_ingredients)

    def parse_and_tokenize_directions(directions_str):
        if not isinstance(directions_str, str):
            return []
        try:
            directions_list = ast.literal_eval(directions_str)
            all_tokens = []
            for step in directions_list:
                all_tokens.extend(clean_and_tokenize(step))
            return all_tokens
        except (ValueError, SyntaxError):
            return clean_and_tokenize(directions_str)

    df_cleaned['directions_tokens'] = df_cleaned['directions'].apply(parse_and_tokenize_directions)

    print("\n--- After Tokenization and Normalization ---")
    print("\nFirst 5 rows with new token columns:")
    print(df_cleaned[['title', 'title_tokens', 'ingredients', 'ingredients_tokens', 'directions', 'directions_tokens']].head())

    # --- Vocabulary Creation & Numericalization (from previous step) ---
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

    for word, count in token_counts.most_common():
        if word not in word_to_id:
            word_to_id[word] = next_id
            next_id += 1

    id_to_word = {id: word for word, id in word_to_id.items()}

    print(f"\n--- Vocabulary Creation ---")
    print(f"Total vocabulary size: {len(word_to_id)}")
    print(f"Number of unique words (excluding special tokens): {len(word_to_id) - len(special_tokens)}")
    print(f"First 10 words in vocabulary (after special tokens): {list(word_to_id.keys())[len(special_tokens):len(special_tokens)+10]}")

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

    print("\n--- After Numericalization ---")
    print("\nFirst 5 rows with new ID columns:")
    print(df_cleaned[['title_tokens', 'title_ids', 'ingredients_tokens', 'ingredients_ids', 'directions_tokens', 'directions_ids']].head())

    # --- NEW CODE FOR DATASET AND DATALOADER ---

    class RecipeDataset(Dataset):
        def __init__(self, dataframe, bos_token_id, eos_token_id):
            self.dataframe = dataframe
            self.bos_token_id = bos_token_id
            self.eos_token_id = eos_token_id

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            # For this example, let's use ingredients as source and directions as target
            # We'll add BOS and EOS tokens to the target sequence
            source_ids = torch.tensor(self.dataframe.loc[idx, 'ingredients_ids'], dtype=torch.long)
            target_ids = torch.tensor([self.bos_token_id] + self.dataframe.loc[idx, 'directions_ids'] + [self.eos_token_id], dtype=torch.long)

            return source_ids, target_ids

    # Custom collate_fn for DataLoader to handle padding
    def collate_fn(batch):
        # batch is a list of (source_ids, target_ids) tuples from __getitem__

        # Separate source and target IDs
        source_ids_batch = [item[0] for item in batch]
        target_ids_batch = [item[1] for item in batch]

        # Pad sequences to the length of the longest sequence in the batch
        # pad_sequence adds padding to the end by default
        padded_source_ids = pad_sequence(source_ids_batch, batch_first=True, padding_value=special_tokens['<PAD>'])
        padded_target_ids = pad_sequence(target_ids_batch, batch_first=True, padding_value=special_tokens['<PAD>'])

        return padded_source_ids, padded_target_ids

    # Create the Dataset instance
    recipe_dataset = RecipeDataset(
        dataframe=df_cleaned,
        bos_token_id=special_tokens['<BOS>'],
        eos_token_id=special_tokens['<EOS>']
    )

    # Create the DataLoader instance
    BATCH_SIZE = 32 # You can adjust this batch size
    recipe_dataloader = DataLoader(
        recipe_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, # Shuffle data for better training
        collate_fn=collate_fn # Use our custom collate function for padding
    )

    print(f"\n--- DataLoader Setup ---")
    print(f"Number of samples in dataset: {len(recipe_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches per epoch: {len(recipe_dataloader)}")

    # Test iterating through one batch
    print("\n--- Testing one batch from DataLoader ---")
    for batch_idx, (source_batch, target_batch) in enumerate(recipe_dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Source Batch Shape (ingredients): {source_batch.shape}")
        print(f"  Target Batch Shape (directions): {target_batch.shape}")
        print(f"  Example Source (ingredients) from batch (first sequence): {source_batch[0][:10]}") # First 10 IDs
        print(f"  Example Target (directions) from batch (first sequence): {target_batch[0][:10]}") # First 10 IDs
        break # Just take one batch for demonstration

    # --- END NEW CODE ---

except FileNotFoundError:
    print(f"Error: The file was not found at {data_file_path}")
    print("Please ensure 'full_dataset.csv' is in your 'data/dataset' folder.")
except Exception as e:
    print(f"An error occurred: {e}")