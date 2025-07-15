import torch
import torch.nn as nn
import os
import re
import ast
from collections import Counter
import pandas as pd  # <--- THIS LINE IS CRUCIAL FOR RESOLVING NameError
from models.seq2seq_model import EncoderRNN, DecoderRNN, Seq2Seq  # Import your model classes
import traceback  # For better error reporting

# --- Data Preprocessing and Vocab Loading (Copied for self-containment) ---
# This part must be identical to how vocab was built in train.py
data_file_path = os.path.join(os.getcwd(), 'data', 'dataset', 'full_dataset.csv')

print(f"Attempting to load data to rebuild vocabulary from: {data_file_path}")

try:
    df = pd.read_csv(data_file_path)
    essential_columns = ['title', 'ingredients', 'directions']
    df_cleaned = df.dropna(subset=essential_columns).reset_index(drop=True)
    print(f"Dataset loaded and cleaned for vocab rebuild. Shape: {df_cleaned.shape}")


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

    # VOCABULARY FILTERING MUST BE IDENTICAL TO TRAIN.PY
    VOCAB_SIZE_LIMIT = 10000  # Must match the limit used in train.py!
    filtered_tokens = [word for word, count in token_counts.most_common(VOCAB_SIZE_LIMIT - len(special_tokens))]

    for word in filtered_tokens:
        if word not in word_to_id:
            word_to_id[word] = next_id
            next_id += 1

    id_to_word = {id: word for word, id in word_to_id.items()}


    def numericalize_tokens(tokens, vocab, unk_id):
        return [vocab.get(token, unk_id) for token in tokens]


    # --- END DATA PREPROCESSING AND VOCAB LOADING ---

    # --- MODEL HYPERPARAMETERS (Must match what was used for training) ---
    INPUT_DIM = len(word_to_id)
    OUTPUT_DIM = len(word_to_id)
    HIDDEN_DIM = 64  # Must match train.py
    DROPOUT_RATE = 0.5  # Must match train.py

    # --- DEVICE SETUP ---
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nModel will run on: {device}")

    # --- MODEL INSTANTIATION AND LOADING WEIGHTS ---
    encoder = EncoderRNN(INPUT_DIM, HIDDEN_DIM, DROPOUT_RATE).to(device)
    decoder = DecoderRNN(OUTPUT_DIM, HIDDEN_DIM, DROPOUT_RATE).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    model_path = 'seq2seq_recipe_model.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights loaded from {model_path}")
    else:
        print(f"Error: Model weights not found at {model_path}. Please train the model first.")
        exit()  # Exit if model not found

    model.eval()  # Set model to evaluation mode (turns off dropout, etc.)


    # --- INFERENCE FUNCTION ---
    def generate_recipe_directions(model, ingredients_text, word_to_id, id_to_word, special_tokens, device,
                                   max_len=100):
        # 1. Preprocess input ingredients text
        ingredients_tokens = parse_and_tokenize_ingredients(ingredients_text)
        ingredients_ids = numericalize_tokens(ingredients_tokens, word_to_id, special_tokens['<UNK>'])

        if not ingredients_ids:
            # If no valid ingredients, provide a placeholder input (e.g., UNK token) or handle appropriately
            # For now, let's just indicate an error
            print("Warning: Input ingredients text could not be processed into valid IDs. Using UNK token.")
            src_tensor = torch.tensor([special_tokens['<UNK>']], dtype=torch.long).unsqueeze(0).to(device)
        else:
            src_tensor = torch.tensor(ingredients_ids, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension

        # 2. Encode the source (ingredients)
        with torch.no_grad():
            encoder_output, encoder_hidden = model.encoder(src_tensor)

        # 3. Decode to generate directions
        # Start with <BOS> token
        input_token = torch.tensor([special_tokens['<BOS>']], dtype=torch.long).unsqueeze(0).to(device)
        generated_directions_ids = []

        hidden = encoder_hidden  # Initial hidden state for decoder

        for _ in range(max_len):
            with torch.no_grad():
                output, hidden = model.decoder(input_token, hidden)

            # Get the predicted next token (highest probability)
            next_token_id = output.argmax(1).item()
            generated_directions_ids.append(next_token_id)

            # If <EOS> token is predicted, stop generation
            if next_token_id == special_tokens['<EOS>']:
                break

            # Use the predicted token as the next input to the decoder
            input_token = torch.tensor([next_token_id], dtype=torch.long).unsqueeze(0).to(device)

        # 4. Convert IDs back to words
        generated_words = [id_to_word.get(idx, '<UNK>') for idx in generated_directions_ids]

        # Remove <EOS> if it's the last token (and other special tokens if desired)
        if generated_words and generated_words[-1] == '<EOS>':
            generated_words = generated_words[:-1]

        # Also remove <BOS> if it somehow appears in the output
        generated_words = [word for word in generated_words if word not in special_tokens.keys()]

        # Join words to form a sentence
        return " ".join(generated_words)


    # --- TEST INFERENCE ---
    print("\n--- Testing Inference ---")

    # Example 1: Simple ingredients
    test_ingredients_1 = "chicken, onion, garlic, salt, pepper"
    print(f"\nIngredients: {test_ingredients_1}")
    generated_directions_1 = generate_recipe_directions(model, test_ingredients_1, word_to_id, id_to_word,
                                                        special_tokens, device)
    print(f"Generated Directions: {generated_directions_1}")

    # Example 2: More complex ingredients (try to mimic dataset format)
    test_ingredients_2 = "['ground beef', 'tomato sauce', 'cheese', 'pasta', 'water']"
    print(f"\nIngredients: {test_ingredients_2}")
    generated_directions_2 = generate_recipe_directions(model, test_ingredients_2, word_to_id, id_to_word,
                                                        special_tokens, device)
    print(f"Generated Directions: {generated_directions_2}")

    # Example 3: Test with some out-of-vocabulary words
    test_ingredients_3 = "['unobtainium', 'mythril', 'rare spice']"
    print(f"\nIngredients: {test_ingredients_3}")
    generated_directions_3 = generate_recipe_directions(model, test_ingredients_3, word_to_id, id_to_word,
                                                        special_tokens, device)
    print(f"Generated Directions: {generated_directions_3}")

except FileNotFoundError:
    print(f"Error: The file was not found at {data_file_path}")
    print("Please ensure 'full_dataset.csv' is in your 'data/dataset' folder.")
except Exception as e:
    print(f"An unexpected error occurred during inference setup or execution: {e}")
    traceback.print_exc()  # Print full traceback for debugging