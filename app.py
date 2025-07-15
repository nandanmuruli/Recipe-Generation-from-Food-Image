import streamlit as st
import torch
import torch.nn as nn
import os
import re
import ast
import pandas as pd
from collections import Counter

# --- Import model components ---
from models.seq2seq_model import EncoderRNN, DecoderRNN, Seq2Seq


# --- Helper Functions (Moved to Global Scope) ---
# These functions must be defined before load_data_and_model or any other usage
def clean_and_tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return tokens


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


def numericalize_tokens(tokens, vocab, unk_id):
    return [vocab.get(token, unk_id) for token in tokens]


# --- Data Preprocessing and Vocab Loading ---
# This runs once when the Streamlit app starts, and results are cached
data_file_path = os.path.join(os.getcwd(), 'data', 'dataset', 'full_dataset.csv')


@st.cache_resource  # Use Streamlit's caching to load data and model only once
def load_data_and_model():
    try:
        df = pd.read_csv(data_file_path)
        essential_columns = ['title', 'ingredients', 'directions']
        df_cleaned = df.dropna(subset=essential_columns).reset_index(drop=True)

        df_cleaned['title_tokens'] = df_cleaned['title'].apply(clean_and_tokenize)
        df_cleaned['ingredients_tokens'] = df_cleaned['ingredients'].apply(parse_and_tokenize_ingredients)
        df_cleaned['directions_tokens'] = df_cleaned['directions'].apply(parse_and_tokenize_directions)

        all_tokens = []
        for col in ['title_tokens', 'ingredients_tokens', 'directions_tokens']:
            for tokens_list in df_cleaned[col]:
                all_tokens.extend(tokens_list)

        token_counts = Counter(all_tokens)

        special_tokens_dict = {  # Renamed to avoid conflict with `special_tokens` variable in global scope
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }

        word_to_id = special_tokens_dict.copy()
        next_id = len(special_tokens_dict)

        # VOCABULARY FILTERING MUST BE IDENTICAL TO TRAIN.PY
        VOCAB_SIZE_LIMIT = 10000  # Corrected to 10000 as per the trained model's vocab size
        filtered_tokens = [word for word, count in
                           token_counts.most_common(VOCAB_SIZE_LIMIT - len(special_tokens_dict))]

        for word in filtered_tokens:
            if word not in word_to_id:
                word_to_id[word] = next_id
                next_id += 1

        id_to_word = {id: word for word, id in word_to_id.items()}

        # --- MODEL HYPERPARAMETERS (Must match what was used for training) ---
        INPUT_DIM = len(word_to_id)
        OUTPUT_DIM = len(word_to_id)
        HIDDEN_DIM = 64  # Must match train.py
        DROPOUT_RATE = 0.5  # Must match train.py

        # --- DEVICE SETUP ---
        device = torch.device('cpu')

        # --- MODEL INSTANTIATION AND LOADING WEIGHTS ---
        encoder = EncoderRNN(INPUT_DIM, HIDDEN_DIM, DROPOUT_RATE).to(device)
        decoder = DecoderRNN(OUTPUT_DIM, HIDDEN_DIM, DROPOUT_RATE).to(device)
        model = Seq2Seq(encoder, decoder, device).to(device)

        model_path = 'seq2seq_recipe_model.pt'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            st.error(f"Error: Model weights not found at {model_path}. Please train the model first.")
            return None, None, None, None, None  # Return 5 Nones on error

        model.eval()  # Set model to evaluation mode

        return model, word_to_id, id_to_word, special_tokens_dict, device  # Return the renamed special_tokens_dict

    except FileNotFoundError:
        st.error(
            f"Error: The file was not found at {data_file_path}. Please ensure 'full_dataset.csv' is in your 'data/dataset' folder.")
        return None, None, None, None, None  # Return 5 Nones on error
    except Exception as e:
        st.error(f"An unexpected error occurred during data/model loading: {e}")
        import traceback
        st.exception(e)  # Show traceback in Streamlit
        return None, None, None, None, None  # Return 5 Nones on error


# Load resources - this function is called only once due to @st.cache_resource
# Unpack the returned special_tokens_dict into the `special_tokens` variable
model, word_to_id, id_to_word, special_tokens, device = load_data_and_model()


# --- INFERENCE FUNCTION (Identical to inference.py) ---
def generate_recipe_directions(model, ingredients_text, word_to_id, id_to_word, special_tokens, device, max_len=100):
    if model is None:
        return "Model not loaded. Please check previous error messages."

    # 1. Preprocess input ingredients text
    ingredients_tokens = parse_and_tokenize_ingredients(ingredients_text)
    ingredients_ids = numericalize_tokens(ingredients_tokens, word_to_id, special_tokens['<UNK>'])

    if not ingredients_ids:
        src_tensor = torch.tensor([special_tokens['<UNK>']], dtype=torch.long).unsqueeze(0).to(device)
    else:
        src_tensor = torch.tensor(ingredients_ids, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension

    # 2. Encode the source (ingredients)
    with torch.no_grad():
        encoder_output, encoder_hidden = model.encoder(src_tensor)

    # 3. Decode to generate directions
    input_token = torch.tensor([special_tokens['<BOS>']], dtype=torch.long).unsqueeze(0).to(device)
    generated_directions_ids = []

    hidden = encoder_hidden

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden = model.decoder(input_token, hidden)

        next_token_id = output.argmax(1).item()
        generated_directions_ids.append(next_token_id)

        if next_token_id == special_tokens['<EOS>']:
            break

        input_token = torch.tensor([next_token_id], dtype=torch.long).unsqueeze(0).to(device)

    # 4. Convert IDs back to words
    generated_words = [id_to_word.get(idx, '<UNK>') for idx in generated_directions_ids]

    if generated_words and generated_words[-1] == '<EOS>':
        generated_words = generated_words[:-1]

    generated_words = [word for word in generated_words if word not in special_tokens.keys()]

    return " ".join(generated_words)


# --- Streamlit APP UI ---
st.set_page_config(page_title="Recipe Generator (Text Prototype)", layout="wide")

st.title("🍽️ Recipe Generator (Text-Based Prototype)")
st.markdown("""
This application demonstrates a prototype AI model capable of generating cooking directions
based on a list of ingredients.
""")

st.header("Upload an Image (For Future Integration)")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image (Visual Placeholder)', use_column_width=True)
    st.info(
        "Note: Image upload is a visual demonstration for future integration. The AI model currently processes text ingredients only.")
    # You could save the image here if needed for future work, e.g.:
    # with open(os.path.join("temp_images", uploaded_file.name), "wb") as f:
    #    f.write(uploaded_file.getbuffer())

st.header("Generate Recipe Directions from Text Ingredients")
st.write("Enter a comma-separated list of ingredients (e.g., `chicken, onion, garlic, salt, pepper`):")

ingredients_input = st.text_area("Ingredients", "chicken, onion, garlic, salt, pepper", height=100)

if st.button("Generate Directions"):
    if model is None:
        st.error("Model or data failed to load. Please check console for errors above.")
    else:
        with st.spinner("Generating recipe directions..."):
            generated_text = generate_recipe_directions(
                model,
                ingredients_input,
                word_to_id,
                id_to_word,
                special_tokens,
                device
            )
        st.subheader("Generated Recipe Directions:")
        st.success(generated_text)
        st.info("Note: Model is trained for a short period on a subset of data. Quality will vary.")

st.markdown("---")
st.subheader("Project Scope & Future Work")
st.markdown("""
This prototype demonstrates the **text generation** component for recipes.

**Current Functionality:** Recipe directions from text ingredients.
**Future Vision:** Generating full recipes from **food images**.

* **Image Integration:** Future work involves building an image processing pipeline
    (e.g., using pre-trained ResNet/Vision Transformers) to extract features from food images,
    which will then be used to condition the recipe generation.
    *(Refer to the official Recipe1M+ dataset access you've achieved!)*
* **Improved Quality:** Training on the full 2.2 million RecipeNLG dataset for more epochs,
    and exploring advanced architectures like Transformer-based models, will significantly improve coherence and detail.
* **Full Recipe Generation:** Expanding to generate dish titles and ingredient lists, not just directions.
""")