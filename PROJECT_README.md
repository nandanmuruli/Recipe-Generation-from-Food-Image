# Recipe Generation from Food (AI Project - Text-Based Prototype)

## 1. Project Goal
The ultimate goal of this project is to develop an Artificial Intelligence system capable of generating comprehensive cooking recipes (dish name, ingredients, instructions) directly from a food image.

## 2. Current Functionality (Text-Based Prototype)
As a foundational prototype for the larger vision, this project currently focuses on **generating cooking directions from a given list of ingredients (text input)**. This demonstrates the core Natural Language Generation (NLG) capabilities required for the full project.

A basic **Graphical User Interface (GUI)** has been implemented using Streamlit to provide an interactive demonstration of this text-to-text generation. It also includes a visual placeholder for image upload, representing future integration.

## 3. Tech Stack
* **Programming Language:** Python 3.9
* **Deep Learning Framework:** PyTorch
* **Core Libraries:** Pandas, NumPy, re, ast, collections.Counter, Streamlit, torch.nn.utils.rnn
* **IDE:** PyCharm
* **Version Control:** Git & GitHub

## 4. Dataset
* **Name:** RecipeNLG Dataset (an extended version of Recipe1M+)
* **Source:** [Kaggle Link to RecipeNLG dataset: `https://www.kaggle.com/datasets/saldenisov/recipenlg`](https://www.kaggle.com/datasets/saldenisov/recipenlg)
* **Size:** Contains over 2.2 million recipes (text data only).
* **Training Subset:** For this prototype, a filtered subset of **50,000 recipes** was used for training due to computational resource constraints.
* **Content:** Each recipe includes `id`, `title`, `ingredients` (list of strings), and `directions` (list of strings for steps).
* **Preprocessing:**
    * Rows with missing essential values (`title`, `ingredients`, `directions`) were dropped.
    * Text was lowercased, punctuation removed, and tokenized (split into words).
    * A vocabulary of the **top 10,000 most frequent words** was created, including special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`).
    * Text sequences were converted to numerical IDs and padded to uniform lengths for efficient batch processing by the neural network.

## 5. Model Architecture
The core of the generative model is a **Sequence-to-Sequence (Seq2Seq) neural network** with a Gated Recurrent Unit (GRU) based Encoder-Decoder architecture.
* **Encoder:** Reads the sequence of numericalized ingredient IDs and compresses it into a contextual hidden state.
* **Decoder:** Takes the encoder's hidden state and generates the sequence of numericalized direction IDs, one token at a time, until an End-of-Sequence token is predicted or a maximum length is reached.
* **Model Parameters:** Input/Output Dimension (Vocabulary Size: 10,000), Hidden Dimension: 64.

## 6. Training Details
* **Training Script:** `train.py`
* **Input:** Numericalized ingredient sequences.
* **Output:** Numericalized direction sequences.
* **Loss Function:** `nn.CrossEntropyLoss` (ignoring padding tokens).
* **Optimizer:** `Adam`
* **Device:** Training was performed on **CPU** due to memory constraints on Apple Silicon's MPS backend when handling large vocabulary sizes.
* **Training Run:** **3 Epochs** on the subset of 50,000 recipes, with a batch size of 8.
* **Result:** The model successfully learned basic patterns, demonstrated by a consistent decrease in training loss across epochs. The generated directions, while still basic, show signs of learning recipe-like structures.

    ```
    # Example Training Output (Your actual values might differ slightly)
    Epoch: 01 | Time: [Approx. 1255m]
            Train Loss: [Approx. 5.715] | Train PPL: [Approx. 303.266]
    Epoch: 02 | Time: [Approx. 318m]
            Train Loss: [Approx. 5.355] | Train PPL: [Approx. 211.733]
    Epoch: 03 | Time: [Approx. 704m]
            Train Loss: [Approx. 5.246] | Train PPL: [Approx. 189.777]
    ```

## 7. How to Run the Project
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/nandanmuruli/Recipe-Generation-from-Food-Image.git](https://github.com/nandanmuruli/Recipe-Generation-from-Food-Image.git)
    cd Recipe-Generation-from-Food-Image
    ```
2.  **Set Up Virtual Environment (if not using PyCharm's auto-setup):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
    pip install transformers Pillow numpy pandas matplotlib seaborn streamlit
    ```
4.  **Download Dataset:**
    * Go to [Kaggle RecipeNLG Dataset](https://www.kaggle.com/datasets/saldenisov/recipenlg) and download `full_dataset.csv` (available as a zip file).
    * Create a `data/` folder in your project root, and then a `dataset/` folder inside it: `Recipe-Generation-from-Food-Image/data/dataset/`.
    * Move the downloaded `full_dataset.csv` into the `data/dataset/` folder.
5.  **Train the Model (Optional - Model Weights Provided):**
    * The `seq2seq_recipe_model.pt` file (trained weights) is included in the repository, allowing you to run inference immediately.
    * If you wish to re-train (e.g., for more epochs or a different subset):
        ```bash
        python train.py
        ```
        *(Note: Training can take many hours on CPU, as detailed in Section 6.)*
6.  **Run Inference (Command Line):**
    ```bash
    python inference.py
    ```
    This script will load the saved model and generate example recipe directions based on predefined ingredient inputs in your terminal.
7.  **Run Web Application (GUI):**
    ```bash
    streamlit run app.py
    ```
    This will launch a local web server and open the interactive GUI in your browser. You can then input ingredients and generate recipes visually.

## 8. Sample Generated Output
Ingredients: chicken, onion, garlic, salt, pepper
Generated Directions: cook for 1 hour minutes stirring occasionally until constantly until constantly until constantly until constantly until butter and butter and butter and butter and and butter and and and butter and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and

Ingredients: ['ground beef', 'tomato sauce', 'cheese', 'pasta', 'water']
Generated Directions: cook for 30 minutes minutes minutes minutes minutes minutes minutes minutes minutes until minutes or until until rice and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and

Ingredients: ['unobtainium', 'mythril', 'rare spice']
Generated Directions: cream and in a a medium heat for for about minutes stirring occasionally until constantly until constantly until constantly until butter and butter and butter and butter and and butter and and and butter and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and

## 9. Limitations and Future Work
This project serves as a strong prototype and a foundation for building a truly comprehensive recipe generation system. Key areas for future development include:

### **Crucial Update: Access to Full Multimodal Dataset Achieved!**
As of [July 15, 2025], I have successfully gained access to the full, official Recipe1M+ dataset, which includes both the extensive recipe metadata (titles, ingredients, directions) AND their corresponding food images. This significantly empowers the future development of this project.

**Dataset Access Details:**
* **Recipe1M Images:** Training (94 GiB), Validation (21 GiB), Test (20 GiB)
* **Recipe1M+ Images:** ~210 GiB per `.tar` file (16 parts)
* **Layer2+ metadata:** (2.5 GiB JSON)
* **Model training files:** (data.h5.gz, vocab.bin.gz, etc.)
*(This access allows for the original project vision to be pursued in full.)*

### Other Future Work:
* **Image Integration (Original Goal):** While the data is now available, integrating actual food images into the model requires building an image processing pipeline (e.g., using pre-trained ResNet or Vision Transformer) to extract visual features. These features would then condition the text generation, moving beyond the current text-only input.
* **Improved Generated Text Quality:**
    * **Full Dataset Training:** Training on the entire 2.2 million RecipeNLG dataset (and eventually the full Recipe1M+ dataset) for many more epochs would vastly improve coherence, detail, and reduce repetition in generated recipes. This requires substantial computing resources (e.g., cloud GPUs).
    * **Advanced Architectures:** Exploring state-of-the-art Transformer-based models (e.g., from Hugging Face) for enhanced language understanding and generation capabilities.
    * **Advanced Decoding:** Implementing techniques like Beam Search or Nucleus Sampling during inference for more diverse and coherent outputs.
* **Full Recipe Generation:** Expanding the model's capabilities to also generate dish titles and comprehensive ingredient lists, not just directions.
* **Quantitative Evaluation:** Implementing and reporting industry-standard NLP evaluation metrics (like BLEU, ROUGE, Perplexity on a dedicated validation set) for a quantitative assessment of model performance.

---
Prepared by: Nandan Muruli,Vijaykumar Veeranna
Student ID: 10004359(Nandan Muruli),100002328(Vijaykumar)
Date: July 17, 2025