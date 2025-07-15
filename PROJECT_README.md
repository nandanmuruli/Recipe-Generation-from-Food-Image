# Recipe Generation from Food Image (AI Project - Text-Based Prototype)

## 1. Project Goal
The ultimate goal of this project is to develop an Artificial Intelligence system capable of generating comprehensive cooking recipes (dish name, ingredients, instructions) directly from a food image.

## 2. Current Functionality (Text-Based Prototype)
As a foundational prototype for the larger vision, this project currently focuses on **generating cooking directions from a given list of ingredients (text input)**. This demonstrates the core Natural Language Generation (NLG) capabilities required for the full project.

## 3. Tech Stack
* **Programming Language:** Python 3.9
* **Deep Learning Framework:** PyTorch
* **Core Libraries:** Pandas, NumPy, re, ast, collections.Counter, torch.nn.utils.rnn
* **IDE:** PyCharm
* **Version Control:** Git & GitHub

## 4. Dataset
* **Name:** RecipeNLG Dataset (an extended version of Recipe1M+)
* **Source:** [Kaggle Link to RecipeNLG dataset: `https://www.kaggle.com/datasets/saldenisov/recipenlg`]
* **Size:** Over 2.2 million recipes (text data only). For this prototype, a subset of 50,000 recipes was used for training.
* **Content:** Each recipe includes `id`, `title`, `ingredients` (list of strings), and `directions` (list of strings for steps).
* **Preprocessing:**
    * Missing essential values (title, ingredients, directions) were dropped.
    * Text was lowercased, punctuation removed, and tokenized.
    * A vocabulary of the top 50,000 most frequent words was created, with special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`).
    * Text sequences were converted to numerical IDs and padded for batch processing.

## 5. Model Architecture
The core of the generative model is a **Sequence-to-Sequence (Seq2Seq) neural network** with a Gated Recurrent Unit (GRU) based Encoder-Decoder architecture.
* **Encoder:** Reads the sequence of ingredient IDs and compresses it into a hidden state.
* **Decoder:** Takes the encoder's hidden state and generates the sequence of direction IDs, one token at a time.
* **Parameters:** Input/Output Dimension (Vocabulary Size: 50,000), Hidden Dimension: 64.

## 6. Training Details
* **Training Script:** `train.py`
* **Input:** Numericalized ingredient sequences.
* **Output:** Numericalized direction sequences.
* **Loss Function:** `nn.CrossEntropyLoss` (ignoring padding tokens).
* **Optimizer:** `Adam`
* **Device:** Training was performed on **CPU** due to memory constraints on Apple Silicon's MPS backend.
* **Training Run:** 3 Epochs on a subset of 50,000 recipes, with a batch size of 8.
* **Result:** The model learned to generate basic, somewhat coherent instructions, demonstrating a reduction in training loss from Epoch 1 to Epoch 3. (You can paste the final training loss/PPL here if you want).
    * Example:
        ```
        Epoch: 01 | Time: 1255m 10s
                Train Loss: 5.715 | Train PPL: 303.266
        Epoch: 02 | Time: 318m 8s
                Train Loss: 5.355 | Train PPL: 211.733
        Epoch: 03 | Time: 704m 7s
                Train Loss: 5.246 | Train PPL: 189.777
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
    pip install transformers Pillow numpy pandas matplotlib seaborn
    ```
4.  **Download Dataset:**
    * Go to [Kaggle RecipeNLG Dataset](https://www.kaggle.com/datasets/saldenisov/recipenlg) and download `full_dataset.csv` (as zip, then unzip).
    * Create a `data/` folder in your project root, and then a `dataset/` folder inside it: `Recipe-Generation-from-Food-Image/data/dataset/`.
    * Move the downloaded `full_dataset.csv` into the `data/dataset/` folder.
5.  **Train the Model (Optional, model weights provided):**
    * The `seq2seq_recipe_model.pt` file (trained weights) is included in the repository.
    * If you wish to re-train (e.g., for more epochs or a larger subset):
        ```bash
        python train.py
        ```
        *(Note: Training can take many hours on CPU.)*
6.  **Run Inference (Generate Recipes):**
    ```bash
    python inference.py
    ```
    This script will load the saved model and generate example recipe directions based on predefined ingredient inputs.

## 8. Sample Generated Output
*(Paste the output you just got from `inference.py` here, exactly as it appeared. Even if it's repetitive, it shows the model running.)*