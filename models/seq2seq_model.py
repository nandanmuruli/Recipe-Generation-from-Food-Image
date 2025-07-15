import torch
import torch.nn as nn
import random

# Define the Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.5):
        super(EncoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        # Embedding layer to convert word IDs to vectors
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        # GRU (Gated Recurrent Unit) for sequence processing
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_seq):
        # input_seq: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(input_seq))
        # embedded: (batch_size, seq_len, hidden_dim)
        output, hidden = self.gru(embedded)
        # output: (batch_size, seq_len, hidden_dim) - all hidden states
        # hidden: (1, batch_size, hidden_dim) - last hidden state
        return output, hidden

# Define the Decoder (simple version without attention for now)
class DecoderRNN(nn.Module):
    def __init__(self, output_dim, hidden_dim, dropout_rate=0.5):
        super(DecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_token, hidden):
        # input_token: (batch_size, 1) - current input token ID
        # hidden: (1, batch_size, hidden_dim) - previous hidden state from encoder or previous decoder step

        embedded = self.dropout(self.embedding(input_token))
        # embedded: (batch_size, 1, hidden_dim)

        output, hidden = self.gru(embedded, hidden)
        # output: (batch_size, 1, hidden_dim)
        # hidden: (1, batch_size, hidden_dim)

        prediction = self.softmax(self.out(output.squeeze(1)))
        # prediction: (batch_size, output_dim) - log probabilities for next token

        return prediction, hidden

# Define the full Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        # Ensure hidden dimensions match
        assert encoder.hidden_dim == decoder.hidden_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        # source: (batch_size, src_seq_len) - e.g., numericalized ingredients
        # target: (batch_size, trg_seq_len) - e.g., numericalized directions

        batch_size = target.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)

        # Encoder outputs (all hidden states) and final hidden state
        encoder_output, encoder_hidden = self.encoder(source)

        # First input to the decoder is the <BOS> token
        input_token = target[:, 0].unsqueeze(1) # Take the <BOS> token for each sequence

        for t in range(1, target_len):
            # Send current input token and hidden state to decoder
            output, hidden = self.decoder(input_token, encoder_hidden)

            # Store prediction for current token
            outputs[:, t, :] = output

            # Decide if we use teacher forcing (actual target token) or decoder's own prediction for next input
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = target[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs

# Example Usage (will be done in a separate training script later)
# This just shows how the model is initialized conceptually
if __name__ == '__main__':
    INPUT_DIM = 10000 # Example vocab size
    OUTPUT_DIM = 10000 # Example vocab size
    HIDDEN_DIM = 256
    DROPOUT_RATE = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    encoder = EncoderRNN(INPUT_DIM, HIDDEN_DIM, DROPOUT_RATE)
    decoder = DecoderRNN(OUTPUT_DIM, HIDDEN_DIM, DROPOUT_RATE)

    model = Seq2Seq(encoder, decoder, device).to(device)

    print(model)
    print(f"Model will run on: {device}")