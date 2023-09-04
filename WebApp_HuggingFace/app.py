import streamlit as st
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import re
import plotly.graph_objects as go

# Assuming TransformerEncoder and TransformerDecoder are defined above
EMBEDDING_DIM = 16
HIDDEN_DIM = 16
LATENT_DIM = 16 # Dimension of the latent space
SEQ_LEN = 16 # Max length of the sequence
NHEAD = 4 # Number of heads in multi-head attention
NUM_LAYERS = 2 # Number of transformer layers

# Gumbel softmax temperature
TAU = 1.0

LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.random.manual_seed(1024)

# Pass embeded into decoder instead of using the original x
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=EMBEDDING_DIM, nhead=NHEAD, num_layers=NUM_LAYERS):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers
        )
        self.fc_logits = nn.Linear(d_model, LATENT_DIM)

    def forward(self, x):
        embedded = self.embedding(x).permute(1, 0, 2)  # Transformer expects seq_len, batch, features
        transformed = self.transformer_encoder(embedded)
        # Use the final state to predict logits for latent space
        logits = self.fc_logits(transformed[-1])
        return logits, embedded


class TransformerDecoder(nn.Module):
    def __init__(self, d_model=EMBEDDING_DIM, nhead=NHEAD, num_layers=NUM_LAYERS):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead), num_layers
        )
        self.fc_out = nn.Linear(d_model, VOCAB_SIZE)
        self.fc_z = nn.Linear(LATENT_DIM, d_model)  # Convert z to feature size for transformer

    def forward(self, embedded, z):
        # embedded = self.embedding(x).permute(1, 0, 2) # Transformer expects [seq_len, batch, features], permute函数用于改变张量的维度顺序
        z_adjusted = self.fc_z(z).unsqueeze(0)
        output = self.transformer_decoder(embedded, z_adjusted)
        return self.fc_out(output.permute(1, 0, 2))


class TransformerCVAE(nn.Module):
    def __init__(self):
        super(TransformerCVAE, self).__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()

    def reparameterize(self, logits):
        return F.gumbel_softmax(logits, tau=TAU, hard=False, dim=-1)

    def forward(self, x):
        logits, emb = self.encoder(x)
        z = self.reparameterize(logits)
        return self.decoder(emb, z), logits
    
def load_and_preprocess_wikitext(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Use regular expressions to split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [sentence.strip() for sentence in sentences]
    
    return sentences

train_file_path = "wikitext-2/wiki.train.tokens"
test_file_path = "wikitext-2/wiki.test.tokens"
val_file_path = "wikitext-2/wiki.valid.tokens"

wikitext_sentences_train = load_and_preprocess_wikitext(train_file_path)
wikitext_sentences_test = load_and_preprocess_wikitext(test_file_path)
wikitext_sentences_val = load_and_preprocess_wikitext(val_file_path)

# Hyperparameters
BATCH_SIZE = 32
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Tokenize the data
tokens = [word for sentence in wikitext_sentences_train for word in sentence.split()]

# Build vocabulary
vocab = [PAD_TOKEN, UNK_TOKEN] + list(set(tokens))
word_index = {word: index for index, word in enumerate(vocab)}
# 添加新的tokens
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
word_index[SOS_TOKEN] = len(word_index)
word_index[EOS_TOKEN] = len(word_index)
vocab = {v: k for k, v in word_index.items()}
# Convert tokens to integers
def tokenize_and_encode(text):
    return [word_index.get(word, word_index[UNK_TOKEN]) for word in text.split()]

encoded_data_train = [tokenize_and_encode(sentence) for sentence in wikitext_sentences_train]

# Create a PyTorch Dataset
class WikiDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if len(sample) < self.sequence_length:
            sample.extend([word_index[PAD_TOKEN]] * (self.sequence_length - len(sample)))
        else:
            sample = sample[:self.sequence_length]
        return torch.tensor(sample)

# dataset = WikiDataset(encoded_data_train, SEQUENCE_LENGTH)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# Split the data into train and validation sets
dataset = WikiDataset(encoded_data_train, SEQ_LEN)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

VOCAB_SIZE = len(vocab)

class MultiMultiSignalingGame:
    def __init__(self, senders: list, receivers: list, optimizer, criterion):
        self.senders = senders
        self.receivers = receivers
        self.optimizer = optimizer
        self.criterion = criterion

    def play_round(self, states):
        all_decoded_outputs = []
        all_logits = []
        interactions = []

        for i, sender in enumerate(self.senders):
            # Sender encodes the state
            logits, emb = sender(states[i])
            all_logits.append(logits)
            z = F.gumbel_softmax(logits, tau=TAU, hard=False, dim=-1)
            
            _, input_sentence_ids = torch.max(states[i], dim=1)
            input_sentence_ids = input_sentence_ids.cpu().numpy()
            input_sentence = ' '.join([vocab[idx] for idx in input_sentence_ids])
            
            # Each receiver decodes the signal from the sender
            for j, receiver in enumerate(self.receivers):
                decoded_output = receiver(emb, z)
                all_decoded_outputs.append(decoded_output)

                _, output_sentence_ids = torch.max(decoded_output[0], dim=1)
                output_sentence_ids = output_sentence_ids.cpu().numpy()
                output_sentence = ' '.join([vocab[idx] for idx in output_sentence_ids])
                
                interactions.append((i, j, input_sentence, output_sentence))
      
        # Calculate loss
        loss, recon_loss, kld_loss = self.compute_loss(states, all_decoded_outputs, all_logits, beta=1.0)
        
        # Update model parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), recon_loss.item(), kld_loss.item(), interactions

    def compute_loss(self, original_states, decoded_states, logits, beta):
        recon_loss = sum([self.criterion(decoded_state.view(-1, VOCAB_SIZE), original_state.view(-1))
                          for original_state, decoded_state in zip(original_states * len(self.receivers), decoded_states)])
        
        # Calculate KLD loss
        kld_losses = []
        for logit in logits:
            mean, logvar = torch.chunk(logit, 2, dim=-1)
            kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            kld_losses.append(kld_loss)

        return recon_loss + beta * sum(kld_losses), recon_loss, sum(kld_losses)
    

def run_signal_game(NUM_SENDERS, NUM_RECEIVERS, num_rounds):
    # para_checker = st.empty()
    # para_checker.text(f"NUM_SENDERS: {NUM_SENDERS}, NUM_RECEIVERS: {NUM_RECEIVERS}, num_rounds: {num_rounds}, EMBEDDING_DIM: {EMBEDDING_DIM}, HIDDEN_DIM: {HIDDEN_DIM}, LATENT_DIM: {LATENT_DIM}, SEQ_LEN: {SEQ_LEN}, TAU: {TAU}, nhead: {NHEAD}, num_layers: {NUM_LAYERS}, BATCH_SIZE: {BATCH_SIZE}")
    senders = [TransformerEncoder().to(device) for _ in range(NUM_SENDERS)]
    receivers = [TransformerDecoder().to(device) for _ in range(NUM_RECEIVERS)]

    params = [list(sender.parameters()) for sender in senders]
    params.extend([list(receiver.parameters()) for receiver in receivers])
    # optimizer = torch.optim.Adam([param for sublist in params for param in sublist], lr=0.001)
    if OPTMIZER == "Adam":
        optimizer = torch.optim.Adam([param for sublist in params for param in sublist], lr=LEARNING_RATE)
    elif OPTMIZER == "AdamW":
        optimizer = torch.optim.AdamW([param for sublist in params for param in sublist], lr=LEARNING_RATE)
    elif OPTMIZER == "SGD":
        optimizer = torch.optim.SGD([param for sublist in params for param in sublist], lr=LEARNING_RATE)
    
    criterion = torch.nn.CrossEntropyLoss()

    game = MultiMultiSignalingGame(senders, receivers, optimizer, criterion)

    losses = []
    recon_losses = []
    kld_losses = []
    input_sentences = []
    output_sentences = []

    # Use Streamlit's progress bar
    loss_plot_placeholder = st.empty()  # 创建一个空位占位符来显示损失图
    progress_bar = st.progress(0)
    interactions_placeholder = st.empty()  # 创建一个空位占位符来显示交互

    for round in range(num_rounds):
        states = [torch.randint(VOCAB_SIZE, (BATCH_SIZE, 16)).to(device) for _ in range(NUM_SENDERS)]
        loss, recon_loss, kld_loss, interactions = game.play_round(states)
        losses.append(loss)
        recon_losses.append(recon_loss)
        kld_losses.append(kld_loss)
        # 刷新显示每轮的损失
        fig, ax = plt.subplots()
        ax.plot(losses, label='Total Losses', color='blue')
        ax.plot(recon_losses, label='Reconstruction Losses', color='green')
        ax.plot(kld_losses, label='KLD Losses', color='red')
        ax.set_xlabel('Round')
        ax.set_ylabel('Loss')
        ax.legend()
        loss_plot_placeholder.pyplot(fig)
        # Close the figure to free up memory
        plt.close(fig)
        
        progress_bar.progress(round / num_rounds)
        # 刷新显示每次交互的句子
        interaction_str = "\n\n".join([f"Sender {i} -> Receiver {j}\nSend(encode): {input_sentence}\nReceive(decode): {output_sentence}" 
                                       for i, j, input_sentence, output_sentence in interactions])
        interactions_placeholder.text(interaction_str)

    # Dynamic plotting of the losses
    fig, ax = plt.subplots()
    ax.plot(losses, label='Total Losses', color='blue')
    ax.plot(recon_losses, label='Reconstruction Losses', color='green')
    ax.plot(kld_losses, label='KLD Losses', color='red')
    ax.set_xlabel('Round')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

# Streamlit UI
st.title('Multi-Agents Signal Game')

NUM_SENDERS = st.sidebar.slider("NUM_SENDERS", 1, 3, 2)
NUM_RECEIVERS = st.sidebar.slider("NUM_RECEIVERS", 1, 3, 2)
num_rounds = st.sidebar.slider("NUM_ROUNDS", 1000, 100000, 10000, 1000)

advanced_settings = st.sidebar.expander("Advanced settings")
with advanced_settings:
    use_cosine_annealing = st.checkbox("USE ANNEALING")
    if use_cosine_annealing:
        annealing_strategy = st.selectbox("ANNEALING STRATEGY", ["linear", "cosine"])
        TAU = st.slider("START TEMP.", 0.1, 10.0, 1.0)
        final_tau = st.slider("FINAL TEMP.", 0.1, 10.0, 1.0)
    else:
        annealing_strategy = None
        TAU = st.slider("TEMP.", 0.1, 10.0, 1.0)

    optimizer_options = ["Adam", "AdamW", "SGD"]
    OPTMIZER = st.selectbox("OPTIMIZER", optimizer_options)
    LEARNING_RATE = st.slider("LEARNING RATE", 1e-5, 1e-2, 1e-3, format="%.5f")

    EMBEDDING_DIM = st.slider("EMBEDDING_DIM", 1, 128, 16)
    HIDDEN_DIM = st.slider("HIDDEN_DIM", 1, 128, 16)
    LATENT_DIM = st.slider("LATENT_DIM", 1, 128, 16)
    SEQ_LEN = st.slider("SEQ_LEN", 1, 128, 16)
    NHEAD = st.slider("NHEAD", 1, 8, 4)
    NUM_LAYERS = st.slider("NUM_LAYERS", 1, 6, 2)
    BATCH_SIZE = st.slider("BATCH_SIZE", 1, 128, 32)

if st.sidebar.button('Start'):
    run_signal_game(NUM_SENDERS, NUM_RECEIVERS, num_rounds)
