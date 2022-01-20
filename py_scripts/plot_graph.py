import torch
import matplotlib.pyplot as plt
from py_scripts.config import AttentionConfig, TransformerConfig
from py_scripts.lstm_corrector import Encoder, Attention, Decoder, Seq2SeqLSTM
import numpy as np
from py_scripts.transformer_corrector import Seq2SeqTransformer


def plot_lstm_correction():
    config_attn = AttentionConfig()
    attention = Attention(config_attn.hidden_size, config_attn.hidden_size)
    encoder_net = Encoder(config_attn.input_size_encoder, config_attn.encoder_embedding_size, config_attn.hidden_size,
                          config_attn.num_layers)

    decoder_net = Decoder(config_attn.input_size_decoder, config_attn.decoder_embedding_size, config_attn.hidden_size,
                          config_attn.output_size, config_attn.num_layers, attention)

    model_lstm = Seq2SeqLSTM(encoder_net, decoder_net, config_attn.output_size)
    checkpoint_lstm = torch.load('../data/model_attention_40.pth', map_location=torch.device('cpu'))
    model_lstm.load_state_dict(checkpoint_lstm['state_dict'])
    train_loss = checkpoint_lstm['train_loss']
    valid_acc = checkpoint_lstm['valid_acc']
    valid_acc.insert(0, 0)
    t = np.arange(1, 41, 1)

    fig, ax1 = plt.subplots()
    ax1.set_title('LSTM Corrector')
    color = 'tab:blue'
    ax1.plot(t, train_loss, color=color)

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.plot([0, 10, 20, 30, 40], valid_acc, color=color)
    ax2.set_ylabel("Valid_Acc")
    ax2.tick_params(axis='y', labelcolor=color)
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def plot_transformer_correction():
    config_trans = TransformerConfig()
    model_transformer = Seq2SeqTransformer(config_trans.SRC_VOCAB_SIZE, config_trans.model_dim,
                                           config_trans.feed_forward_dim,
                                           config_trans.TARGET_VOCAB_SIZE, config_trans.num_layers,
                                           config_trans.max_len, config_trans.PAD_IDX, config_trans.SOS_IDX,
                                           config_trans.EOS_IDX)

    checkpoint_transformer = torch.load('../data/model_transformer_40.pth', map_location=torch.device('cpu'))
    model_transformer.load_state_dict(checkpoint_transformer['state_dict'])
    train_loss = checkpoint_transformer['train_loss']
    valid_acc = checkpoint_transformer['valid_acc']
    print(train_loss)
    print(valid_acc)
    valid_acc.insert(0, 0)
    print(valid_acc)
    t = np.arange(1, 41, 1)
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.plot(t, train_loss, color=color)
    ax1.set_title('Transformer Corrector')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.plot([0, 10, 30, 40], valid_acc, color=color)
    ax2.set_ylabel("Valid acc")
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


plot_lstm_correction()
plot_transformer_correction()
