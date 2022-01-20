import torch
import torch.nn as nn
import math
class PositionEncoder(nn.Module):
    def __init__(self, max_len, emb_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, num_layer, max_len=7):
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.scale = math.sqrt(emb_size)

        self.embedding = nn.Embedding(input_size, emb_size)
        # additional length for sos and eos
        self.pos_encoder = PositionEncoder(max_len + 1, emb_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=8,
                                                   dim_feedforward=hidden_size,
                                                   dropout=0.1, activation='gelu')
        encoder_norm = nn.LayerNorm(emb_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer, norm=encoder_norm)

    def forward(self, src):
        src = self.embedding(src) * self.scale
        src = self.pos_encoder(src)
        output = self.encoder(src)
        return output
class TransformerDetector(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size,
                 num_layer, max_len, pad_token, sos_token, eos_token):
        super().__init__()
        self.encoder = TransformerEncoder(input_size, emb_size, hidden_size, num_layer, max_len)
        self.fc = nn.Linear(emb_size, output_size)
        self.pad_token = pad_token
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def generate_mask(src, pad_token):
        '''
        Generate mask for tensor src
        :param src: tensor with shape (max_src, b)
        :param pad_token: padding token
        :return: mask with shape (b, max_src) where pad_token is masked with 1
        '''
        mask = (src.t() == pad_token)
        return mask.to(src.device)

    def forward(self, src):
        src_mask = self.generate_mask(src, self.pad_token)
        enc_output = self.encoder(src)
        enc_output = self.fc(enc_output)
        output = enc_output.transpose(0, 1)
        output = self.sigmoid(output)

        return output