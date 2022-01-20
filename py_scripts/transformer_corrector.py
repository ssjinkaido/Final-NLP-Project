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
    def __init__(self, input_size, emb_size, hidden_size, num_layer, max_len=64):
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.scale = math.sqrt(emb_size)

        self.embedding = nn.Embedding(input_size, emb_size)
        # additional length for sos and eos
        self.pos_encoder = PositionEncoder(max_len + 10, emb_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=8,
                                                   dim_feedforward=hidden_size,
                                                   dropout=0.1, activation='gelu')
        encoder_norm = nn.LayerNorm(emb_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer, norm=encoder_norm)

    def forward(self, src, src_mask):
        src = self.embedding(src) * self.scale
        src = self.pos_encoder(src)
        output = self.encoder(src, src_key_padding_mask=src_mask)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size, num_layer, max_len=64):
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.scale = math.sqrt(emb_size)

        self.embedding = nn.Embedding(output_size, emb_size)
        self.pos_encoder = PositionEncoder(max_len + 10, emb_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_size, nhead=8,
                                                   dim_feedforward=hidden_size,
                                                   dropout=0.1, activation='gelu')
        decoder_norm = nn.LayerNorm(emb_size)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layer, norm=decoder_norm)
        self.fc = nn.Linear(emb_size, output_size)

    def forward(self, trg, enc_output, sub_mask, mask):
        #         print(trg.size())
        trg = self.embedding(trg) * self.scale
        #         print(trg.size())
        trg = self.pos_encoder(trg)
        #         print(trg.size())
        #         print("Target", trg.size())
        #         print("Target sub mask", sub_mask.size())
        #         print("Target mask", mask.size())
        output = self.decoder(trg, enc_output, tgt_mask=sub_mask, tgt_key_padding_mask=mask)
        #         print(output.size())
        output = self.fc(output)
        return output


class Seq2SeqTransformer(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size,
                 num_layer, max_len, pad_token, sos_token, eos_token):
        super(Seq2SeqTransformer, self).__init__()
        self.encoder = TransformerEncoder(input_size, emb_size, hidden_size, num_layer, max_len)
        self.decoder = TransformerDecoder(output_size, emb_size, hidden_size, num_layer, max_len)
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.encoder.apply(self.initialize_weights)
        self.decoder.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

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

    @staticmethod
    def generate_submask(src):
        sz = src.size(0)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(src.device)

    def forward(self, src, trg):
        #         print("Src size", src.size())
        #         print("Target size", trg.size())
        src_mask = self.generate_mask(src, self.pad_token)
        trg_mask = self.generate_mask(trg, self.pad_token)
        #         print("Src mask size", src_mask.size())
        #         print("Target mask size", trg_mask.size())
        trg_submask = self.generate_submask(trg)
        #         print("Target submask size", trg_submask.size())
        enc_output = self.encoder(src, src_mask)
        #         print("Encoding_output size", enc_output.size())
        output = self.decoder(trg, enc_output, trg_submask, trg_mask)
        return output

    def inference(self, src, max_len, device):
        #         assert src.dim() == 1, 'Can only translate one sentence at a time!'
        #         assert src.size(0) <= max_len + 2, f'Source sentence exceeds max length: {max_len}'

        #         src.unsqueeze_(-1)

        src_mask = self.generate_mask(src, self.pad_token)
        enc_output = self.encoder(src, src_mask)
        #         device = src.device

        trg_list = [self.sos_token]
        for idx in range(max_len):
            trg = torch.tensor(trg_list, dtype=torch.long, device=device).unsqueeze(-1)
            trg_mask = self.generate_mask(trg, self.pad_token)
            trg_submask = self.generate_submask(trg)
            output = self.decoder(trg, enc_output, trg_submask, trg_mask)
            pred = torch.argmax(output.squeeze(1), dim=-1)[-1].item()
            trg_list.append(pred)
            if pred == self.eos_token:
                break
        return torch.tensor(trg_list[1:], dtype=torch.long, device=device)