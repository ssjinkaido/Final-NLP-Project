import streamlit as st
import pickle
import numpy as np
import torch
from py_scripts.vocab import VocabChar
# from py_scripts.config import AttentionConfig, TransformerConfig
# from py_scripts.lstm_corrector import Encoder, Attention, Decoder, Seq2SeqLSTM
# from py_scripts.transformer_corrector import Seq2SeqTransformer
from py_scripts.lstm_detector import LSTMDetector
from py_scripts.transformer_detector import TransformerDetector
from nltk.tokenize import word_tokenize
import math
import torch.nn as nn
import torch.nn.functional as F


class TransformerConfig(object):
    lr = 0.0001
    SRC_VOCAB_SIZE = 232
    TARGET_VOCAB_SIZE = 232
    num_epochs = 20
    model_dim = 256
    feed_forward_dim = 1024
    num_layers = 6
    max_len = 40
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    train_loss = []
    valid_loss = []
    valid_acc = []
    epoch = []


class AttentionConfig(object):
    lr = 0.0005
    SRC_VOCAB_SIZE = 232
    TARGET_VOCAB_SIZE = 232
    num_epochs = 20
    max_len = 40
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    encoder_embedding_size = 128
    decoder_embedding_size = 128
    input_size_encoder = 232
    input_size_decoder = 232
    output_size = 232
    hidden_size = 1024
    num_layers = 1
    decoder_learning_ratio = 5.0

    train_loss = []
    valid_loss = []
    valid_acc = []
    epoch = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)

        # input shape: seq_length, batchsize, embedding_dim
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    #     def init_hidden(self):
    #         # This is what we'll initialise our hidden state as
    #         return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
    #                 torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def forward(self, x):
        # x shape: (N, seq_length) where N is batch size
        embedding = self.embedding(x)
        # embedding shape: (N, seq_length, embedding_size)

        #         embedding = torch.transpose(embedding, 0, 1)
        #         print("Embedding shape", embedding.size())
        outputs, (hidden_state, cell_state) = self.lstm(embedding)
        #         Embedding shape torch.Size([seq_len, batch_size, embedding_size])
        #         Outputs shape torch.Size([seq_len, batch_size, hidden_dim*2])
        #         Hidden state shape torch.Size([num_layer*2, batch_size, hidden_dim])
        #         Cell state shape torch.Size([num_layer*2, batch_size, hidden_dim])

        #         print("Outputs shape", outputs.size())
        #         print("Hidden state shape", hidden_state.size())
        #         print("Cell state shape", cell_state.size())
        #         print("Outputs shape", outputs.size())

        hidden_state = self.fc(torch.cat((hidden_state[0, :, :], hidden_state[1, :, :]), dim=1))
        cell_state = self.fc(torch.cat((cell_state[1, :, :], cell_state[0, :, :]), dim=1))
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, :self.hidden_size]

        #         print("Hidden state shape", hidden_state.size())
        #         print("Cell state shape", cell_state.size())
        #         print("Outputs shape", outputs.size())
        #         Hidden state shape torch.Size([batch_size, hidden_dim])
        #         Cell state shape torch.Size([batch_size, hidden_dim])
        #         Outputs shape torch.Size([seq_length, N, hidden_dim*2])
        # outputs shape: (seq_length, N, hidden_size)

        return outputs, hidden_state, cell_state


class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.V = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        seq_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(0)
        #         hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        #         encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #         print("Hidden after repeat", hidden.size())
        #         print("Encoder outputs now", encoder_outputs.size())
        # [batch_size, seq_len, 2 * enc_hidden_dim] output
        # [batch_size, seq_len, dec_hidden_dim] hidden
        energy = torch.sum(hidden * encoder_outputs, dim=2)
        #         energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch, seq_len, dec_hidden_dim]

        #         attention = self.V(energy).squeeze(dim = 2)
        attention = energy.t()
        # attention = [batch_size, seq_len]

        attention_weights = F.softmax(attention, dim=1)

        return attention_weights


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, attention):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(input_size=hidden_size + embedding_size, hidden_size=hidden_size, num_layers=num_layers)

        self.fc_out = nn.Linear((hidden_size) + hidden_size + embedding_size, output_size)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state, cell_state, encoder_outputs):
        x = x.unsqueeze(0)
        #         print("X shape", x.size())
        #         print("Decoder hidden state shape", hidden_state.size())
        #         print("Decoder cell state shape", cell_state.size())
        embedding = self.embedding(x)
        #         print("Embedding decoder shape", embedding.size())
        # embedding shape (batch_size, 1, embedding_size)

        a = self.attention(hidden_state, encoder_outputs)
        a = a.unsqueeze(1)

        #         print("Shape a", a.size())

        # a shape (batch_size, 1, seq_length)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        #         print("Weighted shape", weighted.size())
        # weighted shape (batch_size, 1, hidden_dim*2)
        weighted = weighted.permute(1, 0, 2)
        lstm_input = torch.cat((embedding, weighted), dim=2)
        #         print("LSTM input shape", lstm_input.size())
        # lstm_input shape = [batch_size, 1, enc_hidden_dim * 2 + embed_dim]

        output, (hidden_state, cell_state) = self.lstm(lstm_input, (hidden_state.unsqueeze(0), cell_state.unsqueeze(0)))

        embedding = embedding.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        #         print("Embedding shape", embedding.size())
        #         print("Output shape", output.size())
        #         print("Weighted", weighted.size())

        prediction = self.fc_out(torch.cat((output, weighted, embedding), dim=1))

        #         print("Predictions shape ", prediction.size())
        # (batch_size, output_dim(223))

        return prediction, hidden_state.squeeze(0), cell_state.squeeze(0), a.squeeze(1)


class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder, target_vocab_size):
        super(Seq2SeqLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab_size = target_vocab_size

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = target.shape[1]
        target_len = target.shape[0]

        #         print("Shape of target", target.size())
        outputs = torch.zeros(target_len, batch_size, self.target_vocab_size).to(device)

        encoder_outputs, hidden_state, cell_state = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        #         print("Target", target.size)
        #         print("Target shape", target.size())
        x = target[0, :]
        #         print("Shape of x", x.size())

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden_state, cell_state, _ = self.decoder(x, hidden_state, cell_state, encoder_outputs)
            # Store next output prediction
            #             output = output.unsqueeze(1)
            #             outputs[:, t, :] = output[:, 0, :]
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            #             best_guess = output.argmax(2).squeeze(1)
            best_guess = output.argmax(1)

            teacher_force = torch.rand(1).item() < teacher_force_ratio

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            #             x = target[:,t] if teacher_force else best_guess
            x = target[t, :] if teacher_force else best_guess
        #             print("X target shape", x.size())

        #         outputs = outputs.transpose(0, 1)
        return outputs


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


def correct_sentence_using_lstm(model, sentence, vocab):
    model.eval()
    encode_sentence = vocab.encode(sentence)
    print(encode_sentence)
    #     print("Encode sentence", encode_sentence)
    sentence_tensor = torch.LongTensor(encode_sentence).unsqueeze(0).transpose(1, 0)
    sentence_tensor = sentence_tensor.to(device)
    print(sentence_tensor)

    with torch.no_grad():
        encoder_outputs, hidden_state, cell_state = model.encoder(sentence_tensor)
        translated_sentence = [[1]]

        #         decoder_input = torch.tensor([1], device=device)
        #         decoded_words = []
        #         outputs = [1]
        for _ in range(50):
            #             print(translated_sentence[-1])
            target_input = torch.LongTensor(translated_sentence).to(device)
            #             previous_word = torch.LongTensor([outputs[-1]])
            #             previous_word = previous_word.to(device)
            output, hidden_state, cell_state, _ = model.decoder(target_input[-1], hidden_state, cell_state,
                                                                encoder_outputs)
            # print("Output size", output.size())
            output = output.argmax(1).tolist()
            # print(output)
            #             print("Output size", output.size())
            #             best_guess = output.argmax(1).item()
            #             topv, topi = torch.topk(output, 1)
            #             topi = topi[:, -1, 0]
            #             topi = topi.squeeze(0).tolist()
            translated_sentence.append(output)
            # print(translated_sentence)
        #             decoded_words.append(topi.item())
        #             print(decoded_words)
        #             if topi.item() == 2:
        #                 break
        #             decoder_input = topi.squeeze(dim=1)
        #             print(decoder_input)
        print("Translated sentence", translated_sentence)
        translated_sentence = np.squeeze(np.array(translated_sentence)).tolist()
    return vocab.decode(translated_sentence)


def correct_sentence_using_transformer(model, sentence, max_len, device, vocab):
    model.eval()
    encode_sentence = vocab.encode(sentence)
    sentence_tensor = torch.LongTensor(encode_sentence).unsqueeze(0).transpose(0, 1)
    sentence_tensor = sentence_tensor.to(device)
    translated_sentence = model.inference(sentence_tensor, max_len, device).detach().cpu().numpy().tolist()
    translated_sentence = vocab.decode(translated_sentence)
    return translated_sentence


class VocabWord(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.go = 1
        self.eos = 2

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

    def encode(self, chars):
        encode_sent = []
        encode_sent.append(self.go)
        for word in word_tokenize(chars):
            if not word in self.word2idx:
                encode_sent.append(self.word2idx['<unk>'])
            else:
                encode_sent.append(self.word2idx[word])
        encode_sent.append(self.eos)
        while len(encode_sent) < 10:
            encode_sent.append(0)

        return encode_sent

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]


if __name__ == "__main__":
    torch.cuda.empty_cache()
    captions = []
    onehot_labels = []
    vocab_word = VocabWord()
    with open('data/vocab.pkl', 'rb') as inp:
        vocab_word = pickle.load(inp)
    alphabets = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬ0bBcCdDđĐeEè1ÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈ2ĩĨíÍịỊjJkKlLmMnNoO3òÒỏỎõÕóÓọỌôÔồ4ỒổỔỗỖốỐộỘơƠờỜ5ởỞỡỠớỚợỢpP6qQrRsStTuUùÙủỦ7ũŨúÚụỤưƯừỪửỬữỮứỨựỰvVw8WxXyYỳỲỷỶ9ỹỸýÝỵỴzZ!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~ '
    vocab_char = VocabChar(alphabets)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    config_attn = AttentionConfig()
    config_trans = TransformerConfig()
    attention = Attention(config_attn.hidden_size, config_attn.hidden_size)
    encoder_net = Encoder(config_attn.input_size_encoder, config_attn.encoder_embedding_size, config_attn.hidden_size,
                          config_attn.num_layers).to(device)

    decoder_net = Decoder(config_attn.input_size_decoder, config_attn.decoder_embedding_size, config_attn.hidden_size,
                          config_attn.output_size, config_attn.num_layers, attention).to(device)

    model_lstm = Seq2SeqLSTM(encoder_net, decoder_net, config_attn.output_size).to(device)
    checkpoint_lstm = torch.load('model/model_attention_40.pth', map_location=torch.device('cpu'))
    checkpoint_transformer = torch.load('model/model_transformer_40.pth', map_location=torch.device('cpu'))
    model_lstm.load_state_dict(checkpoint_lstm['state_dict'])
    model_lstm.to(device)

    model_transformer = Seq2SeqTransformer(config_trans.SRC_VOCAB_SIZE, config_trans.model_dim,
                                           config_trans.feed_forward_dim,
                                           config_trans.TARGET_VOCAB_SIZE, config_trans.num_layers,
                                           config_trans.max_len, config_trans.PAD_IDX, config_trans.SOS_IDX,
                                           config_trans.EOS_IDX).to(
        device)
    model_transformer.load_state_dict(checkpoint_transformer['state_dict'])
    model_transformer.to(device)

    model_detector_lstm = LSTMDetector(len(vocab_word), 1, 256, 1024, 3)
    checkpoint_detector_lstm = torch.load('model/model_detector_lstm_1.pth', map_location=torch.device('cpu'))
    model_detector_lstm.load_state_dict(checkpoint_detector_lstm['state_dict'])
    model_detector_lstm.to(device)

    model_detector_transformer = TransformerDetector(len(vocab_word), 512,
                                                     2048,
                                                     1, 8,
                                                     config_trans.max_len, config_trans.PAD_IDX, config_trans.SOS_IDX,
                                                     config_trans.EOS_IDX)
    checkpoint_detector_transformer = torch.load('model/model_detector_transformer_1.pth',
                                                 map_location=torch.device('cpu'))
    model_detector_transformer.load_state_dict(checkpoint_detector_transformer['state_dict'])
    model_detector_transformer.to(device)

    st.title('PyTorch Spelling Correction')
    clicked_lstm = st.button("Correct and detect error using LSTM model", key='clicked_lstm')
    clicked_trans = st.button("Correct and detect error using Transformer model", key='clicked_trans')
    # clicked_detect_lstm = st.button("Detect error position using LSTM model", key='clicked_detect_lstm')
    # clicked_detect_trans = st.button("Detect error position using Transformer model", key='clicked_detect_trans')
    sentence = st.text_input('Input your sentence here:')
    if clicked_lstm:
        correct_sent = correct_sentence_using_lstm(model_lstm, sentence, vocab_char)
        st.write(f"Correction:   {correct_sent}")

        a = []
        model_detector_lstm.eval()
        sent_encode = vocab_word.encode(sentence)
        print(sent_encode)
        tensor_sent = torch.tensor(sent_encode, dtype=torch.long).unsqueeze(0).transpose(0, 1)
        tensor_sent = tensor_sent.to(device)
        with torch.no_grad():
            output = model_detector_lstm(tensor_sent)
            print("Output", output)
            pred = (output.detach().cpu().numpy() > 0.5).astype(int).squeeze(-1)
            pred = pred[0][1:]
            print(pred)
            for i in range(len(pred)):
                if pred[i] == 1:
                    a.append(i + 1)

        print(a)
        st.write(f"We found error at position {a}")

    if clicked_trans:
        a = []
        correct_sent = correct_sentence_using_transformer(model_transformer, sentence, config_trans.max_len, device,
                                                          vocab_char)
        model_detector_transformer.eval()
        sent_encode = vocab_word.encode(sentence)
        print(sent_encode)
        tensor_sent = torch.LongTensor(sent_encode).unsqueeze(0).transpose(0, 1)
        tensor_sent = tensor_sent.to(device)
        print(tensor_sent)
        with torch.no_grad():
            output = model_detector_transformer(tensor_sent)
            print(output)
            pred = (output.detach().cpu().numpy() > 0.5).astype(int).squeeze(-1)
            pred = pred[0][1:]
            print(pred)
            for i in range(len(pred)):
                if pred[i] == 1:
                    a.append(i + 1)

        st.write(f"Correction:  {correct_sent}")
        st.write(f"We found error at position {a}")

    # if clicked_detect_trans:
    #     model_detector_transformer.eval()
    #     sent_encode = vocab_word.encode(sentence)
    #     print(sent_encode)
    #     tensor_sent = torch.LongTensor(sent_encode).unsqueeze(0).transpose(0, 1)
    #     # tensor_sent = tensor_sent.to(device)
    #     print(tensor_sent)
    #     output = model_detector_transformer(tensor_sent)
    #     print(output)
    #     pred = (output.detach().cpu().numpy() > 0.5).astype(int).squeeze(-1)
    #     pred = pred[0][1:]
    #     print(pred)
    #     for i in range(len(pred)):
    #         if i == 1:
    #             st.write(f"Error at position {i}")
