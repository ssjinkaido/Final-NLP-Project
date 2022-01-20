import torch
import torch.nn as nn
import torch.nn.functional as F


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