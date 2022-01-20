import torch.nn as nn


class LSTMDetector(nn.Module):
    def __init__(self, input_size, output_size, embedding_size, hidden_size, num_layers):
        super(LSTMDetector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)

        # input shape: seq_length, batchsize, embedding_dim
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            dropout=0.1, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (N, seq_length) where N is batch size
        embedding = self.embedding(x)
        # embedding shape: (N, seq_length, embedding_size)

        #         embedding = torch.transpose(embedding, 0, 1)
        #         print("Embedding shape", embedding.size())
        outputs, (h_n, h_c) = self.lstm(embedding)
        print(outputs.size())
        outputs = outputs.transpose(0, 1)
        outputs = self.fc(outputs)
        outputs = self.sigmoid(outputs)
        return outputs
