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
