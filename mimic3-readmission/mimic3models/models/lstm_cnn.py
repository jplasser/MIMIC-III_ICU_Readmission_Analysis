from torch import nn

class LSTM_CNN(nn.Module):
    def __init__(self, input_dim=390, hidden_dim, lstm_layers=1, bidirectional=False, dense=False):

        #dim, batch_norm, dropout, rec_dropout, task,
        #target_repl = False, deep_supervision = False, num_classes = 1,
        #depth = 1, input_dim = 390, ** kwargs

        super(LSTM_CNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = lstm_layers
        self.bidirectional = bidirectional
        self.dense = dense

        # some more parameters
        self.output_dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth
        self.dropout_words = 0.3
        self.dropout_rnn_U = 0.3
        self.drop_conv = 0.5

        # define the LSTM layer
        # in keras we have inputs: A 3D tensor with shape [batch, timesteps, feature]
        # units: Positive integer, dimensionality of the output space. = dim=num_units=hidden_size
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.layers,
                            dropout=self.rec_dropout,
                            bidirectional=self.bidirectional)

        # this is not in the original model
        self.act1 = nn.ReLU()

        self.do1 = nn.Dropout(self.dropout)
        self.cnn = nn.Conv1d()
        # concat the three outputs from the CNN
        self.do2 = nn.Dropout(self.drop_conv)
        self.dense = nn.Linear(self.hidden_dim, self.num_classes)

        # change linear layer inputs depending on if lstm is bidrectional
        #if not bidirectional:
        #    self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        #else:
        #    self.linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        #self.act2 = nn.ReLU()

        # change linear layer inputs depending on if lstm is bidrectional and extra dense layer isn't added
        if bidirectional and not dense:
            self.final = nn.Linear(self.hidden_dim * 2, 1)
        else:
            self.final = nn.Linear(self.hidden_dim, 1)


    def forward(self, inputs, labels=None):
        out = inputs.unsqueeze(1)
        out, h = self.lstm(out)
        out = self.act1(out)
        #if self.dense:
        #    out = self.linear(out)
        #    out = self.act2(out)
        out = self.final(out)
        return out