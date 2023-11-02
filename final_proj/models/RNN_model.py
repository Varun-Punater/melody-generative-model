from pydantic import BaseModel
from torch import nn
import torch

class MusicRNNParams(BaseModel):
    vocab_dim: int # size of vocab along with PAD token
    embedding_dim: int = 32 # used to be 23
    hidden_dim: int = 128 # used to be 64
    num_layers: int = 1
    chord_dim: int # size of vocab along with PAD token
    chord_per_measure: int = 2

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]
    
class SelectLSTMItem(nn.Module):
    def __init__(self, item_index1, item_index2):
        super(SelectLSTMItem, self).__init__()
        self._name = 'selectitem'
        self.item_index1 = item_index1
        self.item_index2 = item_index2

    def forward(self, inputs):
        print(inputs[self.item_index1][self.item_index2].shape)
        return inputs[self.item_index1][self.item_index2]

class MusicRNN(nn.Module):
    def __init__(self, params: MusicRNNParams):
        super(MusicRNN, self).__init__()

        self.params = params

        self.embedding = nn.Embedding(
            num_embeddings=params.vocab_dim, 
            embedding_dim=params.embedding_dim
        )

        self.encoder = nn.LSTM( # changed from RNN
            input_size=params.embedding_dim,
            hidden_size=params.hidden_dim,
            num_layers=params.num_layers,
            batch_first=True,
        )

        self.hidden_linear = nn.Linear(
            in_features=params.hidden_dim,
            out_features=params.hidden_dim
        )

        self.decoder = nn.Linear(
            in_features=params.hidden_dim,
            out_features=params.chord_dim
        )

        nn.init.normal_(self.embedding.weight, mean=0, std=1)

        # self.encoder_decoder = nn.Sequential(
        #     nn.Embedding(
        #         num_embeddings=params.vocab_dim, 
        #         embedding_dim=params.embedding_dim
        #     ),
        #     nn.LSTM( # changed from RNN
        #         input_size=params.embedding_dim,
        #         hidden_size=params.hidden_dim,
        #         num_layers=params.num_layers,
        #         batch_first=True,
        #     ),
        #     SelectLSTMItem(1, 0),
        #     nn.Linear(
        #         in_features=params.hidden_dim,
        #         out_features=params.chord_dim
        #     )
        # )

    def forward(self, x):

        print("some inputs", x[15:20])

        embedded = self.embedding(x)

        print("some embeddings", embedded[15:20])

        print("embedded shape", embedded.shape)

        output, (final_hidden_state, final_cell_state) = self.encoder(embedded)

        print("lstm final hidden state shape", final_hidden_state[-1].shape)

        output = self.hidden_linear(final_hidden_state[-1])

        output = self.decoder(output)

        print("linear output shape", output.shape)
        print(output)

        return output   
        
