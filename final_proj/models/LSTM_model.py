from pydantic import BaseModel
from torch import nn
import torch

class MusicRNNParams(BaseModel):
    vocab_dim: int = 16 # size of vocab along with PAD token
    embedding_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 1
    chord_dim: int = 148 # size of vocab along with PAD token
    chord_per_measure: int = 2

class SelectItem(nn.Module):
    def __init__(self, item_index_1, item_index_2):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index_1 = item_index_1
        self.item_index_2 = item_index_2

    def forward(self, inputs):
        return inputs[self.item_index_1][self.item_index_2][-1]

class MusicRNN(nn.Module):
    def __init__(self, params: MusicRNNParams):
        super(MusicRNN, self).__init__()

        self.encoder_decoder = nn.Sequential(
            nn.Embedding(
                num_embeddings=params.vocab_dim, 
                embedding_dim=params.embedding_dim
            ),
            nn.LSTM(
                input_size=params.embedding_dim,
                hidden_size=params.hidden_dim,
                num_layers=params.num_layers,
                batch_first=True
            ),
            SelectItem(item_index_1=1, item_index_2=0),
            nn.Linear(
                in_features=params.hidden_dim,
                out_features=params.chord_dim
            )
        )

    def forward(self, x):
        return self.encoder_decoder(x)
