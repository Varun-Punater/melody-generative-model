from pydantic import BaseModel
from torch import nn


class MusicRNNParams(BaseModel):
    vocab_dim: int = 17
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    chord_dim: int # calculate chord dims by going through every song and checking unique chords


class MusicRNN(nn.Module):
    def __init__(self, params: MusicRNNParams):
        super(MusicRNN, self).__init__()

        self.encoder_decoder = nn.Sequential(
            nn.Embedding(
                num_embeddings=params.vocab_dim, 
                embedding_dim=params.embedding_dim
            ),
            nn.RNN()
        )