from abc import ABC, abstractmethod

import math
from typing import Union, Optional, Any

import torch
from beartype import beartype
from torch import Tensor, nn
from torch.nn import functional as F
from ssm import S4Block as S4
from charactertokenizer import CharacterTokenizer


# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d


class LinearRNNCell(nn.Module):
    """
    A linear RNN cell without nonlinearities, similar to PyTorch's RNNCell but without activation functions.
    
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights b_ih and b_hh
    """
    
    def __init__(self, input_size, hidden_size, bias=True, mlp=False):
        super(LinearRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mlp = mlp
        self.bias = bias
        # Weight matrices
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        # Optional bias
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        # Use standard initialization method from PyTorch
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        if self.bias:
            nn.init.zeros_(self.bias_ih)
            nn.init.zeros_(self.bias_hh)
            
    def forward(self, input, hx=None):
        """
        Forward pass of the linear RNN cell.
        
        Args:
            input: tensor of shape (batch, input_size) containing input features
            hx: tensor of shape (batch, hidden_size) containing the initial hidden state
                or None for zero initial hidden state
                
        Returns:
            h': tensor of shape (batch, hidden_size) containing the next hidden state
        """
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, 
                             dtype=input.dtype, device=input.device)
        
        
        # MLP
        if self.mlp:
            h_next = F.linear(input, self.weight_ih, self.bias_ih) 
        else: 
            #Linear transformations
            h_next = F.linear(input, self.weight_ih, self.bias_ih) + \
                     F.linear(hx, self.weight_hh, self.bias_hh)
        
            # No activation function
        return h_next 


class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=128,
        d_state=64,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4(d_model, d_state=d_state, dropout=dropout, transposed=True)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Decode the outputs
        x = self.decoder(x)  # (B, L, d_model) -> (B, L, d_output)

        return x


class DeepRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, activation='relu', readout_activation='identity', rnncell='rnn', residual=False, mlp=False):
        super(DeepRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.residual = residual
        self.mlp = mlp
        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'tanh':
            self.activation_fn = torch.tanh
        elif activation == 'identity':
            self.activation_fn = lambda x: x
        else:
            raise ValueError("activation must be 'relu', 'tanh', or 'identity'")
        
        if readout_activation == 'relu':
            self.readout_activation_fn = F.relu
        elif readout_activation == 'tanh':
            self.readout_activation_fn = torch.tanh
        elif readout_activation == 'identity':
            self.readout_activation_fn = lambda x: x
        else:
            raise ValueError("readout_activation must be 'relu', 'tanh', or 'identity'")
        
        if rnncell == 'linear':
            self.rnn_cell = LinearRNNCell
        elif rnncell == 'rnn':
            self.rnn_cell = nn.RNNCell
        
        self.rnn_layers = nn.ModuleList([
            self.rnn_cell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: Any, batch_first=True) -> Any:
        if batch_first:
            # Swap the first two dimensions (batch, seq_len, ...) -> (seq_len, batch, ...)
            x = x.transpose(0, 1)

        # Extract sequence length and batch size for either 2D or 3D input
        seq_length = x.size(0)
        batch_size = x.size(1)
        h = self.init_hidden(batch_size)
        outputs = []
        
        for t in range(seq_length):
            out, h = self.forward_one_timestep(x[t], h)
            outputs.append(out)

        outputs = torch.stack(outputs)

        if batch_first:
            # Swap back two dimensions (seq_len, batch, ...) -> (batch, seq_len, ...)
            outputs = outputs.transpose(0, 1)

        return outputs 

    def forward_one_timestep(self, x, h):
        h_depth = []
        h_rec = []
        for i, rnn in enumerate(self.rnn_layers):
            h_i = rnn(x if i == 0 else h_depth[i-1], h[i])
            h_rec.append(h_i)
            h_i = self.activation_fn(h_i)
            if i>0 and self.residual:
                h_i = h_i + h_depth[i-1]
            h_depth.append(h_i)
        out = self.fc(h_depth[-1])
        out = self.readout_activation_fn(out)
        
        return out, torch.stack(h_rec)
        
    def init_hidden(self, batch_size):
        return [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]


class SimpleSeq2SeqRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(SimpleSeq2SeqRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Embedding layer to convert input tokens to vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Single RNN cell
        self.rnn = nn.RNNCell(embedding_dim, hidden_size)
        
        # Output projection to vocabulary size (no activation - raw logits)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        """
        x: Input sequence tensor of shape (seq_len, batch_size) containing token indices
        hidden: Initial hidden state of shape (batch_size, hidden_size) or None
        
        Returns:
        - outputs: Tensor of shape (seq_len, batch_size, output_size) containing logits
        - hidden: Final hidden state
        """
        seq_len, batch_size = x.size()
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Process the sequence one step at a time
        outputs = []
        for t in range(seq_len):
            # Get current input tokens and convert to embeddings
            emb = self.embedding(x[t])  # (batch_size, embedding_dim)
            
            # Update hidden state
            hidden = self.rnn(emb, hidden)
            
            # Project to output size (logits)
            output = self.fc(hidden)  # (batch_size, output_size)
            outputs.append(output)
        
        # Stack outputs along sequence dimension
        outputs = torch.stack(outputs)  # (seq_len, batch_size, output_size)
        
        return outputs, hidden


class DeepRNNWithEmbedding(DeepRNN):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers, activation='relu', readout_activation='identity', rnncell='rnn',residual=False, mlp=False):
        if embedding_dim: 
            super().__init__(embedding_dim, hidden_size, output_size, num_layers, activation, readout_activation, rnncell, residual, mlp)
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else: 
            super().__init__(vocab_size, hidden_size, output_size, num_layers, activation, readout_activation, rnncell, residual, mlp)
            self.embedding_dim = embedding_dim
    
    def forward(self, x, batch_first=True):
        if self.embedding_dim: 
            x = self.embedding(x)
        return super().forward(x, batch_first=batch_first)


class S4ModelWithEmbedding(S4Model):
    def __init__(
        self,
        d_input,
        embedding_dim,
        d_output=10,
        d_model=128,
        d_state=64,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        padding_idx=None,
    ):
        super().__init__(embedding_dim, d_output, d_model, d_state, n_layers, dropout, prenorm)
        self.embedding = nn.Embedding(d_input, embedding_dim, padding_idx=padding_idx)

    def forward(self, x):
        x = self.embedding(x)
        return super().forward(x)

class CPRNN(nn.Module):
    """CP-Factorized LSTM. Outputs logits (no softmax)

    Args:
        hidden_size: Dimension of hidden features.
        vocab_size: Size of vocabulary
        use_embedding: Whether to use embedding layer or one-hot encoding
        rank: Rank of cp factorization
        tokenizer: Character tokenizer
        batch_first: Whether to use batch first or not
        dropout: Dropout rate

    """
    def __init__(self, vocab_size : int, hidden_size: int, embedding_dim: Optional[int] = None, rank: int = 8,
                 tokenizer: CharacterTokenizer = None, batch_first: bool = True, dropout: float = 0.5,
                 gate: str = 'tanh', **kwargs):
        super().__init__()

        self.dropout = dropout
        self.batch_first = batch_first
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.rank = rank
        self.gate = {"tanh": torch.tanh, "sigmoid": torch.sigmoid, "identity": lambda x: x}[gate]

        # Define embedding and decoder layers
        if embedding_dim != None:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
            self.input_size = self.embedding_dim
        else:
            self.input_size = self.vocab_size

        self.decoder = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.vocab_size)
        )

        # Encoder using CP factors
        self.A = nn.Parameter(torch.Tensor(self.hidden_size, self.rank))
        self.B = nn.Parameter(torch.Tensor(self.input_size, self.rank))
        self.C = nn.Parameter(torch.Tensor(self.hidden_size, self.rank))
        self.U = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.V = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.d = nn.Parameter(torch.Tensor(self.hidden_size))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_hidden(self, batch_size, device=torch.device('cpu')):
        h = torch.zeros(batch_size, self.hidden_size).to(device)
        return h

    def predict(self, inp: Union[torch.LongTensor, str], init_states: tuple = None, top_k: int = 1,
                device=torch.device('cpu')):

        with torch.no_grad():

            if isinstance(inp, str):
                if self.tokenizer is None:
                    raise ValueError("Tokenizer not defined. Please provide a tokenizer to the model.")
                x = torch.tensor(self.tokenizer.char_to_ix(inp)).reshape(1, 1).to(device)
            else:
                x = inp.to(device)

            output, init_states = self.forward(x, init_states)
            output_conf = torch.softmax(output, dim=-1)  # [S, B, Din]
            output_topk = torch.topk(output_conf, top_k, dim=-1)  # [S, B, K]

            prob = output_topk[0].reshape(-1) / output_topk[0].reshape(-1).sum()
            k_star = np.random.choice(np.arange(top_k), p=prob.cpu().numpy())
            output_ids = output_topk[1][:, :, k_star]

            if isinstance(inp, str):
                output_char = self.tokenizer.ix_to_char(output_ids.item())
                return output_char, init_states
            else:
                return output_ids, init_states

    def forward(self, inp: torch.LongTensor, init_states: torch.Tensor = None):

        if self.batch_first:
            inp = inp.transpose(0, 1)

        if self.embedding_dim is not None:
            if len(inp.shape) != 2:
                raise ValueError("Expected input tensor of order 2, but got order {} tensor instead".format(len(inp.shape)))
            x = self.embedding(inp)  # [S, B, D_in] (i.e. [sequence, batch, input_size])
        else:
            x = inp
        sequence_length, batch_size, _ = x.size()
        hidden_seq = []

        device = x.device

        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        else:
            h_t = init_states
            h_t = h_t.to(device)

        for t in range(sequence_length):
            x_t = x[t, :, :]

            A_prime = h_t @ self.A
            B_prime = x_t @ self.B

            h_t = self.gate(
                torch.einsum("br,br,hr -> bh", A_prime, B_prime, self.C) + h_t @ self.V + x_t @ self.U + self.d
            )

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        output = self.decoder(hidden_seq.contiguous())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output


class CPRNN_cell(nn.Module):
    """CP-Factorized LSTM, single cell.

    Args:
        input_size: Input size
        hidden_size: Dimension of hidden features.
        rank: Rank of cp factorization
        tokenizer: Character tokenizer
        batch_first: Whether to use batch first or not
        dropout: Dropout rate
        gate: Gate function (activation from t to t+1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rank: int,
        tokenizer, 
        batch_first: bool,
        dropout: float,
        gate: callable,
        **kwargs
    ):
        super().__init__()

        self.dropout = dropout
        self.batch_first = batch_first
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rank = rank
        self.gate = gate

        # Encoder using CP factors
        self.A = nn.Parameter(torch.Tensor(self.hidden_size, self.rank))
        self.B = nn.Parameter(torch.Tensor(self.input_size, self.rank))
        self.C = nn.Parameter(torch.Tensor(self.hidden_size, self.rank))
        self.U = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.V = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.d = nn.Parameter(torch.Tensor(self.hidden_size))
        self.tokenizer = tokenizer
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def predict(
        self,
        inp: Union[torch.LongTensor, str],
        init_states: tuple = None,
        top_k: int = 1,
        device=torch.device("cpu"),
    ):

        with torch.no_grad():

            if isinstance(inp, str):
                if self.tokenizer is None:
                    raise ValueError(
                        "Tokenizer not defined. Please provide a tokenizer to the model."
                    )
                x = (
                    torch.tensor(self.tokenizer.char_to_ix(inp))
                    .reshape(1, 1)
                    .to(device)
                )
            else:
                x = inp.to(device)

            output, init_states = self.forward(x, init_states)
            output_conf = torch.softmax(output, dim=-1)  # [S, B, Din]
            output_topk = torch.topk(output_conf, top_k, dim=-1)  # [S, B, K]

            prob = output_topk[0].reshape(-1) / output_topk[0].reshape(-1).sum()
            k_star = np.random.choice(np.arange(top_k), p=prob.cpu().numpy())
            output_ids = output_topk[1][:, :, k_star]

            if isinstance(inp, str):
                output_char = self.tokenizer.ix_to_char(output_ids.item())
                return output_char, init_states
            else:
                return output_ids, init_states

    def forward(self, x, h):
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
            # h = torch.ones(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        A_prime = h @ self.A
        B_prime = x @ self.B

        h_next = self.gate(
            torch.einsum("br,br,hr -> bh", A_prime, B_prime, self.C)
            + h @ self.V
            + x @ self.U
            + self.d
        )
        return h_next


class DeepCPRNN(nn.Module):
    """CP-Factorized LSTM. Outputs logits (no softmax)

    Args:
        hidden_size: Dimension of hidden features.
        vocab_size: Size of vocabulary
        num_layers: Number of layers
        embedding_dim: Dimension of the embedding (`None` means no embedding)
        rank: Rank of cp factorization
        tokenizer: Character tokenizer
        batch_first: Whether to use batch first or not
        dropout: Dropout rate
        activation: Activation function (activation from l to l+1)
        readout_activation: Readout activation function
        gate: Gate function (activation from t to t+1)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        embedding_dim: Optional[int] = None,
        rank: int = 8,
        tokenizer: CharacterTokenizer = None,
        batch_first: bool = True,
        dropout: float = 0.0,
        dropout_between_layers: bool = False,
        activation="identity",
        readout_activation="identity",
        gate: str = "identity",
        **kwargs
    ):
        super().__init__()

        self.dropout = dropout
        self.dropout_between_layers = dropout_between_layers
        self.batch_first = batch_first
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # here specifically
        self.output_size = self.vocab_size

        if activation == "relu":
            self.activation_fn = F.relu
        elif activation == "tanh":
            self.activation_fn = torch.tanh
        elif activation == "identity":
            self.activation_fn = lambda x: x
        else:
            raise ValueError("activation must be 'relu', 'tanh', or 'identity'")

        if readout_activation == "relu":
            self.readout_activation_fn = F.relu
        elif readout_activation == "tanh":
            self.readout_activation_fn = torch.tanh
        elif readout_activation == "identity":
            self.readout_activation_fn = lambda x: x
        else:
            raise ValueError("readout_activation must be 'relu', 'tanh', or 'identity'")

        self.rank = rank
        self.gate = {
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "identity": lambda x: x,
            "relu": F.relu,
        }[gate]

        # Define embedding and decoder layers
        if embedding_dim != None:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
            self.input_size = self.embedding_dim
        else:
            self.input_size = self.vocab_size


        self.init_weights()

        self.cprnn_cell = CPRNN_cell
        self.cprnn_layers = nn.ModuleList(
            [
                self.cprnn_cell(
                    self.input_size if i == 0 else self.hidden_size,
                    self.hidden_size,
                    self.rank,
                    self.tokenizer,
                    self.batch_first,
                    self.dropout,
                    self.gate,
                )
                for i in range(self.num_layers)
            ]
        )

        self.decoder = nn.Linear(self.hidden_size, self.output_size)

    # what is this used for? not clear
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def predict(
        self,
        inp: Union[torch.LongTensor, str],
        init_states: tuple = None,
        top_k: int = 1,
        device=torch.device("cpu"),
    ):

        with torch.no_grad():

            if isinstance(inp, str):
                if self.tokenizer is None:
                    raise ValueError(
                        "Tokenizer not defined. Please provide a tokenizer to the model."
                    )
                x = (
                    torch.tensor(self.tokenizer.char_to_ix(inp))
                    .reshape(1, 1)
                    .to(device)
                )
            else:
                x = inp.to(device)

            output, init_states = self.forward(x, init_states)
            output_conf = torch.softmax(output, dim=-1)  # [S, B, Din]
            output_topk = torch.topk(output_conf, top_k, dim=-1)  # [S, B, K]

            prob = output_topk[0].reshape(-1) / output_topk[0].reshape(-1).sum()
            k_star = np.random.choice(np.arange(top_k), p=prob.cpu().numpy())
            output_ids = output_topk[1][:, :, k_star]

            if isinstance(inp, str):
                output_char = self.tokenizer.ix_to_char(output_ids.item())
                return output_char, init_states
            else:
                return output_ids, init_states


    def forward(self, x: torch.LongTensor):
        if self.batch_first:
            x = x.transpose(0, 1)

        if self.embedding_dim is not None:
            if len(x.shape) != 2:
                raise ValueError(
                    "Expected input tensor of order 2, but got order {} tensor instead".format(
                        len(x.shape)
                    )
                )
            x = self.embedding(x)  # [S, B, D_in] (i.e. [sequence, batch, input_size])
        seq_length, batch_size, _ = x.size()
        device = x.device
        # not input possible for init states for now, like DeepRNN
        h = self.init_hidden(batch_size, device)
        outputs = []

        for t in range(seq_length):
            out, h = self.forward_one_timestep(x[t], h)
            # Below: out.unsqueeze(0) -> out 
            # unsqueeze(0) was messing the computation 
            outputs.append(out)


        outputs = torch.stack(outputs,dim=0)
        # I don't know what this does but I'm keeping it
        outputs = outputs.contiguous()
        # This reproduces CPRNN behaviour: Dropout after last layer, after stacking
        # just before decoding
        if not self.dropout_between_layers:
            outputs = nn.Dropout(self.dropout)(outputs)
        outputs = self.decoder(outputs)

        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        
        return outputs

    def forward_one_timestep(self, x, h):
        h_depth = []
        h_rec = []
        for i, cprnn in enumerate(self.cprnn_layers):
            h_i = cprnn(x if i == 0 else h_depth[i - 1], h[i])
            h_rec.append(h_i)
            h_i = self.activation_fn(h_i)
            # This reproduces S4 behaviour: Dropout between layers, 
            # after activation
            if self.dropout_between_layers: 
                h_i = nn.Dropout(self.dropout)(h_i)
            h_depth.append(h_i)
        out = h_depth[-1]
        out = self.readout_activation_fn(out)

        return out, torch.stack(h_rec)
    
    def init_hidden(self, batch_size, device=torch.device("cpu")):
        return [
            torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers)
        ]

def test_cprnn():
    vocab_size = 100
    input_size = 50
    hidden_size = 128
    rank = 8
    batch_size = 32
    seq_len = 10

    model = CPRNN(input_size, hidden_size, vocab_size, rank=rank)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(x)
    print(output.shape)  # Should be [batch_size, seq_len, vocab_size]

def test_seq2seq_rnn(vocab_size=100, embedding_dim=64, hidden_size=128):
    vocab_size = 100
    embedding_dim = 64
    hidden_size = 128
    output_size = vocab_size
    model = SimpleSeq2SeqRNN(vocab_size, embedding_dim, hidden_size, output_size)

    # Example input: a sequence of token indices
    seq_len = 10
    batch_size = 32
    input_tensor = torch.randint(0, vocab_size, (seq_len, batch_size))  # Random token indices
    print(input_tensor.shape)  # Should be (seq_len, batch_size)
    output, hidden = model(input_tensor)
    print("Output shape:", output.shape)  # Should be (seq_len, batch_size, output_size)

def test_deep_rnn(input_size=100, hidden_size=20, output_size=5, num_layers=2):
    model = DeepRNN(input_size, hidden_size, output_size, num_layers, rnncell='linear')
    
    # Example input: a sequence of vectors
    batch_size = 32 
    seq_len = 10 
    x = torch.randn(batch_size, seq_len, input_size)
    print("Input shape:", x.shape)  # Should be (batch_size, seq_len, input_size)

    outputs = model(x)
    
    print("Output shape:", outputs.shape)  # Should be (batch_size, output_size)

def test_deep_rnn_with_embedding(vocab_size=100, embedding_dim=64, hidden_size=128):
    model = DeepRNNWithEmbedding(vocab_size, embedding_dim, hidden_size, vocab_size, num_layers=2)

    # Example input: a sequence of token indices
    seq_len = 10
    batch_size = 32
    input_tensor = torch.randint(0, vocab_size, (seq_len, batch_size))  # Random token indices
    print(input_tensor.shape)  # Should be (seq_len, batch_size)

    outputs = model(input_tensor)
    print("Output shape:", outputs.shape)  # Should be (seq_len, batch_size, output_size)

def test_s4():
    s4_model = S4Model(d_input=100, d_output=5, d_model=128, n_layers=4, dropout=0.2)
    batch_size = 32 
    seq_len =10 
    x = torch.randn(batch_size, seq_len, 100)
    print("Input shape:", x.shape)  # Should be (batch_size, seq_len, d_input)
    outs = s4_model(x)
    print("S4 Model Output shape:", outs.shape)  # Should be (batch_size, seq_len, d_output)


if __name__ == "__main__":
    print("Testing SimpleSeq2SeqRNN...")
    test_seq2seq_rnn()
    print("Testing DeepRNN...")
    test_deep_rnn()
    print("Testing DeepRNNWithEmbedding...")
    test_deep_rnn_with_embedding()
    print("Testing S4Model...")
    test_s4()
    print("Testing CPRNN...")
    test_cprnn()