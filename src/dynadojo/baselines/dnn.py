"""
Deep Neural Network (DNN)
===========================
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..abstractions import AbstractAlgorithm


class DNN(AbstractAlgorithm):
    """Deep Neural Network (DNN). Contains 5 hidden layers with 30 neurons each."""

    def __init__(
            self,
            embed_dim,
            timesteps,
            max_control_cost=0,
            activation='relu',
            seed=None,
            **kwargs):
        """
        Initialize the class.

        Parameters
        -------------
        embed_dim : int
            The embedded dimension of the system. Recommended to keep embed dimension small (e.g., <10).
        timesteps : int
            The timesteps of the training trajectories. Must be greater than 2.
        activation : str, optional
            The activation function used in the hidden layers. See ``tensorflow`` documentation for more details on
            acceptable activations. Defaults to ``relu``.
        max_control_cost : float, optional
            Ignores control, so defaults to 0.
        **kwargs : dict, optional
            Additional keyword arguments
        """
        super().__init__(embed_dim, timesteps, max_control_cost, seed=seed, **kwargs)
        if seed:
            keras.utils.set_random_seed(812)
            # tf.config.experimental.enable_op_determinism()
        kreg = "l2"
        self.model = tf.keras.Sequential([
            keras.Input(shape=(None, embed_dim)),
            keras.layers.Dense(30, activation=activation, kernel_regularizer=kreg),
            keras.layers.Dense(30, activation=activation, kernel_regularizer=kreg),
            keras.layers.Dense(30, activation=activation, kernel_regularizer=kreg),
            keras.layers.Dense(30, activation=activation, kernel_regularizer=kreg),
            keras.layers.Dense(30, activation=activation, kernel_regularizer=kreg),
            keras.layers.Dense(embed_dim, kernel_regularizer=kreg)
        ])
        self.model.compile(optimizer="adam", loss="mean_squared_error")

    def fit(self, x: np.ndarray, epochs=2000, verbose=0, **kwargs):
        head = x[:, :-1, :]
        tail = x[:, 1:, :]
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.model.fit(head, tail, validation_split=0.2, epochs=epochs, callbacks=[callback], verbose=verbose)

    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            trajectory = x0.reshape(x0.shape[0], 1, x0.shape[1])
            next_input = torch.tensor(trajectory, dtype=torch.float32).to(self.device)
            
            # Iteratively predict future values
            for _ in range(timesteps - 1):
                next_input = self.forward(next_input)
                to_add = next_input.cpu().numpy().reshape(next_input.shape[0], 1, next_input.shape[-1])
                trajectory = np.concatenate([trajectory, to_add], axis=1)
        return trajectory          

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, start_from_epoch=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_weights = None
        self.start_from_epoch = start_from_epoch

    def early_stop(self, epoch, validation_loss, weights):
        if epoch < self.start_from_epoch:
            return False
        
        if weights is None:
            self.best_weights = weights
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.best_weights = weights
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class DNN(TorchBaseClass):
    def __init__(
            self,
            embed_dim,
            timesteps,
            **kwargs):
        super().__init__(embed_dim, timesteps, **kwargs)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim, embed_dim*10),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim*10, embed_dim*10),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim*10, embed_dim*5),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim*5, embed_dim*10),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim*10, embed_dim*10),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim*10, self.embed_dim),
        )
        self.model.to(self.device)
    

    def forward(self, x):
        # The model will flatten the input so each timestep is a separate sample we will unflatten it back
        # Note x.shape = (batch_size, timesteps, embed_dim)
        timesteps = x.shape[0]
        x_flat = x.reshape(-1, x.shape[-1])
        return self.model(x_flat).unflatten(0, (timesteps, -1))

# class CNN(TorchBaseClass):
#     def __init__(self, 
#             embed_dim,
#             timesteps,
#             **kwargs):
#         super().__init__(embed_dim, timesteps, **kwargs)
#         self.CNN = torch.nn.Sequential(
#             torch.nn.Conv1d(embed_dim, embed_dim,  kernel_size=7, padding=3),
#             torch.nn.ReLU(),
#             torch.nn.Conv1d(embed_dim, embed_dim,  kernel_size=5, padding=2),
#             torch.nn.ReLU(),
#         )
#         self.linear = torch.nn.Sequential(
#             torch.nn.Linear(embed_dim, 50),
#             torch.nn.ReLU(),
#             torch.nn.Linear(50, 50),
#             torch.nn.ReLU(),
#             torch.nn.Linear(50, 50),
#             torch.nn.ReLU(),
#             torch.nn.Linear(50, embed_dim),
#         )
#         self.CNN.to(self.device)
#         self.linear.to(self.device)
    

#     def forward(self, x):
#         # The model will flatten the input so each timestep is a separate sample we will unflatten it back
#         # x.shape = (batch_size, timesteps, embed_dim)
#         return self.linear(self.CNN(x.permute(0, 2, 1)).permute(0, 2, 1))
    
# class PositionalEncoding(torch.nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = torch.nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, d_model)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         if d_model%2 != 0:
#             pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
#         else:
#             pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Arguments:
#             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         x = x + self.pe[:x.size(1)]
#         return self.dropout(x)

# class Transformer(TorchBaseClass):
#     def __init__(self, 
#             embed_dim,
#             timesteps,
#             # Decoder layer hyperparameters
#             model_dim=512, num_heads=16, dim_feedforward=2048, dropout=0.1, activation='relu', 
#             # Decoder Transformer hyperparameters
#             num_layers=6, norm=None, 
#             **kwargs):
        
#         super().__init__(embed_dim, timesteps, **kwargs)
#         self.model_dim = model_dim

#         self.embedding = torch.nn.Linear(embed_dim, model_dim).to(self.device)
#         self.max_timesteps = 5000
#         self.positional_encoder = PositionalEncoding(model_dim, max_len=self.max_timesteps).to(self.device)      
#         # decoder_layer = torch.nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, 
#         #                                            activation=activation, batch_first=True)
#         # self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers).to(self.device)
#         # self.fc_out = torch.nn.Linear(model_dim, embed_dim).to(self.device)
#         self.transformer = torch.nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, 
#                                                 dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, batch_first=True).to(self.device)

        
    

#     def forward(self, x):
#         assert x.shape[1] <= self.max_timesteps, f"Sequence length {x.shape[1]} exceeds maximum sequence length {self.max_timesteps}"
        
#         batch_size, seq_len, input_dim = x.size()
#         # Embed input to model_dim
#         x = self.embedding(x)
#         # Scale the embeddings
#         x *= torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
#         # Add positional encoding
#         x = self.positional_encoder(x)
        
#         # Generate mask for the decoder to prevent attending to future positions
#         tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len)
#         tgt_mask.to(x.device)
#         output = self.transformer(x, x, src_mask=tgt_mask) 
#         # output = self.fc_out(output)
#         return output
    
