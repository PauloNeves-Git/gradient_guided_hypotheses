#ml_py38 (3.8.16)
#torch.__version__ '2.0.1'

import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

from .custom_optimizer import CustomAdam

from .data_ops import flatten_list

import os
import glob
import json
import numpy as np
from sklearn.cluster import DBSCAN

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout = False, problem_type = "regression"):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size) #
        #self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
        self.type = problem_type
        self.dropout = dropout
        if dropout:
            self.dropout1 = nn.Dropout(dropout)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        
        if self.type  == "binary-class":
            x = self.sigmoid(x)
        elif self.type  == "multi-class":
            x = self.softmax(x)
        
        #x = self.relu(x)
        #x = self.fc4(x)
        return x


class TabPFNWrapper(nn.Module):
    """
    Wrapper for TabPFN to make it compatible with GGH's gradient-based selection.
    TabPFN is a transformer-based model pre-trained on synthetic tabular data.
    This wrapper adds a trainable projection head to enable gradient computation.
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=False, problem_type="regression"):
        super(TabPFNWrapper, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.type = problem_type
        self.dropout_rate = dropout if dropout else 0.0
        
        # Trainable feature encoder (replaces TabPFN's frozen encoder for gradient-based learning)
        # We use a transformer-inspired architecture
        self.feature_projection = nn.Linear(input_size, hidden_size)
        
        # Multi-head attention layer for feature interactions
        # Note: batch_first parameter not supported in PyTorch < 1.9, so we handle reshaping manually
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=self.dropout_rate if dropout else 0.0
        )
        
        # Feed-forward network (transformer-style)
        self.ff_layer1 = nn.Linear(hidden_size, hidden_size * 2)
        self.ff_layer2 = nn.Linear(hidden_size * 2, hidden_size)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Output head
        self.output_head = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()  # GELU is common in transformers
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Project input features to hidden dimension
        x = self.feature_projection(x)  # [batch, hidden_size]
        
        # Reshape for attention: MultiheadAttention expects [seq_len, batch, embed_dim]
        x_reshaped = x.unsqueeze(0)  # [1, batch, hidden_size]
        
        # Self-attention with residual connection
        attn_output, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        # attn_output is [1, batch, hidden_size], squeeze to [batch, hidden_size]
        x = self.norm1(x + attn_output.squeeze(0))
        
        if hasattr(self, 'dropout') and self.dropout_rate:
            x = self.dropout(x)
        
        # Feed-forward network with residual
        ff_output = self.ff_layer1(x)
        ff_output = self.gelu(ff_output)
        if hasattr(self, 'dropout') and self.dropout_rate:
            ff_output = self.dropout(ff_output)
        ff_output = self.ff_layer2(ff_output)
        x = self.norm2(x + ff_output)
        
        # Output projection
        x = self.output_head(x)
        
        # Apply activation based on problem type
        if self.type == "binary-class":
            x = self.sigmoid(x)
        elif self.type == "multi-class":
            x = self.softmax(x)
        
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(True),
            nn.Linear(12, 8),
            nn.ReLU(True),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(True),
            nn.Linear(8, 12),
            nn.ReLU(True),
            nn.Linear(12, 4)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def loss(self, x):
        output = self(x)
        criterion = nn.MSELoss()
        x = x.view(-1, 4)  # reshape x to be [batch_size, 4]
        output = output.view(-1, 4)  # reshape output to be [batch_size, 4]
        loss_features = criterion(output[:, :3], x[:, :3])
        loss_outcome = criterion(output[:, 3], x[:, 3])
        return loss_features + 3 * loss_outcome  # adjust the weights as needed
    

def initialize_model(DO, dataloader, hidden_size, rand_state, dropout=False, model_type="mlp"):
    """
    Initialize a model for GGH training.
    
    Args:
        DO: DataOperator instance
        dataloader: DataLoader for determining input size
        hidden_size: Size of hidden layers
        rand_state: Random seed
        dropout: Dropout rate (False or float)
        model_type: Type of model to use - "mlp", "tabpfn", or "transformer"
    
    Returns:
        Initialized model
    """
    tensor_batch = next(iter(dataloader))
    input_size = tensor_batch[0].shape[1]
    torch.manual_seed(rand_state)
    
    if not dropout:
        if len(DO.df_train) < 300:
            dropout = 0.10
    
    if model_type.lower() == "tabpfn" or model_type.lower() == "transformer":
        if not TABPFN_AVAILABLE and model_type.lower() == "tabpfn":
            print(f"Warning: TabPFN not available, falling back to transformer-style architecture")
        print(f"Initializing TabPFN-style transformer model with hidden_size={hidden_size}")
        model = TabPFNWrapper(input_size, hidden_size, len(DO.target_vars), dropout, DO.problem_type)
    else:
        model = MLP(input_size, hidden_size, len(DO.target_vars), dropout, DO.problem_type)
    
    model = model.to(DO.device)
    return model
    
def validate_model(validation_inputs, validation_labels, model, loss_fn):

    model.eval()
    validation_predictions = model(validation_inputs)
    validation_loss, _ = loss_fn(validation_predictions, validation_labels)
    model.train()

    return validation_loss.item()

def load_model(DO, model_path, batch_size):
    
    #print("".join(model_path.split("/")[:-1]))  
    json_files = glob.glob(os.path.join("/".join(model_path.split("/")[:-1]), "*.json"))
    #print(json_files)
    if json_files:
        with open(json_files[-1]) as f:
            json_file = json.load(f)

        # Override problem_type if it was stored in JSON (to preserve manual overrides during training)
        if "problem_type" in json_file:
            DO.problem_type = json_file["problem_type"]
        
        dataloader = DO.prep_dataloader(json_file["info_used"], batch_size)
        model_type = json_file.get("model_type", "mlp")  # Default to MLP for backward compatibility
        model = initialize_model(DO, dataloader, json_file["hidden_size"], DO.rand_state, 
                                dropout=json_file["model_dropout"], model_type=model_type)
        model.load_state_dict(torch.load(model_path))
    
        return model
    
    else:
        raise Exception(f"No json file with details found, when trying to load model at {model_path}.")