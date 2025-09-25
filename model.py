import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class TrainingConfig:
    def __init__(self):
        self.epochs = 5
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.patience = 10  # For early stopping
        self.grad_clip = 1.0

class ModelConfig:
    def __init__(self, context_len=50, target_len=20):
        self.input_dim = 5
        self.hidden_dim = 128  # Reduced for faster training
        self.lstm_hidden_dim = 64
        self.lstm_layers = 2
        self.transformer_layers = 2  # Reduced for faster training
        self.nhead = 4
        self.fusion_dim = 128
        self.output_dim = 128
        self.time_dim = 32  # Reduced for non-diffusion
        self.dropout = 0.1
        self.context_len = context_len
        self.target_len = target_len

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        # self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = "cpu"
        
        self.model.to(self.device)
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, batch in enumerate(progress_bar):
            context = batch['context'].to(self.device)
            target = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass - predict target from context
            outputs = self.model(context, target)
            predicted_target = outputs['predicted_target']
            
            loss = self.criterion(predicted_target, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                context = batch['context'].to(self.device)
                target = batch['target'].to(self.device)
                
                outputs = self.model(context, target, teacher_forcing_ratio=0.0)  # No teacher forcing
                predicted_target = outputs['predicted_target']
                
                loss = self.criterion(predicted_target, target)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        print(f"Starting training on {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}, Val samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{self.config.epochs}:')
            print(f'  Train Loss: {train_loss:.6f}')
            print(f'  Val Loss:   {val_loss:.6f}')
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                print('  ↳ New best model saved!')
            else:
                self.patience_counter += 1
                print(f'  ↳ Early stopping counter: {self.patience_counter}/{self.config.patience}')
            
            if self.patience_counter >= self.config.patience:
                print('Early stopping triggered!')
                break
            
            print('-' * 50)
    
    def plot_training(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curve.png')
        plt.show()

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim,hidden_dim,num_layers = 2,dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True ,
            bidirectional=True,
            dropout = dropout if num_layers > 1 else 0.0
            )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]  (vital signs)
        """
        outputs, (hidden,cell) = self.lstm(x)  # output: [batch, seq_len, hidden_dim*2]
        outputs = self.dropout(outputs)
        last_hidden = hidden[-1] # [batch, hidden_dim]
        return outputs, last_hidden
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_model, max_len = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x= x + self.pe[:x.size(0), :]
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model,nhead,num_layers,dim_feedforward=2048,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead = nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x , src_mask=None, src_key_padding_mask=None):
        """
        x: [batch, seq_len, d_model]
        transformer expects input of shape [seq_len, batch, d_model]
        """
        x = x.transpose(0,1)  # [seq_len, batch, d_model]
        x= self.pos_encoder(x)
        x = x.transpose(0,1)  # [batch, seq_len, d_model]
        encoded = self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return encoded

class ContextFusion(nn.Module):
    def __init__(self, lstm_hidden_dim, transformer_hidden_dim, fusion_dim):
        super().__init__()
        self.lstm_proj = nn.Linear(lstm_hidden_dim, fusion_dim)
        self.transformer_proj = nn.Linear(transformer_hidden_dim, fusion_dim)
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        self.fusion_layer = nn.Linear(fusion_dim * 2, fusion_dim)
        
    def forward(self, lstm_output, transformer_output):
        # lstm_output: [batch_size, lstm_hidden_dim] (summary vector)
        # transformer_output: [batch_size, seq_len, transformer_hidden_dim]
        
        # Project both to same dimension
        lstm_proj = self.lstm_proj(lstm_output)  # [batch_size, fusion_dim]
        transformer_summary = transformer_output.mean(dim=1)  # [batch_size, transformer_hidden_dim]
        transformer_proj = self.transformer_proj(transformer_summary)  # [batch_size, fusion_dim]
        
        # Gated fusion
        combined = torch.cat([lstm_proj, transformer_proj], dim=-1)
        gate_weights = self.gate(combined)
        
        # Apply gating
        gated_lstm = lstm_proj * gate_weights
        gated_transformer = transformer_proj * (1 - gate_weights)
        
        # Final fusion
        fused = torch.cat([gated_lstm, gated_transformer], dim=-1)
        fused = self.fusion_layer(fused)
        
        return fused
    
class LSTMTransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection if needed
        if config.input_dim != config.hidden_dim:
            self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        else:
            self.input_proj = nn.Identity()
        
        # LSTM Encoder
        self.lstm_encoder = LSTMEncoder(
            input_dim=config.hidden_dim,
            hidden_dim=config.lstm_hidden_dim,
            num_layers=config.lstm_layers,
            dropout=config.dropout
        )
        
        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            d_model=config.hidden_dim,
            nhead=config.nhead,
            num_layers=config.transformer_layers,
            dropout=config.dropout
        )
        
        # Fusion mechanism
        self.fusion = ContextFusion(
            lstm_hidden_dim=config.lstm_hidden_dim,
            transformer_hidden_dim=config.hidden_dim,
            fusion_dim=config.fusion_dim
        )
        
        # Projection layers for sequence representation - FIXED: define as proper layers
        if config.lstm_hidden_dim != config.hidden_dim:
            self.lstm_proj_layer = nn.Linear(config.lstm_hidden_dim*2, config.hidden_dim)
        else:
            self.lstm_proj_layer = nn.Identity()
            
        # Project concatenated sequence to hidden_dim if needed
        self.sequence_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # Input projection to hidden_dim
        x_proj = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]
        
        # Process through LSTM
        lstm_sequence, lstm_summary = self.lstm_encoder(x_proj)
        # lstm_sequence: [batch_size, seq_len, lstm_hidden_dim]
        # lstm_summary: [batch_size, lstm_hidden_dim]
        
        # Process through Transformer
        transformer_output = self.transformer_encoder(
            x_proj, 
            src_key_padding_mask=mask
        )  # [batch_size, seq_len, hidden_dim]
        
        # Fuse LSTM and Transformer representations
        context_embedding = self.fusion(lstm_summary, transformer_output)
        # context_embedding: [batch_size, fusion_dim]
        
        # Create sequence representation for decoder memory
        # Project LSTM output to hidden_dim if needed
        lstm_proj = self.lstm_proj_layer(lstm_sequence)  # [batch_size, seq_len, hidden_dim]
        
        # Concatenate and project to hidden_dim
        sequence_representation = torch.cat([lstm_proj, transformer_output], dim=-1)
        sequence_representation = self.sequence_proj(sequence_representation)  # [batch_size, seq_len, hidden_dim]
        
        return {
            'context_embedding': context_embedding,
            'sequence_representation': sequence_representation,
            'lstm_features': lstm_sequence,
            'transformer_features': transformer_output
        }
    
class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def forward(self, x, hidden_state=None, cell_state=None):
        # x: [batch_size, target_len, input_dim]
        # hidden_state, cell_state: [num_layers, batch_size, hidden_dim]
        
        if hidden_state is None or cell_state is None:
            # Initialize with zeros if no initial state provided
            batch_size = x.size(0)
            hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        outputs, (hidden, cell) = self.lstm(x, (hidden_state, cell_state))
        # outputs: [batch_size, target_len, hidden_dim]
        # hidden, cell: [num_layers, batch_size, hidden_dim]
        
        return outputs, hidden, cell
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Self-attention for target sequence
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Cross-attention to encoder context
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # tgt: [batch_size, target_len, d_model] - target sequence
        # memory: [batch_size, context_len, d_model] - encoder output
        
        # Self-attention on target sequence
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, 
                             key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention: target queries, context keys/values
        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask,
                              key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # tgt: [batch_size, target_len, d_model]
        # memory: [batch_size, context_len, d_model]
        
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask,
                          tgt_key_padding_mask, memory_key_padding_mask)
        
        return output
    
class LSTMTransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection for target sequence
        self.target_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Time step embedding (simplified for non-diffusion)
        self.time_embedding = nn.Sequential(
            nn.Linear(config.time_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # LSTM Decoder
        self.lstm_decoder = LSTMDecoder(
            input_dim=config.hidden_dim,
            hidden_dim=config.lstm_hidden_dim,
            num_layers=config.lstm_layers,
            dropout=config.dropout
        )
        
        # Project LSTM output to transformer dimension
        self.lstm_to_transformer = nn.Linear(config.lstm_hidden_dim, config.hidden_dim)
        
        # Project context embedding to LSTM hidden dimension for initialization
        self.context_to_hidden = nn.Linear(config.fusion_dim, config.lstm_hidden_dim * config.lstm_layers)
        
        # Transformer Decoder
        self.transformer_decoder = TransformerDecoder(
            d_model=config.hidden_dim,
            nhead=config.nhead,
            num_layers=config.transformer_layers,
            dropout=config.dropout
        )
        
        # Output projection to target dimension
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.input_dim)
        )
        
        # Projection for context sequence if dimensions don't match
        if config.fusion_dim != config.hidden_dim:
            self.context_proj = nn.Linear(config.fusion_dim, config.hidden_dim)
        else:
            self.context_proj = nn.Identity()
        
    def forward(self, target_seq, context_embedding, context_sequence, time_embed, 
                target_mask=None, context_mask=None):
        """
        Args:
            target_seq: [batch_size, target_len, input_dim]
            context_embedding: [batch_size, fusion_dim]
            context_sequence: [batch_size, context_len, hidden_dim]
            time_embed: [batch_size, time_dim]
        """
        batch_size, target_len, _ = target_seq.shape
        
        # 1. Process target sequence
        target_proj = self.target_proj(target_seq)  # [batch_size, target_len, hidden_dim]
        
        # 2. Add time embedding (simplified for non-diffusion)
        time_embed_expanded = self.time_embedding(time_embed).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        time_embed_expanded = time_embed_expanded.expand(-1, target_len, -1)
        target_with_time = target_proj + time_embed_expanded  # [batch_size, target_len, hidden_dim]
        
        # 3. Initialize LSTM hidden state from context embedding
        hidden_init = self.context_to_hidden(context_embedding)  # [batch_size, lstm_hidden_dim * lstm_layers]
        hidden_init = hidden_init.view(batch_size, self.config.lstm_layers, self.config.lstm_hidden_dim)
        hidden_init = hidden_init.transpose(0, 1)  # [lstm_layers, batch_size, lstm_hidden_dim]
        cell_init = torch.zeros_like(hidden_init)
        
        # 4. Process through LSTM Decoder
        lstm_output, hidden, cell = self.lstm_decoder(
            target_with_time, hidden_init, cell_init
        )  # lstm_output: [batch_size, target_len, lstm_hidden_dim]
        
        # 5. Project LSTM output to transformer dimension
        lstm_transformed = self.lstm_to_transformer(lstm_output)  # [batch_size, target_len, hidden_dim]
        
        # 6. Prepare memory for transformer decoder
        if context_sequence is not None:
            memory = context_sequence  # [batch_size, context_len, hidden_dim]
        else:
            # Fallback: use context_embedding expanded
            memory = self.context_proj(context_embedding).unsqueeze(1)  # [batch_size, 1, hidden_dim]
            memory = memory.expand(-1, self.config.context_len, -1)
        
        # 7. Process through Transformer Decoder
        transformer_output = self.transformer_decoder(
            tgt=lstm_transformed,
            memory=memory,
            tgt_key_padding_mask=target_mask,
            memory_key_padding_mask=context_mask
        )  # [batch_size, target_len, hidden_dim]
        
        # 8. Final output projection to target dimension
        output = self.output_proj(transformer_output)  # [batch_size, target_len, input_dim]
        
        return {
            'predicted_noise': output,
            'lstm_output': lstm_output,
            'transformer_output': transformer_output,
            'final_hidden': hidden,
            'final_cell': cell
        }
    
class LSTMTransformerDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder for context sequence
        self.encoder = LSTMTransformerEncoder(config)
        
        # Decoder for target sequence
        self.decoder = LSTMTransformerDecoder(config)
        
    def forward(self, context_seq, target_seq, timestep, context_mask=None, target_mask=None):
        """
        Full forward pass for the first stage model
        """
        batch_size = context_seq.size(0)
        
        # Create timestep embedding
        time_embed = self.get_timestep_embedding(timestep, batch_size)
        
        # Encode context
        encoder_outputs = self.encoder(context_seq, mask=context_mask)
        context_embedding = encoder_outputs['context_embedding']
        context_sequence = encoder_outputs['sequence_representation']
        
        # Decode target with context conditioning
        decoder_outputs = self.decoder(
            target_seq=target_seq,
            context_embedding=context_embedding,
            context_sequence=context_sequence,
            time_embed=time_embed,
            target_mask=target_mask,
            context_mask=context_mask
        )
        
        return {
            'predicted_noise': decoder_outputs['predicted_noise'],
            'context_embedding': context_embedding,
            'encoder_outputs': encoder_outputs,
            'decoder_outputs': decoder_outputs
        }
    
    def get_timestep_embedding(self, timesteps, batch_size):
        """
        Create sinusoidal timestep embeddings
        """
        if isinstance(timesteps, int):
            timesteps = torch.tensor([timesteps] * batch_size, device=next(self.parameters()).device)
        
        half_dim = self.config.time_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if self.config.time_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        
        return emb
    

class SimpleLSTMTransformerSeq2Seq(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder for context sequence
        self.encoder = LSTMTransformerEncoder(config)
        
        # Decoder for target sequence
        self.decoder = LSTMTransformerDecoder(config)
        
    def forward(self, context_seq, target_seq=None, teacher_forcing_ratio=0.5):
        """
        Simple seq2seq forward pass without diffusion
        """
        batch_size = context_seq.size(0)
        
        # Encode context
        encoder_outputs = self.encoder(context_seq)
        context_embedding = encoder_outputs['context_embedding']  # [batch_size, fusion_dim]
        context_sequence = encoder_outputs['sequence_representation']  # [batch_size, context_len, hidden_dim]
        
        # Create dummy time embedding for non-diffusion training
        time_embed = torch.zeros(batch_size, self.config.time_dim).to(context_seq.device)
        
        # If no target_seq provided (inference), create zero-initialized target
        if target_seq is None:
            target_seq = torch.zeros(batch_size, self.config.target_len, self.config.input_dim).to(context_seq.device)
        
        # Use teacher forcing during training
        use_teacher_forcing = self.training and (torch.rand(1).item() < teacher_forcing_ratio)
        
        if use_teacher_forcing:
            # Use actual target sequence as input to decoder
            decoder_outputs = self.decoder(
                target_seq=target_seq,
                context_embedding=context_embedding,
                context_sequence=context_sequence,
                time_embed=time_embed
            )
        else:
            # For simplicity in initial validation, still use zeros
            # Later we can implement proper autoregressive generation
            dummy_target = torch.zeros(batch_size, self.config.target_len, self.config.input_dim).to(context_seq.device)
            decoder_outputs = self.decoder(
                target_seq=dummy_target,
                context_embedding=context_embedding,
                context_sequence=context_sequence,
                time_embed=time_embed
            )
        
        # The decoder already outputs the right dimension through output_proj
        predicted_target = decoder_outputs['predicted_noise']  # [batch_size, target_len, input_dim]
        
        return {
            'predicted_target': predicted_target,
            'context_embedding': context_embedding,
            'encoder_outputs': encoder_outputs,
            'decoder_outputs': decoder_outputs
        }
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SequenceDataset(Dataset):
    def __init__(self, num_samples=1000, context_len=50, target_len=20, 
                 input_dim=5, train=True):
        self.num_samples = num_samples
        self.context_len = context_len
        self.target_len = target_len
        self.input_dim = input_dim
        self.train = train
        
        # Generate synthetic time series data
        self.data = self._generate_data()
        
    def _generate_data(self):
        """Generate synthetic multivariate time series data"""
        data = []
        for i in range(self.num_samples):
            # Base signal with trend and seasonality
            t = np.linspace(0, 10, self.context_len + self.target_len)
            
            # Multiple features with different patterns
            features = []
            for feat_idx in range(self.input_dim):
                # Each feature has different characteristics
                trend = 0.1 * t * (feat_idx + 1)
                seasonal = 2 * np.sin(2 * np.pi * t / (feat_idx + 2))
                noise = 0.1 * np.random.randn(len(t))
                
                feature_signal = trend + seasonal + noise
                features.append(feature_signal)
            
            # Shape: [seq_len, input_dim]
            sequence = np.stack(features, axis=1)
            data.append(sequence)
        
        return np.array(data, dtype=np.float32)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sequence = self.data[idx]  # [total_len, input_dim]
        
        # Split into context and target
        context = sequence[:self.context_len]  # [context_len, input_dim]
        target = sequence[self.context_len:self.context_len + self.target_len]  # [target_len, input_dim]
        
        return {
            'context': torch.FloatTensor(context),
            'target': torch.FloatTensor(target),
            'full_sequence': torch.FloatTensor(sequence)
        }
class ModelAnalyzer:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        
    def analyze_predictions(self, num_examples=3):
        self.model.eval()
        
        with torch.no_grad():
            batch = next(iter(self.dataloader))
            context = batch['context'].to(self.device)
            target = batch['target'].to(self.device)
            
            outputs = self.model(context, target, teacher_forcing_ratio=0.0)
            predictions = outputs['predicted_target']
            
            # Plot examples
            fig, axes = plt.subplots(num_examples, 1, figsize=(12, 3*num_examples))
            if num_examples == 1:
                axes = [axes]
            
            for i in range(num_examples):
                # Plot first feature dimension for clarity
                context_len = context.shape[1]
                total_len = context_len + target.shape[1]
                
                # Create time axis
                time_axis = np.arange(total_len)
                
                # Plot context (input)
                axes[i].plot(time_axis[:context_len], context[i, :, 0].cpu().numpy(), 
                           'b-', label='Context', linewidth=2)
                
                # Plot actual target
                axes[i].plot(time_axis[context_len:], target[i, :, 0].cpu().numpy(), 
                           'g-', label='True Target', linewidth=2)
                
                # Plot predicted target
                axes[i].plot(time_axis[context_len:], predictions[i, :, 0].cpu().numpy(), 
                           'r--', label='Predicted', linewidth=2)
                
                axes[i].axvline(x=context_len-1, color='k', linestyle='--', alpha=0.5)
                axes[i].set_title(f'Example {i+1}')
                axes[i].set_xlabel('Time Step')
                axes[i].set_ylabel('Value')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('predictions_analysis.png')
            plt.show()
    
    def print_model_summary(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        # Print layer-wise breakdown
        for name, module in self.model.named_children():
            num_params = sum(p.numel() for p in module.parameters())
            print(f"{name:20} {num_params:>12,} parameters")
# Create datasets
train_dataset = SequenceDataset(num_samples=800, train=True)
val_dataset = SequenceDataset(num_samples=200, train=False)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def main():
    # Configuration
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Create model
    model = SimpleLSTMTransformerSeq2Seq(model_config)
    
    # Print model summary
    analyzer = ModelAnalyzer(model, val_loader, 
                           device="cpu")  # Change to "mps" if using MPS
    analyzer.print_model_summary()
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, training_config)
    
    # Train the model
    trainer.train()
    
    # Plot training curves
    trainer.plot_training()
    
    # Analyze predictions
    analyzer.analyze_predictions(num_examples=3)
    
    print("Training completed successfully!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")

if __name__ == "__main__":
    main()