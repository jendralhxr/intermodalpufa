import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Parameters
N_SAMPLES = 100
GC_DIM = 128
NIR_DIM = 256
LATENT_DIM = 64

# loader helper
import pandas as pd

def load_gc_csv(filepath):
    """
    Load GC-FID data from a CSV file.
    Each row should be a sample; each column is a retention time bin.
    Returns: np.ndarray of shape (n_samples, n_features)
    """
    df = pd.read_csv(filepath)
    return df.values.astype(np.float32)

def load_nir_csv(filepath):
    """
    Load FT-NIR data from a CSV file.
    Each row should be a sample; each column is a wavelength bin.
    Returns: np.ndarray of shape (n_samples, n_features)
    """
    df = pd.read_csv(filepath)
    return df.values.astype(np.float32)

 


# Dummy data: each row is a sample
gc_data = np.random.rand(N_SAMPLES, GC_DIM).astype(n.float32)
nir_data = np.random.rand(N_SAMPLES, NIR_DIM).astype(np.float32)

# Dataset
class PairedSpectraDataset(Dataset):
    def __init__(self, gc_data, nir_data):
        self.gc = torch.tensor(gc_data)
        self.nir = torch.tensor(nir_data)

    def __len__(self):
        return len(self.gc)

    def __getitem__(self, idx):
        return self.gc[idx], self.nir[idx]

dataset = PairedSpectraDataset(gc_data, nir_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Neural network blocks
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, z):
        return self.net(z)

# Models
gc_encoder = Encoder(GC_DIM, LATENT_DIM)
nir_encoder = Encoder(NIR_DIM, LATENT_DIM)
gc_decoder = Decoder(LATENT_DIM, GC_DIM)
nir_decoder = Decoder(LATENT_DIM, NIR_DIM)

# Training setup
params = list(gc_encoder.parameters()) + list(nir_encoder.parameters()) + \
         list(gc_decoder.parameters()) + list(nir_decoder.parameters())
optimizer = optim.Adam(params, lr=1e-3)
mse = nn.MSELoss()

# Training loop
EPOCHS = 20
for epoch in range(EPOCHS):
    total_loss = 0
    for gc, nir in dataloader:
        optimizer.zero_grad()
        
        # Encode
        gc_latent = gc_encoder(gc)
        nir_latent = nir_encoder(nir)
        
        # Decode
        gc_recon = gc_decoder(gc_latent)
        nir_recon = nir_decoder(nir_latent)
        
        # Losses
        loss_gc = mse(gc_recon, gc)
        loss_nir = mse(nir_recon, nir)
        loss_align = mse(gc_latent, nir_latent)
        
        # Combine
        loss = loss_gc + loss_nir + 0.5 * loss_align
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")
    
    
 # Inference: GC-FID → FT-NIR
def predict_nir_from_gc(gc_input, gc_encoder, nir_decoder):
    """
    Predict FT-NIR spectrum from a GC-FID chromatogram.
    gc_input: 1D numpy array or tensor of shape (GC_DIM,)
    """
    gc_input = torch.tensor(gc_input, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        latent = gc_encoder(gc_input)
        predicted_nir = nir_decoder(latent)
    return predicted_nir.squeeze(0).numpy()

# Inference: FT-NIR → GC-FID
def predict_gc_from_nir(nir_input, nir_encoder, gc_decoder):
    """
    Predict GC-FID chromatogram from an FT-NIR spectrum.
    nir_input: 1D numpy array or tensor of shape (NIR_DIM,)
    """
    nir_input = torch.tensor(nir_input, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        latent = nir_encoder(nir_input)
        predicted_gc = gc_decoder(latent)
    return predicted_gc.squeeze(0).numpy()

# dry run test
# Example test input
# test_gc = gc_data[0]     # input GC chromatogram
# test_nir = nir_data[0]   # input FT-NIR spectrum

# Predict FT-NIR from GC
# nir_pred = predict_nir_from_gc(test_gc, gc_encoder, nir_decoder)

# Predict GC-FID from FT-NIR
# gc_pred = predict_gc_from_nir(test_nir, nir_encoder, gc_decoder)

# Print prediction shapes
# print("Predicted NIR shape:", nir_pred.shape)
# print("Predicted GC shape:", gc_pred.shape)

