import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader

# --- 1. KAGGLE HYPERPARAMETERS & PATHS ---
DATA_DIR = '/kaggle/input/datasets/pratham00727829/cost2100-indoor-csi/' 
N_c = 2048 # Total elements
BATCH_SIZE = 200
EPOCHS = 500
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} 🚀")

# --- 2. TRANSNET ARCHITECTURE ---
class TransNet(nn.Module):
    def __init__(self, m_dim):
        super(TransNet, self).__init__()
        
        # 1. CNN Feature Extractor (Encoder)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.3),
            nn.Conv2d(8, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.3)
        )
        self.encoder_fc = nn.Linear(2048, m_dim)

        # 2. Fully Connected to Sequence (Decoder)
        self.decoder_fc = nn.Linear(m_dim, 2048)
        
        # 3. Transformer Blocks
        # Reshaping 2048 into a sequence of 64 tokens, each of dimension 32
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=8, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 4. Final Refinement
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.3),
            nn.Conv2d(8, 2, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.size(0)
        
        # --- Encoder ---
        x_img = x.view(B, 2, 32, 32)
        conv_feat = self.encoder_conv(x_img)
        flat_feat = conv_feat.view(B, 2048)
        codeword = self.encoder_fc(flat_feat)
        
        # --- Decoder ---
        dec_init = self.decoder_fc(codeword)
        
        # Prepare for Transformer (Batch, Seq_Len=64, D_Model=32)
        seq_in = dec_init.view(B, 64, 32)
        trans_out = self.transformer(seq_in)
        
        # Reshape back to spatial dimensions for final convolution
        img_in = trans_out.view(B, 2, 32, 32)
        out = self.decoder_conv(img_in)
        
        return out.view(B, 2048)

# --- 3. DATA LOADING (Loaded ONLY ONCE) ---
print("\nLoading dataset into memory...")
mat_train = sio.loadmat(os.path.join(DATA_DIR, 'DATA_Htrainin.mat'))
mat_test = sio.loadmat(os.path.join(DATA_DIR, 'DATA_Htestin.mat'))

x_train = torch.tensor(mat_train['HT'], dtype=torch.float32)
x_test = torch.tensor(mat_test['HT'], dtype=torch.float32)

# Pin memory for faster GPU transfer
train_loader = DataLoader(TensorDataset(x_train, x_train), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(TensorDataset(x_test, x_test), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
print(f"Data loaded! Train size: {x_train.shape[0]}, Test size: {x_test.shape[0]}")

# --- 4. MULTI-RATIO TRAINING LOOP ---
ratios_to_run = [1/8, 1/4, 1/64]
final_results = {}

for cr in ratios_to_run:
    M = int(N_c * cr)
    print("\n" + "="*50)
    print(f"🚀 INITIALIZING TRANSNET FOR CR = {cr:.5f} (M={M})")
    print("="*50)
    
    # Re-initialize model, criterion, and optimizer
    model = TransNet(M).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping is heavily recommended for Transformer-based architectures
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 50 == 0:
            print(f"CR [{cr:.5f}] -> Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.6f}")

    # --- EVALUATION ---
    model.eval()
    nmse_sum = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            mse = torch.sum((target - output) ** 2, dim=1)
            norm = torch.sum(target ** 2, dim=1)
            nmse_batch = mse / (norm + 1e-10)
            nmse_sum += torch.sum(nmse_batch).item()

    final_nmse_linear = nmse_sum / len(x_test)
    final_nmse_db = 10 * np.log10(final_nmse_linear)
    
    final_results[cr] = final_nmse_db
    print(f"\n✅ COMPLETED CR = {cr:.5f} | Final NMSE: {final_nmse_db:.2f} dB")

# --- 5. MASTER SUMMARY ---
print("\n" + "="*50)
print(" MASTER SUMMARY: TRANSNET RESULTS")
print("="*50)
for cr, nmse in final_results.items():
    print(f"Compression Ratio: {cr:.5f}  ->  NMSE: {nmse:.2f} dB")
print("="*50)
