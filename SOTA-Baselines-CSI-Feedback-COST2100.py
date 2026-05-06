import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = '/kaggle/input/datasets/pratham00727829/cost2100-indoor-csi'
N_c = 2048 # Total elements
BATCH_SIZE = 200
EPOCHS = 500
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ")

class RefineNetBlock(nn.Module):
    def __init__(self):
        super(RefineNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.LeakyReLU(0.3)
        self.conv2 = nn.Conv2d(8, 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(2)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

class CsiNet(nn.Module):
    def __init__(self, m_dim):
        super(CsiNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.3),
            nn.Flatten(),
            nn.Linear(2048, m_dim)
        )
        self.decoder_fc = nn.Linear(m_dim, 2048)
        self.decoder_refine = nn.Sequential(
            RefineNetBlock(),
            RefineNetBlock()
        )
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        x_reshaped = x.view(-1, 2, 32, 32)
        encoded = self.encoder(x_reshaped)
        decoded = self.decoder_fc(encoded)
        decoded = decoded.view(-1, 2, 32, 32)
        refined = self.decoder_refine(decoded)
        return self.final_activation(refined).view(-1, 2048)

print("Loading dataset into memory...")
mat_train = sio.loadmat(os.path.join(DATA_DIR, 'DATA_Htrainin.mat'))
mat_test = sio.loadmat(os.path.join(DATA_DIR, 'DATA_Htestin.mat'))

x_train = torch.tensor(mat_train['HT'], dtype=torch.float32)
x_test = torch.tensor(mat_test['HT'], dtype=torch.float32)

train_loader = DataLoader(TensorDataset(x_train, x_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, x_test), batch_size=BATCH_SIZE, shuffle=False)
print(f"Data loaded! Train size: {x_train.shape[0]}, Test size: {x_test.shape[0]}")

ratios_to_run = [1/8, 1/4, 1/64]
final_results = {}

for cr in ratios_to_run:
    M = int(N_c * cr)
    print("\n" + "="*50)
    print(f"🚀 INITIALIZING CSINET FOR CR = {cr:.5f} (M={M})")
    print("="*50)
    
    model = CsiNet(M).to(device)
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
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 50 == 0:
            print(f"CR [{cr:.5f}] -> Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.6f}")

    model.eval()
    nmse_sum = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            mse = torch.sum((target - output) ** 2, dim=1)
            norm = torch.sum(target ** 2, dim=1)
            nmse_batch = mse / norm
            nmse_sum += torch.sum(nmse_batch).item()

    final_nmse_linear = nmse_sum / len(x_test)
    final_nmse_db = 10 * np.log10(final_nmse_linear)
    
    final_results[cr] = final_nmse_db
    print(f"\n COMPLETED CR = {cr:.5f} | Final NMSE: {final_nmse_db:.2f} dB")

print("\n" + "="*50)
print(" MASTER SUMMARY: CSINET RESULTS")
print("="*50)
for cr, nmse in final_results.items():
    print(f"Compression Ratio: {cr:.5f}  ->  NMSE: {nmse:.2f} dB")
print("="*50)
