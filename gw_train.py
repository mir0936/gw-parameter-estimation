import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Load Fortran data
print("Loading data...")
data = np.loadtxt('waveforms.csv', delimiter=',')

# Extract labels (Mass) and normalize to 0-1 range for stable training
y_raw = torch.tensor(data[:, 0], dtype=torch.float32).unsqueeze(1)
y_norm = (y_raw - 10.0) / 40.0 

# Extract waveforms and inject LIGO-style static noise
X = torch.tensor(data[:, 1:], dtype=torch.float32).unsqueeze(1)
noise = torch.randn_like(X) * 1.5 
X_noisy = X + noise

train_size = 4000
X_train, y_train = X_noisy[:train_size], y_norm[:train_size]

# Define 1D Residual Network (ResNet)
class GWResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 16, kernel_size=5, padding=2)
        self.fc = nn.Linear(16 * 200, 1) # Output is 1 continuous value

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        res = out1 + out2 # Skip connection prevents vanishing gradients
        res = res.view(res.size(0), -1)
        return self.fc(res)

model = GWResNet()
criterion = nn.MSELoss() # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.002)

# Train model
epochs = 15
losses = []
print("Training 1D ResNet...")
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Epoch {epoch+1}/{epochs}, MSE Loss: {loss.item():.4f}")

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(losses, color='purple', linewidth=2)
ax1.set_title('Regression Training Loss (MSE)')
ax1.set_xlabel('Epoch')

sample_idx = 0
ax2.plot(X_noisy[sample_idx, 0].numpy(), label='Detected Signal (Noisy)', color='gray', alpha=0.6)
ax2.plot(X[sample_idx, 0].numpy(), label='True Waveform', color='black', linewidth=1.5)
ax2.set_title(f'Gravitational Wave (Chirp Mass: {y_raw[sample_idx, 0].item():.1f} M_sun)')
ax2.legend()

plt.tight_layout()
plt.savefig('gw_results.png')
print("Complete. Saved architecture visualization to gw_results.png.")
