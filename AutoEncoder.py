
import torch.nn as nn
import torch.nn.functional as F
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

def DimentionalityReductionCluster(df5):
    import torch.optim as optim
    import torch
    
    input_dim = df5.shape[1]
    data_tensor = torch.tensor(df5.values, dtype=torch.float32)
    
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    latent_dim = 5
    autoencoder = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)
    epochs = 30
    autoencoder.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            xb = batch[0]
            optimizer.zero_grad()
            xb_recon = autoencoder(xb)
            loss = criterion(xb_recon, xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    # Extract Latent Features
    autoencoder.eval()
    with torch.no_grad():
        latent_features = autoencoder.encode(data_tensor).numpy()
    
    # Scale Latent Features
    latent_features_scaled = StandardScaler().fit_transform(latent_features)
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_features_scaled)
    
    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i in range(num_clusters):
        cluster_points = latent_2d[cluster_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    label=f'Cluster {i + 1}', c=colors[i % len(colors)], alpha=0.7, edgecolor='k')
    
    plt.title("Latent Features Clustering Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2 ")
    plt.legend()
    plt.grid(True)
    plt.show()
            
        
    
