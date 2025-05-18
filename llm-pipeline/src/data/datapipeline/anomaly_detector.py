import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from feature_extractor import FeatureExtractor

# Define the Sparse Autoencoder as a PyTorch Module.
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8, dropout_rate=0.2):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AnomalyDetector:
    def __init__(self, input_dim, latent_dim=15, dropout_rate=0.2):
        self.model = SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim, dropout_rate=dropout_rate)
        self.scaler = None  # To be set after feature normalization
        self.threshold = None

    def train_autoencoder(self, data_loader, num_epochs=300, lr=1e-3, weight_decay=1e-5):
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in data_loader:
                optimizer.zero_grad()
                batch_features = batch[0]
                loss = criterion(self.model(batch_features), batch_features)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}: Loss = {total_loss / len(data_loader):.4f}")

    @staticmethod
    def compute_reconstruction_error(model, feature_vector):
        model.eval()
        with torch.no_grad():
            reconstructed = model(feature_vector)
            return F.mse_loss(reconstructed, feature_vector, reduction='mean').item()

    @staticmethod
    def compute_reconstruction_error_vector(model, feature_vector):
        model.eval()
        with torch.no_grad():
            reconstructed = model(feature_vector)
            return torch.abs(reconstructed - feature_vector).squeeze(0)

    @staticmethod
    def get_anomaly_details(error_vector, raw_features, threshold_ratio=0.3, absolute_threshold=0.2):
        errors_np = error_vector.cpu().numpy()
        sorted_errors = np.sort(errors_np)[::-1]
        index_cutoff = max(1, int(len(errors_np) * threshold_ratio))
        dynamic_threshold = sorted_errors[index_cutoff - 1]
        details = []
        for i, error in enumerate(errors_np):
            if error >= max(dynamic_threshold, absolute_threshold):
                details.append(f"{FeatureExtractor.FEATURE_NAMES[i]} (raw: {raw_features[i]:.2f}, error: {error:.3f})")
        return details

    @staticmethod
    def smart_generate_label(reconstruction_error, threshold):
        return "ANOMALY_DETECTED" if reconstruction_error > threshold else "NO_ACTION"

    def generate_smart_instruction_label(self, snapshot):
        """Generate a smart label and prompt for a single snapshot."""
        raw_features = FeatureExtractor.extract_features(snapshot)
        # Assume that the scaler has been fitted and set on this instance.
        features_norm = self.scaler.transform(raw_features.reshape(1, -1))
        feature_tensor = torch.tensor(features_norm, dtype=torch.float32)
        reconstruction_error = self.compute_reconstruction_error(self.model, feature_tensor)
        error_vector = self.compute_reconstruction_error_vector(self.model, feature_tensor)
        anomaly_details = self.get_anomaly_details(error_vector, raw_features) if reconstruction_error > self.threshold else []
        label = self.smart_generate_label(reconstruction_error, self.threshold)
        # Reuse the prompt formatter for instructions
        from prompt_generation import PromptFormatter
        instruction = PromptFormatter.format_llm_prompt(snapshot)
        return {
            "instruction": instruction,
            "label": label,
            "anomaly_score": reconstruction_error,
            "anomaly_details": anomaly_details
        }

    def compute_threshold(self, features_tensor):
        """
        Compute a threshold based on reconstruction errors across the training set.
        """
        errors = [self.compute_reconstruction_error(self.model, features_tensor[i].unsqueeze(0))
                  for i in range(features_tensor.size(0))]
        errors = np.array(errors)
        self.threshold = errors.mean() + 3 * errors.std()
        print("Autoencoder error threshold:", self.threshold)
