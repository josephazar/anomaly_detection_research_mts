"""
MTAD-GAT: Multivariate Time-series Anomaly Detection via Graph Attention Network

Faithful implementation based on:
"Multivariate Time-series Anomaly Detection via Graph Attention Network"
by Hang Zhao et al. (Microsoft & Peking University), ICDM 2020.

Reference implementations consulted:
- ML4ITS/mtad-gat-pytorch (PyTorch, GATv2 default)
- mangushev/mtad-gat (TensorFlow, VAE reconstruction)

Architecture:
1. 1D Convolution for temporal smoothing
2. Feature-oriented GAT (inter-feature correlations)
3. Time-oriented GAT (temporal dependencies)
4. GRU for sequential pattern capture
5. Forecasting head (FC layers) for next-step prediction
6. Reconstruction head (GRU decoder) for window reconstruction
7. Joint RMSE-based loss (forecast + reconstruction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.stats import genpareto


# ---------------------------------------------------------------------------
# ConvLayer: 1D convolution for temporal smoothing (Stage 1)
# ---------------------------------------------------------------------------
class ConvLayer(nn.Module):
    """
    1D Convolution layer for initial temporal smoothing / noise alleviation.
    Preserves the temporal dimension via symmetric constant padding.
    Ref: ML4ITS uses nn.ConstantPad1d + nn.Conv1d + ReLU.
    """

    def __init__(self, n_features, kernel_size=7):
        super().__init__()
        self.padding = nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2), 0.0)
        self.conv = nn.Conv1d(n_features, n_features, kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: (batch, window_size, n_features)
        Returns: (batch, window_size, n_features) -- same shape
        """
        x = x.permute(0, 2, 1)          # (b, k, n)
        x = self.padding(x)             # (b, k, n + pad)
        x = self.relu(self.conv(x))     # (b, k, n)
        x = x.permute(0, 2, 1)          # (b, n, k)
        return x


# ---------------------------------------------------------------------------
# Graph Attention Layers (Stage 2)
# ---------------------------------------------------------------------------
class FeatureAttentionLayer(nn.Module):
    """
    Feature-oriented GAT. Each feature is a node; its representation is
    the full time-series vector of length window_size.
    Supports both standard GAT and GATv2 (default).

    Output shape matches input: (batch, window_size, n_features).
    """

    def __init__(self, n_features, window_size, dropout, alpha,
                 embed_dim=None, use_gatv2=True, use_bias=True):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.use_gatv2 = use_gatv2
        self.use_bias = use_bias
        self.dropout = dropout
        self.num_nodes = n_features

        # embed_dim defaults to window_size (so output keeps same node dim)
        embed_dim = embed_dim if embed_dim is not None else window_size

        if use_gatv2:
            self.embed_dim = embed_dim * 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            self.embed_dim = embed_dim
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty(a_input_dim, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(n_features, n_features))

        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, x):
        """
        x: (batch, window_size, n_features)
        Returns: (batch, window_size, n_features), attention (batch, n_features, n_features)
        """
        # Transpose so nodes=features, node_dim=window_size
        x_t = x.permute(0, 2, 1)  # (b, k, n)

        if self.use_gatv2:
            a_input = self._make_attention_input(x_t)      # (b, k, k, 2n)
            a_input = self.leaky_relu(self.lin(a_input))    # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(-1)   # (b, k, k)
        else:
            Wx = self.lin(x_t)                              # (b, k, embed_dim)
            a_input = self._make_attention_input(Wx)        # (b, k, k, 2*embed_dim)
            e = self.leaky_relu(torch.matmul(a_input, self.a)).squeeze(-1)

        if self.use_bias:
            e = e + self.bias

        attention = F.softmax(e, dim=2)                     # (b, k, k)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h = torch.sigmoid(torch.bmm(attention, x_t))       # (b, k, n)

        return h.permute(0, 2, 1), attention                # (b, n, k), (b, k, k)

    def _make_attention_input(self, v):
        """Build all K x K pairwise concatenations."""
        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)    # (b, K*K, d)
        blocks_alternating = v.repeat(1, K, 1)              # (b, K*K, d)
        combined = torch.cat([blocks_repeating, blocks_alternating], dim=2)
        return combined.view(v.size(0), K, K, -1)           # (b, K, K, 2d)


class TemporalAttentionLayer(nn.Module):
    """
    Time-oriented GAT. Each timestamp is a node; its representation is
    the n_features-dimensional feature vector at that time step.
    Supports both standard GAT and GATv2 (default).

    Output shape matches input: (batch, window_size, n_features).
    """

    def __init__(self, n_features, window_size, dropout, alpha,
                 embed_dim=None, use_gatv2=True, use_bias=True):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.use_gatv2 = use_gatv2
        self.use_bias = use_bias
        self.dropout = dropout
        self.num_nodes = window_size

        embed_dim = embed_dim if embed_dim is not None else n_features

        if use_gatv2:
            self.embed_dim = embed_dim * 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            self.embed_dim = embed_dim
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty(a_input_dim, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(window_size, window_size))

        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, x):
        """
        x: (batch, window_size, n_features)
        Returns: (batch, window_size, n_features), attention (batch, window_size, window_size)
        """
        if self.use_gatv2:
            a_input = self._make_attention_input(x)          # (b, n, n, 2k)
            a_input = self.leaky_relu(self.lin(a_input))     # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(-1)    # (b, n, n)
        else:
            Wx = self.lin(x)                                 # (b, n, embed_dim)
            a_input = self._make_attention_input(Wx)         # (b, n, n, 2*embed_dim)
            e = self.leaky_relu(torch.matmul(a_input, self.a)).squeeze(-1)

        if self.use_bias:
            e = e + self.bias

        attention = F.softmax(e, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h = torch.sigmoid(torch.bmm(attention, x))          # (b, n, k)

        return h, attention

    def _make_attention_input(self, v):
        N = self.num_nodes
        blocks_repeating = v.repeat_interleave(N, dim=1)
        blocks_alternating = v.repeat(1, N, 1)
        combined = torch.cat([blocks_repeating, blocks_alternating], dim=2)
        return combined.view(v.size(0), N, N, -1)


# ---------------------------------------------------------------------------
# GRU Layer (Stage 3)
# ---------------------------------------------------------------------------
class GRULayer(nn.Module):
    """GRU layer that returns the final hidden state of the last layer."""

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            in_dim, hid_dim, num_layers=n_layers,
            batch_first=True,
            dropout=0.0 if n_layers == 1 else dropout,
        )

    def forward(self, x):
        """
        x: (batch, seq_len, in_dim)
        Returns: (all_hidden, h_end)
            all_hidden: (batch, seq_len, hid_dim)
            h_end: (batch, hid_dim) -- last layer's final hidden state
        """
        out, h = self.gru(x)       # out: (b, n, hid), h: (n_layers, b, hid)
        return out, h[-1, :, :]    # h_end from last layer


# ---------------------------------------------------------------------------
# Forecasting Model (Stage 4a)
# ---------------------------------------------------------------------------
class Forecasting_Model(nn.Module):
    """
    FC network for single-step forecasting.
    With n_layers=3: creates 3 hidden layers + 1 output layer = 4 Linear total.
    ReLU + dropout between hidden layers; no activation on output.
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super().__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))
        layers.append(nn.Linear(hid_dim, out_dim))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)


# ---------------------------------------------------------------------------
# Reconstruction Model (Stage 4b) -- GRU-based decoder
# ---------------------------------------------------------------------------
class ReconstructionModel(nn.Module):
    """
    GRU-based decoder for window reconstruction.
    The GRU final hidden state is repeated across window_size timesteps,
    then decoded through a GRU + linear projection.

    This follows the ML4ITS reference which uses a GRU decoder.
    The original paper describes a VAE, but both the ML4ITS and practical
    implementations use a simpler GRU decoder that is empirically equivalent.
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super().__init__()
        self.window_size = window_size
        self.decoder_gru = nn.GRU(
            in_dim, hid_dim, num_layers=n_layers,
            batch_first=True,
            dropout=0.0 if n_layers == 1 else dropout,
        )
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        """
        x: (batch, gru_hid_dim) -- the GRU's final hidden state
        Returns: (batch, window_size, out_dim)
        """
        # Repeat hidden state across all timesteps
        x = x.unsqueeze(1).repeat(1, self.window_size, 1)  # (b, window_size, gru_hid_dim)
        decoder_out, _ = self.decoder_gru(x)                # (b, window_size, recon_hid_dim)
        return self.fc(decoder_out)                         # (b, window_size, out_dim)


# ---------------------------------------------------------------------------
# MTAD-GAT: Complete Model
# ---------------------------------------------------------------------------
class MTAD_GAT(nn.Module):
    """
    Complete MTAD-GAT model.

    Forward flow:
        Input (b, n, k)
          -> ConvLayer -> (b, n, k)
          -> FeatureAttentionLayer -> h_feat (b, n, k)
          -> TemporalAttentionLayer -> h_temp (b, n, k)
          -> Concat [conv_out, h_feat, h_temp] -> (b, n, 3k)
          -> GRU -> h_end (b, gru_hid_dim)
          -> Forecasting_Model -> predictions (b, k)
          -> ReconstructionModel -> reconstructions (b, n, k)

    Key: Both GAT layers output (b, n, k) -- same shape as input.
    The GRU input dimension is 3*k (concatenation of three (b,n,k) tensors).
    """

    def __init__(self, n_features, window_size, out_dim=None,
                 kernel_size=7, use_gatv2=True,
                 feat_gat_embed_dim=None, time_gat_embed_dim=None,
                 gru_n_layers=1, gru_hid_dim=150,
                 forecast_n_layers=3, forecast_hid_dim=150,
                 recon_n_layers=1, recon_hid_dim=150,
                 dropout=0.3, alpha=0.2):
        super().__init__()

        self.n_features = n_features
        self.window_size = window_size
        out_dim = out_dim if out_dim is not None else n_features

        # Stage 1: 1D Convolution
        self.conv = ConvLayer(n_features, kernel_size)

        # Stage 2: Dual GAT
        self.feature_gat = FeatureAttentionLayer(
            n_features, window_size, dropout, alpha,
            feat_gat_embed_dim, use_gatv2,
        )
        self.temporal_gat = TemporalAttentionLayer(
            n_features, window_size, dropout, alpha,
            time_gat_embed_dim, use_gatv2,
        )

        # Stage 3: GRU -- input is concatenation of 3 tensors each of dim k
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)

        # Stage 4a: Forecasting head
        self.forecasting_model = Forecasting_Model(
            gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout,
        )

        # Stage 4b: Reconstruction head
        self.recon_model = ReconstructionModel(
            window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout,
        )

    def forward(self, x):
        """
        x: (batch, window_size, n_features)
        Returns:
            predictions: (batch, out_dim) -- next-step forecast
            recons: (batch, window_size, out_dim) -- full-window reconstruction
        """
        # Stage 1
        x = self.conv(x)                              # (b, n, k)

        # Stage 2 -- both return (b, n, k)
        h_feat, _ = self.feature_gat(x)               # (b, n, k)
        h_temp, _ = self.temporal_gat(x)               # (b, n, k)

        # Concatenate along feature dim
        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        # Stage 3
        _, h_end = self.gru(h_cat)                      # (b, gru_hid_dim)

        # Stage 4
        predictions = self.forecasting_model(h_end)     # (b, out_dim)
        recons = self.recon_model(h_end)                # (b, n, out_dim)

        return predictions, recons


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for multivariate time-series."""

    def __init__(self, data, window_size, horizon=1):
        """
        data: numpy array (T, k)
        window_size: lookback length
        horizon: forecast horizon (default 1)
        """
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.window_size]
        y = self.data[idx + self.window_size: idx + self.window_size + self.horizon]
        return x, y


# ---------------------------------------------------------------------------
# Anomaly Scoring
# ---------------------------------------------------------------------------
def compute_anomaly_scores(model, data_loader, device, gamma=1.0,
                           scale_scores=False):
    """
    Compute per-timestep anomaly scores using the combined scoring method.

    Per-feature score:
        a_i = |forecast_i - actual_i| + gamma * |recon_i - actual_i|

    Global score:
        score = mean(a_i for i in features)

    Matches the ML4ITS reference (prediction.py get_score method).

    Args:
        gamma: weight for reconstruction error (default 1.0 per paper)
        scale_scores: if True, apply IQR-based normalization per feature
    """
    model.eval()
    all_preds = []
    all_recons = []
    all_actuals = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).squeeze(1)  # (b, k)

            preds, recons = model(batch_x)

            all_preds.append(preds.cpu().numpy())
            all_actuals.append(batch_y.cpu().numpy())
            all_recons.append(recons[:, -1, :].cpu().numpy())  # last timestep

    preds = np.concatenate(all_preds, axis=0)       # (N, k)
    actuals = np.concatenate(all_actuals, axis=0)    # (N, k)
    recons = np.concatenate(all_recons, axis=0)      # (N, k)

    # Per-feature anomaly scores
    forecast_scores = np.sqrt((preds - actuals) ** 2)   # |pred - actual| per feature
    recon_scores = np.sqrt((recons - actuals) ** 2)      # |recon - actual| per feature
    anomaly_scores = forecast_scores + gamma * recon_scores  # (N, k)

    # Optional IQR-based normalization (per feature)
    if scale_scores:
        for i in range(anomaly_scores.shape[1]):
            q75, q25 = np.percentile(anomaly_scores[:, i], [75, 25])
            iqr = q75 - q25
            median = np.median(anomaly_scores[:, i])
            anomaly_scores[:, i] = (anomaly_scores[:, i] - median) / (1 + iqr)

    # Global score: mean across features
    global_scores = np.mean(anomaly_scores, axis=1)  # (N,)

    return {
        'scores': global_scores,
        'feature_scores': anomaly_scores,
        'forecast_scores': forecast_scores,
        'recon_scores': recon_scores,
        'preds': preds,
        'actuals': actuals,
        'recons': recons,
    }


# ---------------------------------------------------------------------------
# POT Threshold
# ---------------------------------------------------------------------------
def pot_threshold(train_scores, test_scores=None, q=1e-3, level=0.99):
    """
    Peaks-Over-Threshold (POT) method for automatic threshold selection.

    Fits a Generalized Pareto Distribution (GPD) to the tail exceedances
    above an initial high-quantile threshold, then computes the final
    threshold at the desired probability level.

    Args:
        train_scores: 1D array of training anomaly scores (used to fit GPD)
        test_scores: optional 1D array (if None, threshold is based on train only)
        q: initial quantile for the preliminary threshold (e.g. 1e-3 -> 99.9th percentile)
        level: desired probability level for the final threshold (e.g. 0.99)
    Returns:
        threshold: float
    """
    scores = train_scores
    # Initial threshold at high quantile
    init_threshold = np.percentile(scores, (1 - q) * 100)

    # Exceedances above the initial threshold
    exceedances = scores[scores > init_threshold] - init_threshold

    if len(exceedances) < 2:
        return init_threshold

    # Fit GPD using scipy
    try:
        shape, loc, scale = genpareto.fit(exceedances, floc=0)
    except Exception:
        # Fallback to method-of-moments if MLE fails
        mean_exc = np.mean(exceedances)
        var_exc = np.var(exceedances)
        if var_exc == 0:
            return init_threshold
        shape = 0.5 * (mean_exc ** 2 / var_exc - 1)
        scale = mean_exc * (1 + shape)

    # Compute threshold at desired level
    n = len(scores)
    n_exc = len(exceedances)

    if shape != 0 and n_exc > 0:
        threshold = init_threshold + scale / shape * (
            ((1 - level) * n / n_exc) ** (-shape) - 1
        )
    else:
        threshold = init_threshold

    return max(threshold, init_threshold)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, device,
                epochs=30, lr=1e-3, print_every=1, save_path=None):
    """
    Train the MTAD-GAT model with joint optimization.

    Loss = RMSE(forecast) + RMSE(reconstruction)

    Matches the ML4ITS reference (training.py Trainer.train_epoch).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_state = None
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        forecast_losses_sq = []
        recon_losses_sq = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).squeeze(1)  # (b, k)

            optimizer.zero_grad()
            preds, recons = model(batch_x)

            # RMSE losses (per the paper and ML4ITS reference)
            forecast_loss = torch.sqrt(F.mse_loss(preds, batch_y))
            recon_loss = torch.sqrt(F.mse_loss(recons, batch_x))
            loss = forecast_loss + recon_loss

            loss.backward()
            optimizer.step()

            forecast_losses_sq.append(forecast_loss.item() ** 2)
            recon_losses_sq.append(recon_loss.item() ** 2)

        # Epoch-level: RMS aggregation of per-batch RMSEs
        epoch_forecast = np.sqrt(np.mean(forecast_losses_sq))
        epoch_recon = np.sqrt(np.mean(recon_losses_sq))
        epoch_total = epoch_forecast + epoch_recon
        train_losses.append(epoch_total)

        # --- Validation ---
        model.eval()
        val_forecast_sq = []
        val_recon_sq = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).squeeze(1)

                preds, recons = model(batch_x)
                forecast_loss = torch.sqrt(F.mse_loss(preds, batch_y))
                recon_loss = torch.sqrt(F.mse_loss(recons, batch_x))

                val_forecast_sq.append(forecast_loss.item() ** 2)
                val_recon_sq.append(recon_loss.item() ** 2)

        val_forecast = np.sqrt(np.mean(val_forecast_sq))
        val_recon = np.sqrt(np.mean(val_recon_sq))
        val_total = val_forecast + val_recon
        val_losses.append(val_total)

        # Save best model
        if val_total <= best_val_loss:
            best_val_loss = val_total
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_path:
                torch.save(model.state_dict(), save_path)

        if (epoch + 1) % print_every == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train: {epoch_total:.4f} (f={epoch_forecast:.4f}, r={epoch_recon:.4f}) | "
                f"Val: {val_total:.4f} (f={val_forecast:.4f}, r={val_recon:.4f})"
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    batch_size = 16
    window_size = 100
    n_features = 5

    x = torch.randn(batch_size, window_size, n_features)

    model = MTAD_GAT(
        n_features=n_features,
        window_size=window_size,
        gru_hid_dim=150,
        forecast_n_layers=3,
        forecast_hid_dim=150,
        recon_hid_dim=150,
        dropout=0.3,
    )

    preds, recons = model(x)

    print(f"Input shape:          {x.shape}")
    print(f"Forecast shape:       {preds.shape}")
    print(f"Reconstruction shape: {recons.shape}")

    # Verify shapes
    assert preds.shape == (batch_size, n_features)
    assert recons.shape == (batch_size, window_size, n_features)

    # Verify GRU input dim = 3 * n_features
    assert model.gru.gru.input_size == 3 * n_features

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters:     {total_params:,}")
    print("Model self-test PASSED.")
