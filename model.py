"""
VAJRA Nowcasting — Model
=========================
A ConvLSTM encoder-decoder: the standard architecture family behind
operational nowcasting systems (cf. Shi et al. 2015 "ConvLSTM"; Google's
MetNet; the UK Met Office's tools). It takes the last 60 minutes of fused
radar+satellite+lightning+NWP grids and predicts the next 3 hours as two
outputs:
  - future radar reflectivity (regression, dBZ)      -> storm intensity/track
  - future lightning occurrence probability (0-1)      -> lightning nowcast

Requires: pip install torch --break-system-packages
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad)

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch, H, W, device):
        return (
            torch.zeros(batch, self.hidden_ch, H, W, device=device),
            torch.zeros(batch, self.hidden_ch, H, W, device=device),
        )


class ConvLSTMEncoder(nn.Module):
    def __init__(self, in_ch, hidden_ch=32, n_layers=2):
        super().__init__()
        self.cells = nn.ModuleList(
            [ConvLSTMCell(in_ch if i == 0 else hidden_ch, hidden_ch) for i in range(n_layers)]
        )

    def forward(self, x_seq):
        # x_seq: (B, T, C, H, W)
        B, T, C, H, W = x_seq.shape
        device = x_seq.device
        states = [cell.init_state(B, H, W, device) for cell in self.cells]
        for t in range(T):
            inp = x_seq[:, t]
            for i, cell in enumerate(self.cells):
                h, c = cell(inp, states[i])
                states[i] = (h, c)
                inp = h
        return states  # final hidden states, seed the decoder


class NowcastNet(nn.Module):
    """Encoder-decoder ConvLSTM with two prediction heads."""

    def __init__(self, in_channels=5, hidden_ch=32, output_steps=18):
        super().__init__()
        self.output_steps = output_steps
        self.encoder = ConvLSTMEncoder(in_channels, hidden_ch, n_layers=2)
        self.decoder_cells = nn.ModuleList(
            [ConvLSTMCell(hidden_ch, hidden_ch) for _ in range(2)]
        )
        self.reflectivity_head = nn.Conv2d(hidden_ch, 1, kernel_size=1)
        self.lightning_head = nn.Sequential(
            nn.Conv2d(hidden_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x_seq):
        # x_seq: (B, T_in, C, H, W)
        states = self.encoder(x_seq)
        B, _, H, W = x_seq[:, 0].shape
        device = x_seq.device
        dec_input = states[-1][0]  # last hidden state feeds the decoder loop

        reflect_preds, lightning_preds = [], []
        for _ in range(self.output_steps):
            inp = dec_input
            for i, cell in enumerate(self.decoder_cells):
                h, c = cell(inp, states[i])
                states[i] = (h, c)
                inp = h
            reflect_preds.append(self.reflectivity_head(inp))
            lightning_preds.append(self.lightning_head(inp))
            dec_input = inp

        reflect = torch.stack(reflect_preds, dim=1).squeeze(2)     # (B, T_out, H, W)
        lightning = torch.stack(lightning_preds, dim=1).squeeze(2)  # (B, T_out, H, W)
        return reflect, lightning


def nowcast_loss(
    pred_reflect, pred_lightning, true_reflect, true_lightning,
    lightning_weight=2.0, reflect_mean=None, reflect_std=None,
):
    """Combined loss: MSE on reflectivity + weighted BCE on lightning.

    true_reflect/pred_reflect arrive in raw dBZ, while pred_lightning/
    true_lightning are 0-1 probabilities. Computing MSE on the raw dBZ scale
    makes it orders of magnitude larger than the BCE term, so the model
    would effectively ignore lightning loss no matter what lightning_weight
    is set to. Normalize reflectivity onto a comparable scale before the MSE
    so the two loss terms (and lightning_weight) mean what they say.

    reflect_mean/reflect_std are required (no silent default): callers must
    pass the real archive-derived radar_dbz stats from
    normalization.load_normalization_stats(), not a guessed constant, since
    those values move as the dataset changes. See train.py for the pattern.
    """
    if reflect_mean is None or reflect_std is None:
        raise ValueError(
            "nowcast_loss requires reflect_mean/reflect_std from "
            "normalization.load_normalization_stats() — no hardcoded "
            "default is provided on purpose, so a missing stats file "
            "fails loudly instead of silently training against the "
            "wrong scale."
        )
    pred_reflect_norm = (pred_reflect - reflect_mean) / reflect_std
    true_reflect_norm = (true_reflect - reflect_mean) / reflect_std
    mse = nn.functional.mse_loss(pred_reflect_norm, true_reflect_norm)
    bce = nn.functional.binary_cross_entropy(pred_lightning, true_lightning)
    return mse + lightning_weight * bce, {"mse": mse.item(), "bce": bce.item()}


if __name__ == "__main__":
    B, T_in, C, H, W = 2, 6, 5, 128, 128
    model = NowcastNet(in_channels=C, hidden_ch=16, output_steps=18)
    x = torch.randn(B, T_in, C, H, W)
    reflect, lightning = model(x)
    print("Reflectivity forecast:", reflect.shape)   # (2, 18, 128, 128)
    print("Lightning probability:", lightning.shape)  # (2, 18, 128, 128)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
