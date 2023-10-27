import os
import numpy as np
import torch
import torch.nn as nn

__all__ = ["AccuracyPredictor", "SigmoidAccuracyPredictor", "AccuracyPredictorFactory"]


class AccuracyPredictor(nn.Module):
    def __init__(
        self,
        arch_encoder,
        hidden_size=400,
        n_layers=3,
        checkpoint_path=None,
        device="cuda:0",
    ):
        super(AccuracyPredictor, self).__init__()
        self.arch_encoder = arch_encoder
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        # build layers
        layers = []
        for i in range(self.n_layers):
            layers.append(
                nn.Sequential(
                    nn.Linear(
                        self.arch_encoder.n_dim if i == 0 else self.hidden_size,
                        self.hidden_size,
                    ),
                    nn.ReLU(inplace=True),
                )
            )
        layers.append(nn.Linear(self.hidden_size, 2, bias=False))
        self.layers = nn.Sequential(*layers)
        self.base_acc = nn.Parameter(
            torch.zeros(2, device=self.device), requires_grad=False
        )

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            self.load_state_dict(checkpoint)
            print("Loaded checkpoint from %s" % checkpoint_path)

        self.layers = self.layers.to(self.device)

    def forward(self, x):
        y = self.layers(x).squeeze()
        return y + self.base_acc

    def predict_acc(self, arch_dict_list):
        X = [self.arch_encoder.arch2feature(arch_dict) for arch_dict in arch_dict_list]
        X = torch.tensor(np.array(X)).float().to(self.device)
        return self.forward(X)


class SigmoidAccuracyPredictor(AccuracyPredictor):
    def forward(self, x):
        return torch.sigmoid(AccuracyPredictor.forward(self, x))


AccuracyPredictorFactory = {
    'AccuracyPredictor': AccuracyPredictor,
    'SigmoidAccuracyPredictor': SigmoidAccuracyPredictor
}