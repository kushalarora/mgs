import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class ScoreNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=1024):
        super(ScoreNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1))

    def forward(self, model, batch, pad):
        device = batch.device
        mask = batch.ne(pad).float().to(device=device)
        batch_ = deepcopy(batch).to(device=device)
        batch_[batch == pad] = 0

        model_output = model(batch_,
                            attention_mask=mask,
                            output_hidden_states=True)

        emb = model_output \
                .hidden_states[-1][:, -1, :] \
                .detach()
        output = self.fc(emb)
        return output