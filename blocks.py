import torch
import torch.nn as nn


class BatchRenorm1d(nn.Module):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            decay_rate: float = 0.99,
            r_max: float = 3.0,
            d_max: float = 5.0,
            warmup_steps: int = 10,
            device: torch.device | str | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.momentum = 1.0 - decay_rate  # EMA: new = decay * old + (1 - decay) * batch
        self.r_max = r_max
        self.d_max = d_max
        self.warmup_steps = warmup_steps

        # scale / offset, matching Haiku's create_scale / create_offset.
        self.weight = nn.Parameter(torch.ones(num_features, device=device))
        self.bias = nn.Parameter(torch.zeros(num_features, device=device))
        self.register_buffer("running_mean", torch.zeros(num_features, device=device))
        self.register_buffer("running_var", torch.ones(num_features, device=device))
        self.register_buffer("num_batches_tracked", torch.zeros((), device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = (x * x).mean(dim=0) - batch_mean ** 2
            batch_var = batch_var.clamp_min(self.eps)

            std = torch.sqrt((batch_var + self.eps).clamp_min(self.eps))
            ra_std = torch.sqrt((self.running_var + self.eps).clamp_min(self.eps))

            # r and d are treated as constants (stop-gradient)
            r = (std / ra_std).detach().clamp(1.0 / self.r_max, self.r_max)
            d = ((batch_mean - self.running_mean) / ra_std).detach().clamp(-self.d_max, self.d_max)

            warmed_up = (self.num_batches_tracked >= self.warmup_steps).float()
            renorm_var = batch_var / (r ** 2)
            renorm_mean = batch_mean - d * torch.sqrt(batch_var) / r
            used_var = warmed_up * renorm_var + (1.0 - warmed_up) * batch_var
            used_mean = warmed_up * renorm_mean + (1.0 - warmed_up) * batch_mean

            # EMA update uses the raw (un-renormalised) batch statistics.
            with torch.no_grad():
                self.running_mean += self.momentum * (batch_mean - self.running_mean)
                self.running_var += self.momentum * (batch_var - self.running_var)
                self.num_batches_tracked += 1
        else:
            used_mean = self.running_mean
            used_var = self.running_var

        inv = self.weight * torch.rsqrt(used_var + self.eps)
        return (x - used_mean) * inv + self.bias


def build_bn_mlp(
        input_dim: int,
        hidden_sizes: list[int],
        output_size: int,
        activation: type[nn.Module],
        device: torch.device | str | None = None,
) -> nn.Sequential:
    """ ``BN(input) -> [Linear -> act -> BN] * len(hidden) -> Linear(output)`` """
    layers: list[nn.Module] = [BatchRenorm1d(input_dim, device=device)]
    in_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(in_dim, hidden_size, device=device))
        layers.append(activation())
        layers.append(BatchRenorm1d(hidden_size, device=device))
        in_dim = hidden_size
    layers.append(nn.Linear(in_dim, output_size, device=device))
    return nn.Sequential(*layers)
