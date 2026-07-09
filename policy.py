import torch
import torch.nn as nn

from blocks import build_bn_mlp


class ReferencePolicy(nn.Module):
    def __init__(
            self,
            n_obs: int,
            n_act: int,
            hidden_dim: int,
            log_std_max: float,
            log_std_min: float,
            device: torch.device | str | None = None,
            action_scale: torch.Tensor | None = None,
            action_bias: torch.Tensor | None = None,
            num_hidden_layers: int = 3,
    ):
        super().__init__()
        self.n_obs = n_obs
        self.n_act = n_act
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        # Setup the network - this will be overridden in subclasses if needed
        self.setup_network()

        # Register action scaling parameters as buffers
        if action_scale is not None:
            self.register_buffer("action_scale", action_scale.to(device))
        else:
            self.register_buffer("action_scale", torch.ones(n_act, device=device))

        if action_bias is not None:
            self.register_buffer("action_bias", action_bias.to(device))
        else:
            self.register_buffer("action_bias", torch.zeros(n_act, device=device))

    def setup_network(self) -> None:
        """Setup the network architecture. Can be overridden by subclasses."""
        self._setup_network_with_input_dim(self.n_obs)

    def _setup_network_with_input_dim(self, input_dim: int) -> None:
        """Setup network with specific input dimension. """
        self.net = build_bn_mlp(
            input_dim,
            [self.hidden_dim] * self.num_hidden_layers,
            self.n_act * 2,
            nn.ReLU,
            device=self.device,
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.net(obs)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        tanh_mean = torch.tanh(mean)
        action = tanh_mean * self.action_scale + self.action_bias
        return action, mean, log_std

    def get_actions_and_log_probs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        # Apply tanh to get bounded actions in [-1, 1]
        tanh_action = torch.tanh(raw_action)
        # Scale and bias to get final actions
        action = tanh_action * self.action_scale + self.action_bias
        # Compute log probability with proper Jacobian correction
        log_prob = dist.log_prob(raw_action)
        # Jacobian correction for tanh transformation
        log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
        # Jacobian correction for scaling transformation
        log_prob -= torch.log(self.action_scale + 1e-6)
        log_prob = log_prob.sum(1)
        return action, log_prob

    @torch.no_grad()
    def explore(self, obs: torch.Tensor, ) -> torch.Tensor:
        _, mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        tanh_action = torch.tanh(raw_action)
        action = tanh_action * self.action_scale + self.action_bias
        return action


class VelocityField(nn.Module):
    def __init__(
            self,
            n_obs: int,
            n_act: int,
            hidden_dim: int,
            device: torch.device,
            num_hidden_layers: int = 3,
    ):
        super().__init__()
        self.net = build_bn_mlp(
            n_obs + n_act + 1,
            [hidden_dim] * num_hidden_layers,
            n_act,
            nn.ReLU,
            device=device,
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act, t], dim=-1)
        return self.net(x)


class QFlexActor(nn.Module):
    """Reference policy + velocity field with the clipped-Euler flow sampler."""

    def __init__(
            self,
            reference: ReferencePolicy,
            velocity_field: VelocityField,
            num_timesteps: int,
            device: torch.device,
            action_low: float,
            action_high: float,
    ):
        super().__init__()
        self.reference = reference
        self.velocity_field = velocity_field
        self.num_timesteps = num_timesteps
        self.action_low = action_low
        self.action_high = action_high
        self.device = device

    @torch.no_grad()
    def apply_flow(self, x0: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        x = x0.clone()

        # Clipped forward Euler over t in linspace(0, 1, num_timesteps + 1)[1:].
        ts = torch.linspace(0.0, 1.0, self.num_timesteps + 1, device=obs.device)
        dt = ts[1] - ts[0]
        for i in range(1, self.num_timesteps + 1):
            ti = ts[i].expand(obs.shape[0], 1)
            dx = self.velocity_field(obs, x, ti)
            dx = dx.clamp(-1.0 / dt, 1.0 / dt)
            x = x + dt * dx

        return x

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Inference"""
        x0 = self.reference(obs)[0]
        act = self.apply_flow(x0=x0, obs=obs)
        return act

    @torch.no_grad()
    def explore(self, obs: torch.Tensor) -> torch.Tensor:
        x0 = self.reference.explore(obs)
        act = self.apply_flow(x0=x0, obs=obs)
        return act
