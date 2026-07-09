import torch
import torch.nn as nn

from blocks import build_bn_mlp


class CrossQNetwork(nn.Module):
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
            input_dim=n_obs + n_act,
            hidden_sizes=[hidden_dim] * num_hidden_layers,
            output_size=1,
            activation=nn.ReLU,
            device=device,
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        x = self.net(x)
        return x


class Critic(nn.Module):
    def __init__(
            self,
            n_obs: int,
            n_act: int,
            hidden_dim: int,
            device: torch.device,
            num_q_networks: int = 2,
            num_hidden_layers: int = 3,
    ):
        super().__init__()

        self.n_obs = n_obs
        self.n_act = n_act
        self.hidden_dim = hidden_dim
        self.num_q_networks = num_q_networks
        self.num_hidden_layers = num_hidden_layers
        self.device = device

        # Setup Q-networks - this will be overridden in subclasses if needed
        assert num_q_networks >= 1, "Number of Q networks must be at least 1"
        self.setup_qnetworks()
        return

    def setup_qnetworks(self) -> None:
        """Setup Q-networks. Can be overridden by subclasses."""
        self._setup_qnetworks_with_obs_dim(self.n_obs)

    def _setup_qnetworks_with_obs_dim(self, n_obs: int) -> None:
        """Setup Q-networks with specific observation dimension."""
        self.qnets = nn.ModuleList(
            [
                CrossQNetwork(
                    n_obs=n_obs,
                    n_act=self.n_act,
                    hidden_dim=self.hidden_dim,
                    device=self.device,
                    num_hidden_layers=self.num_hidden_layers,
                )
                for _ in range(self.num_q_networks)
            ]
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """ Returns Q(s,a) for each Q-network """
        x = obs
        outputs = [qnet(x, actions) for qnet in self.qnets]
        return torch.stack(outputs, dim=0)  # (num_q_networks, batch, 1)

    def forward_joint(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            next_obs: torch.Tensor,
            next_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate (obs, actions) and (next_obs, next_actions) along the
        batch dimension push them through each Q-network in a single forward pass.
        """
        batch_size = obs.shape[0]
        joint_obs = torch.cat([obs, next_obs], dim=0)
        joint_actions = torch.cat([actions, next_actions], dim=0)

        joint_outputs = [qnet(joint_obs, joint_actions) for qnet in self.qnets]
        joint_outputs = torch.stack(joint_outputs, dim=0)  # (num_q_networks, 2*batch, 1)

        current_q = joint_outputs[:, :batch_size]
        next_q = joint_outputs[:, batch_size:]
        return current_q, next_q

    def q_value(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """ Returns clipped double-Q value min_i Q_i(s, a) """
        q_values = self.forward(obs, actions).squeeze(-1)  # (num_q_networks, batch)
        return q_values.amin(dim=0)  # (batch)
