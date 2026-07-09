from dataclasses import dataclass


@dataclass
class QFlexConfig:
    num_envs: int = 1024
    """number of parallel environments"""

    num_learning_iterations: int = 150000
    """total timesteps of the experiments"""

    learning_rate: float = 3e-4
    """learning rate"""
    betas: tuple[float, float] = (0.9, 0.95)
    """AdamW betas"""
    weight_decay: float = 0.001
    """AdamW weight_decay"""

    buffer_size: int = 256 * 20
    """the replay memory buffer size per environment"""
    num_steps: int = 1
    """the number of steps to use for the multi-step return"""
    gamma: float = 0.99
    """the discount factor"""
    batch_size: int = 8192
    """the batch size of sample from the replay memory"""

    learning_starts: int = 10
    """timestep to start learning"""
    num_updates: int = 8
    """the number of updates to perform per step"""
    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""

    actor_hidden_dim: int = 512
    """the hidden dimension of the actor network"""
    velocity_hidden_dim: int = 768
    """hidden dimension of velocity network"""
    log_std_max: float = 0.0
    """the maximum value of the log std"""
    log_std_min: float = -5.0
    """the minimum value of the log std"""
    alpha_init: float = 0.0
    """the initial value of the alpha"""
    alpha_learning_rate: float = 3e-4
    """the learning rate for the alpha"""

    critic_hidden_dim: int = 768
    """the hidden dimension of the critic network"""
    num_q_networks: int = 2
    """number of Q-networks to ensemble"""

    obs_normalization: bool = True
    """ whether to normalize observations """

    num_flow_steps: int = 20
    """Euler steps for sampling from the flow (``diffusion_steps``)."""
    grad_step_size: float = 1e-2
    """step size of the Q-gradient ascent that builds the flow target."""
    grad_step_num: int = 20
    """number of Q-gradient ascent steps."""
    clamp_action_bounds: bool = False
    """whether to clamp the velocity field output to action bounds"""

    @staticmethod
    def pretty_print(qflex_config):
        print("QFlex Configuration:")
        for field in qflex_config.__dataclass_fields__:
            value = getattr(qflex_config, field)
            print(f"  {field}: {value}")
