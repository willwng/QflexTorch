import math

import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from buffer import SimpleReplayBuffer, sample_and_prepare_batches, collect_experience
from critic import Critic
from hyperparams import QFlexConfig
from normalizers import EmpiricalNormalization
from policy import ReferencePolicy, QFlexActor, VelocityField


def train(
        cfg: QFlexConfig,
        envs,
        device: torch.device,
):
    # ------------------------------------------------------------------ envs
    n_obs, n_act = envs.num_obs(), envs.num_actions()
    action_low, action_high = envs.action_range

    # --------------------------------------------------------------- networks
    critic = Critic(
        n_obs=n_obs,
        n_act=n_act,
        hidden_dim=cfg.critic_hidden_dim,
        num_q_networks=cfg.num_q_networks,
        num_hidden_layers=cfg.hidden_num,
        device=device,
    )

    reference = ReferencePolicy(
        n_obs=n_obs,
        n_act=n_act,
        hidden_dim=cfg.actor_hidden_dim,
        log_std_max=cfg.log_std_max,
        log_std_min=cfg.log_std_min,
        num_hidden_layers=cfg.hidden_num,
        device=device
    )
    velocity_field = VelocityField(
        n_obs=n_obs,
        n_act=n_act,
        hidden_dim=cfg.velocity_hidden_dim,
        num_hidden_layers=cfg.hidden_num,
        device=device
    )
    actor = QFlexActor(
        reference=reference,
        velocity_field=velocity_field,
        num_timesteps=cfg.num_flow_steps,
        device=device,
        action_low=action_low,
        action_high=action_high,
    ).to(device)

    q_optimizer = optim.AdamW(
        list(critic.parameters()),
        lr=cfg.learning_rate,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    ref_optimizer = optim.AdamW(
        list(reference.parameters()),
        lr=cfg.learning_rate,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    vel_optimizer = optim.AdamW(
        list(velocity_field.parameters()),
        lr=cfg.learning_rate,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    obs_normalizer = (
        EmpiricalNormalization(shape=n_obs, device=device)
        if cfg.obs_normalization
        else torch.nn.Identity()
    )

    rb = SimpleReplayBuffer(
        n_env=cfg.num_envs,
        buffer_size=cfg.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_steps=cfg.num_steps,
        gamma=cfg.gamma,
        device=device,
    )

    max_update = 2.0 * math.sqrt(n_act)  # bound on the Q-gradient-ascent step

    # ---------------------------------------------------------- update steps
    def update_critic(data):
        critic.train()
        observations = data["observations"]
        next_observations = data["next"]["observations"]
        critic_observations = observations
        next_critic_observations = next_observations
        actions = data["actions"]
        rewards = data["next"]["rewards"]
        dones = data["next"]["dones"].bool()
        truncations = data["next"]["truncations"].bool()
        bootstrap = (truncations | ~dones).float()

        with torch.no_grad():
            next_actions, next_log_probs = reference.get_actions_and_log_probs(next_observations)
            discount = cfg.gamma ** data["next"]["effective_n_steps"]

        qf_current, qf_next = critic.forward_joint(
            critic_observations, actions, next_critic_observations, next_actions
        )
        qf1, qf2 = qf_current.squeeze(-1)
        qf1_next_value, qf2_next_value = qf_next.detach().squeeze(-1)
        qf_next_value = torch.minimum(qf1_next_value, qf2_next_value)
        qf_next_target = rewards + bootstrap * discount * qf_next_value.unsqueeze(-1)

        qf1_loss = F.mse_loss(qf1, qf_next_target.squeeze(-1))
        qf2_loss = F.mse_loss(qf2, qf_next_target.squeeze(-1))
        q_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        q_optimizer.step()

        return {
            "q_loss": q_loss.detach(),
            "q_min": qf_next_value.min().detach(),
            "q_max": qf_next_value.max().detach(),
        }

    def update_reference(data):
        critic.eval()
        critic_observations = data["observations"]
        actions, log_probs = reference.get_actions_and_log_probs(data["observations"])

        q_value = critic.q_value(critic_observations, actions)
        ref_loss = -q_value.mean()

        ref_optimizer.zero_grad(set_to_none=True)
        ref_loss.backward()
        ref_optimizer.step()
        return {
            "reference_loss": ref_loss.detach()
        }

    def update_velocity(data):
        critic.eval()

        next_obs = data["next"]["observations"]
        with torch.no_grad():
            action_init, _ = reference.get_actions_and_log_probs(next_obs)

        # Bounded Q-gradient ascent in action space builds the flow target.
        y = action_init.clone()
        for _ in range(cfg.grad_step_num):
            y = y.detach().requires_grad_(True)
            q = critic.q_value(next_obs, y)
            grad_y = torch.autograd.grad(q.sum(), y)[0]
            grad_norm = grad_y.norm(dim=1, keepdim=True)
            step = torch.minimum(
                torch.full_like(grad_norm, cfg.grad_step_size),
                max_update / (grad_norm + 1e-6),
            )
            y = (y + step * grad_y).detach()

        action_flow_update = y

        # Conditional flow matching toward the straight line init -> updated
        t = torch.rand(next_obs.shape[0], 1, device=next_obs.device)
        temp_action = (1.0 - t) * action_init + t * action_flow_update
        temp_velocity = velocity_field(next_obs, temp_action, t)
        target_velocity = action_flow_update - action_init
        vel_loss = ((temp_velocity - target_velocity) ** 2).mean()

        vel_optimizer.zero_grad(set_to_none=True)
        vel_loss.backward()
        vel_optimizer.step()

        q_init = critic.q_value(next_obs, action_init).detach()
        q_flow = critic.q_value(next_obs, action_flow_update).detach()
        q_flow_diff = q_flow - q_init
        return {
            "velocity_loss": vel_loss.detach(),
            "q_init": q_init.mean(),
            "q_flow_update": q_flow.mean(),
            "q_flow_diff": q_flow_diff.mean(),
            "target_velocity_norm": target_velocity.norm(dim=1).mean().detach(),
        }

    # -------------------------------------------------------------- training
    obs = envs.reset()
    global_step = 0
    pbar = tqdm.tqdm(total=cfg.num_learning_iterations, initial=global_step)
    while global_step < cfg.num_learning_iterations:
        # --- Collection
        with torch.no_grad():
            norm_obs = obs_normalizer(obs, update=False)
            if global_step < cfg.learning_starts:
                actions = torch.rand((cfg.num_envs, n_act), device=device) * 2.0 - 1.0
            else:
                actor.eval()
                actions = actor.explore(norm_obs)

        next_obs, rewards, terminated, truncations, info = envs.step(actions)
        collect_experience(
            rb=rb, obs=obs, actions=actions, next_obs=next_obs, rewards=rewards,
            terminated=terminated, truncations=truncations, info=info,
        )
        obs = next_obs

        # --- Training
        batch_size = max(cfg.batch_size // cfg.num_envs, 1)
        if rb.ptr >= cfg.learning_starts:
            prepared_batches = sample_and_prepare_batches(
                rb=rb, obs_normalizer=obs_normalizer,
                num_updates=cfg.num_updates, target_batch_size=batch_size
            )
            for i, data in enumerate(prepared_batches):
                c_logs = update_critic(data)
                if cfg.num_updates > 1:
                    if i % cfg.policy_frequency == 1:
                        r_logs = update_reference(data)
                        v_logs = update_velocity(data)
                elif global_step % cfg.policy_frequency == 0:
                    r_logs = update_reference(data)
                    v_logs = update_velocity(data)

        global_step += 1
        pbar.update(1)
