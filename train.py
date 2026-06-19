import copy
import math

import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from buffer import SimpleReplayBuffer, sample_and_prepare_batches, collect_experience
from hyperparams import QFlexConfig
from networks import Critic, ReferencePolicy, QFlexActor, VelocityField
from normalizers import EmpiricalNormalization


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
        num_atoms=cfg.num_atoms,
        v_min=cfg.v_min,
        v_max=cfg.v_max,
        hidden_dim=cfg.critic_hidden_dim,
        use_layer_norm=cfg.use_layer_norm,
        num_q_networks=cfg.num_q_networks,
        device=device,
    )
    critic_target = copy.deepcopy(critic).to(device)
    for p in critic_target.parameters():
        p.requires_grad_(False)

    reference = ReferencePolicy(
        n_obs=n_obs,
        n_act=n_act,
        hidden_dim=cfg.actor_hidden_dim,
        layer_norm=cfg.use_layer_norm,
        device=device
    )
    velocity_field = VelocityField(
        n_obs=n_obs,
        n_act=n_act,
        hidden_dim=cfg.velocity_hidden_dim,
        layer_norm=cfg.use_layer_norm,
        device=device
    )
    actor = QFlexActor(
        reference=reference,
        velocity_field=velocity_field,
        num_timesteps=cfg.num_flow_steps,
        device=device,
        n_act=n_act,
        num_envs=cfg.num_envs,
        std_min=cfg.std_min,
        std_max=cfg.std_max,
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
        list(reference.parameters()),
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
            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(cfg.policy_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            next_actions = reference(next_observations)
            next_actions = (next_actions + clipped_noise).clamp(action_low, action_high)

            discount = cfg.gamma ** data["next"]["effective_n_steps"]
            qf1_next_target_projected, qf2_next_target_projected = (
                critic_target.projection(
                    next_critic_observations,
                    next_actions,
                    rewards,
                    bootstrap,
                    discount,
                )
            )
            qf1_next_target_value = critic_target.get_value(qf1_next_target_projected)
            qf2_next_target_value = critic_target.get_value(qf2_next_target_projected)
            if cfg.use_cdq:
                qf_next_target_dist = torch.where(
                    qf1_next_target_value.unsqueeze(1)
                    < qf2_next_target_value.unsqueeze(1),
                    qf1_next_target_projected,
                    qf2_next_target_projected,
                )
                qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
            else:
                qf1_next_target_dist, qf2_next_target_dist = (
                    qf1_next_target_projected,
                    qf2_next_target_projected,
                )
        qf1, qf2 = critic(critic_observations, actions)
        qf1_loss = -torch.sum(qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1).mean()
        qf2_loss = -torch.sum(qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1).mean()
        q_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        q_optimizer.step()

        return {
            "q_loss": q_loss.detach(),
            "q_min": qf1_next_target_value.min().detach(),
            "q_max": qf1_next_target_value.max().detach(),
        }

    def update_reference(data):
        ref_actions = reference(data["observations"])
        critic_observations = data["observations"]
        q_value = critic.q_value(critic_observations, ref_actions, use_cdq=cfg.use_cdq)
        ref_loss = -q_value.mean()

        ref_optimizer.zero_grad(set_to_none=True)
        ref_loss.backward()
        ref_optimizer.step()
        return {
            "reference_loss": ref_loss.detach()
        }

    def update_velocity(data):
        critic_target.eval()

        next_obs = data["next"]["observations"]
        with torch.no_grad():
            action_init = reference(next_obs)
        q_init = critic_target.q_value(next_obs, action_init, cfg.use_cdq).detach()

        # Bounded Q-gradient ascent in action space builds the flow target.
        y = action_init.clone()
        for _ in range(cfg.grad_step_num):
            y = y.detach().requires_grad_(True)
            q = critic_target.q_value(next_obs, y, cfg.use_cdq)
            grad_y = torch.autograd.grad(q.sum(), y)[0]
            grad_norm = grad_y.norm(dim=1, keepdim=True)
            step = torch.minimum(
                torch.full_like(grad_norm, cfg.grad_step_size),
                max_update / (grad_norm + 1e-6),
            )
            y = torch.clamp(y + step * grad_y, action_low, action_high).detach()
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

        q_flow = critic_target.q_value(next_obs, action_flow_update, cfg.use_cdq).detach()
        return {
            "velocity_loss": vel_loss.detach(),
            "q_init": q_init.mean(),
            "q_flow_update": q_flow.mean(),
            "target_velocity_norm": target_velocity.norm(dim=1).mean().detach(),
        }

    @torch.no_grad()
    def update_target():
        src_ps = [p.data for p in critic.parameters()]
        tgt_ps = [p.data for p in critic_target.parameters()]
        torch._foreach_mul_(tgt_ps, 1.0 - cfg.tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=cfg.tau)
        return

    # -------------------------------------------------------------- training
    obs = envs.reset()
    dones = None
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
                actions = actor.explore(norm_obs, dones)

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
                update_target()

        global_step += 1
        pbar.update(1)
