import torch
import torch.nn as nn

from tensordict import TensorDict
from typing import Callable


class SimpleReplayBuffer(nn.Module):
    def __init__(
            self,
            n_env: int,
            buffer_size: int,
            n_obs: int,
            n_act: int,
            n_steps: int,
            gamma: float,
            device: torch.device
    ):
        super().__init__()

        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device

        self.observations = torch.zeros((n_env, buffer_size, n_obs), device=device, dtype=torch.float)
        self.actions = torch.zeros((n_env, buffer_size, n_act), device=device, dtype=torch.float)
        self.rewards = torch.zeros((n_env, buffer_size), device=device, dtype=torch.float)
        self.dones = torch.zeros((n_env, buffer_size), device=device, dtype=torch.long)
        self.truncations = torch.zeros((n_env, buffer_size), device=device, dtype=torch.long)
        self.next_observations = torch.zeros((n_env, buffer_size, n_obs), device=device, dtype=torch.float)
        self.ptr = 0

    @torch.no_grad()
    def extend(self, tensor_dict: TensorDict):
        observations = tensor_dict["observations"]
        actions = tensor_dict["actions"]
        rewards = tensor_dict["next"]["rewards"]
        dones = tensor_dict["next"]["dones"]
        truncations = tensor_dict["next"]["truncations"]
        next_observations = tensor_dict["next"]["observations"]

        ptr = self.ptr % self.buffer_size
        self.observations[:, ptr] = observations
        self.actions[:, ptr] = actions
        self.rewards[:, ptr] = rewards
        self.dones[:, ptr] = dones
        self.truncations[:, ptr] = truncations
        self.next_observations[:, ptr] = next_observations
        self.ptr += 1

    @torch.no_grad()
    def sample(self, batch_size: int):
        # we will sample n_env * batch_size transitions

        if self.n_steps == 1:
            indices = torch.randint(
                0,
                min(self.buffer_size, self.ptr),
                (self.n_env, batch_size),
                device=self.device,
            )
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
            observations = torch.gather(self.observations, 1, obs_indices).reshape(self.n_env * batch_size, self.n_obs)
            next_observations = torch.gather(self.next_observations, 1, obs_indices).reshape(self.n_env * batch_size,
                                                                                             self.n_obs)
            actions = torch.gather(self.actions, 1, act_indices).reshape(self.n_env * batch_size, self.n_act)
            rewards = torch.gather(self.rewards, 1, indices).reshape(self.n_env * batch_size)
            dones = torch.gather(self.dones, 1, indices).reshape(self.n_env * batch_size)
            truncations = torch.gather(self.truncations, 1, indices).reshape(self.n_env * batch_size)
            effective_n_steps = torch.ones_like(dones)
        else:
            # Sample base indices
            if self.ptr >= self.buffer_size:
                # When the buffer is full, there is no protection against
                # sampling across different episodes. We avoid this by
                # temporarily setting self.pos - 1 to truncated = True if not done
                current_pos = self.ptr % self.buffer_size
                curr_truncations = self.truncations[:, current_pos - 1].clone()
                self.truncations[:, current_pos - 1] = torch.logical_not(self.dones[:, current_pos - 1])
                indices = torch.randint(
                    0,
                    self.buffer_size,
                    (self.n_env, batch_size),
                    device=self.device,
                )
            else:
                # Buffer not full - ensure n-step sequence doesn't exceed valid data
                max_start_idx = max(1, self.ptr - self.n_steps + 1)
                indices = torch.randint(
                    0,
                    max_start_idx,
                    (self.n_env, batch_size),
                    device=self.device,
                )
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

            # Get base transitions
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )
            # Create sequential indices for each sample
            # This creates a [n_env, batch_size, n_step] tensor of indices
            seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
            all_indices = (
                                  indices.unsqueeze(-1) + seq_offsets
                          ) % self.buffer_size  # [n_env, batch_size, n_step]

            # Gather all rewards and terminal flags
            # Using advanced indexing - result shapes: [n_env, batch_size, n_step]
            all_rewards = torch.gather(
                self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_dones = torch.gather(
                self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_truncations = torch.gather(
                self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps),
                1,
                all_indices,
            )

            # Create masks for rewards *after* first done
            # This creates a cumulative product that zeroes out rewards after the first done
            all_dones_shifted = torch.cat(
                [torch.zeros_like(all_dones[:, :, :1]), all_dones[:, :, :-1]], dim=2
            )  # First reward should not be masked
            done_masks = torch.cumprod(
                1.0 - all_dones_shifted, dim=2
            )  # [n_env, batch_size, n_step]
            effective_n_steps = done_masks.sum(2)

            # Create discount factors
            discounts = torch.pow(
                self.gamma, torch.arange(self.n_steps, device=self.device)
            )  # [n_steps]

            # Apply masks and discounts to rewards
            masked_rewards = all_rewards * done_masks  # [n_env, batch_size, n_step]
            discounted_rewards = masked_rewards * discounts.view(
                1, 1, -1
            )  # [n_env, batch_size, n_step]

            # Sum rewards along the n_step dimension
            n_step_rewards = discounted_rewards.sum(dim=2)  # [n_env, batch_size]

            # Find index of first done or truncation or last step for each sequence
            first_done = torch.argmax(
                (all_dones > 0).float(), dim=2
            )  # [n_env, batch_size]
            first_trunc = torch.argmax(
                (all_truncations > 0).float(), dim=2
            )  # [n_env, batch_size]

            # Handle case where there are no dones or truncations
            no_dones = all_dones.sum(dim=2) == 0
            no_truncs = all_truncations.sum(dim=2) == 0

            # When no dones or truncs, use the last index
            first_done = torch.where(no_dones, self.n_steps - 1, first_done)
            first_trunc = torch.where(no_truncs, self.n_steps - 1, first_trunc)

            # Take the minimum (first) of done or truncation
            final_indices = torch.minimum(
                first_done, first_trunc
            )  # [n_env, batch_size]

            # Create indices to gather the final next observations
            final_next_obs_indices = torch.gather(
                all_indices, 2, final_indices.unsqueeze(-1)
            ).squeeze(-1)  # [n_env, batch_size]

            # Gather final values
            final_next_observations = self.next_observations.gather(
                1, final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            )
            final_dones = self.dones.gather(1, final_next_obs_indices)
            final_truncations = self.truncations.gather(1, final_next_obs_indices)

            # Reshape everything to batch dimension
            rewards = n_step_rewards.reshape(self.n_env * batch_size)
            dones = final_dones.reshape(self.n_env * batch_size)
            truncations = final_truncations.reshape(self.n_env * batch_size)
            effective_n_steps = effective_n_steps.reshape(self.n_env * batch_size)
            next_observations = final_next_observations.reshape(
                self.n_env * batch_size, self.n_obs
            )

        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next": {
                    "rewards": rewards,
                    "dones": dones,
                    "truncations": truncations,
                    "observations": next_observations,
                    "effective_n_steps": effective_n_steps,
                },
            },
            batch_size=self.n_env * batch_size,
        )
        if self.n_steps > 1 and self.ptr >= self.buffer_size:
            # Roll back the truncation flags introduced for safe sampling
            self.truncations[:, current_pos - 1] = curr_truncations
        return out


def collect_experience(
        rb: SimpleReplayBuffer,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncations: torch.Tensor,
        info: dict,
):
    dones = (terminated + truncations).bool()

    # Compute 'true' next_obs for saving
    true_next_obs = torch.where(dones[:, None] > 0, info["final_observation"], next_obs)

    transition = TensorDict(
        {
            "observations": obs,
            "actions": torch.as_tensor(actions, device=rb.device, dtype=torch.float),
            "next": {
                "observations": true_next_obs,
                "rewards": torch.as_tensor(rewards, device=rb.device, dtype=torch.float),
                "truncations": truncations.long(),
                "dones": dones.long(),
            },
        },
        batch_size=(rb.n_env,),
        device=rb.device,
    )
    rb.extend(transition)
    return


def sample_and_prepare_batches(
        rb: SimpleReplayBuffer,
        obs_normalizer: Callable,
        num_updates: int,
        target_batch_size: int
) -> list[TensorDict]:
    """
    Sample a large batch once and split it into smaller batches for each update.
    This reduces sampling overhead by `num_updates` and normalization overhead by `num_updates`.
    """
    # Sample a large batch (batch_size * num_updates)
    large_batch_size = target_batch_size * num_updates
    large_data = rb.sample(large_batch_size)
    samples_per_update = target_batch_size * rb.n_env

    # Normalize all data once
    large_data["observations"] = obs_normalizer(large_data["observations"])
    large_data["next"]["observations"] = obs_normalizer(large_data["next"]["observations"])

    # Split into smaller batches
    prepared_batches = []

    for i in range(num_updates):
        start_idx = i * samples_per_update
        end_idx = (i + 1) * samples_per_update

        # Create a slice of the large batch
        batch_data = TensorDict(
            {
                "observations": large_data["observations"][start_idx:end_idx],
                "actions": large_data["actions"][start_idx:end_idx],
                "next": {
                    "rewards": large_data["next"]["rewards"][start_idx:end_idx],
                    "dones": large_data["next"]["dones"][start_idx:end_idx],
                    "truncations": large_data["next"]["truncations"][start_idx:end_idx],
                    "observations": large_data["next"]["observations"][start_idx:end_idx],
                    "effective_n_steps": large_data["next"]["effective_n_steps"][start_idx:end_idx],
                },
            },
            batch_size=samples_per_update,
        )
        prepared_batches.append(batch_data)
    return prepared_batches
