from typing import Callable, Optional, Sequence, Tuple, List
import torch
from torch import nn
import numpy as np
from cs285.infrastructure import pytorch_util as ptu


from cs285.agents.dqn_agent import DQNAgent


class AWACAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        temperature: float,
        **kwargs,
    ):
        super().__init__(observation_shape=observation_shape, num_actions=num_actions, **kwargs)

        self.actor = make_actor(observation_shape, num_actions)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.temperature = temperature

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """

        observation = ptu.from_numpy(np.asarray(observation))
        if observation.ndim == 1: 
            observation = observation[None] 
        # to tensor and append a new dimension

        action = self.actor(observation).sample()
        return ptu.to_numpy(action).squeeze(0).item() # squeeze the NONE dimension


    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        with torch.no_grad():
            # TODO(student): compute the actor distribution, then use it to compute E[Q(s, a)]
            next_qa_values = self.target_critic(next_observations) 
            next_actions = self.actor(next_observations).sample()

            # Use the actor to compute a critic backup
            next_qs = torch.gather(next_qa_values, 1, next_actions.unsqueeze(-1)).squeeze(-1)

            # TODO(student): Compute the TD target
            target_values = rewards + 0.99 * next_qs * (~dones) 
        
        # TODO(student): Compute Q(s, a) and loss similar to DQN
        qa_values = self.critic(observations)
        q_values = torch.gather(qa_values, 1, actions.unsqueeze(-1)).squeeze(-1)
        assert q_values.shape == target_values.shape

        loss = self.critic_loss(q_values, target_values) 

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): compute the advantage of the actions compared to E[Q(s, a)]
        with torch.no_grad():
            expect_action = self.actor(observations).sample() 
            qa_values = self.critic(observations) 
            x = torch.gather(qa_values, 1, actions.unsqueeze(-1)).squeeze(-1)
            y = torch.gather(qa_values, 1, expect_action.unsqueeze(-1)).squeeze(-1) 
            advantages = x - y
        return advantages

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        # TODO(student): update the actor using AWAC
        action_dist = self.actor(observations)
        advantages = self.compute_advantage(observations, actions, action_dist)
        loss = torch.neg((action_dist.log_prob(actions) * advantages.exp()).mean())

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()

    def update(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, dones: torch.Tensor, step: int):
        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the actor.
        actor_loss = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_loss

        return metrics
