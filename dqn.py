import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from FootballGame import FootballGameEnv
import csv
import os
from datetime import datetime
import pickle
from torch.nn import functional as F


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.buffer = pickle.load(f)


class DQNAgent:
    def __init__(self, state_shape, n_actions, device="cuda"):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.device = device

        self.policy_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = ReplayBuffer(100_000)

        self.batch_size = 128
        self.gamma = 0.99
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.target_update = 500
        self.epsilon = self.eps_start

        self.best_reward = float("-inf")
        self.best_policy = None

    def select_action(self, state, legal_actions):
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                mask = torch.tensor(
                    [
                        1 if i in legal_actions else float("-inf")
                        for i in range(self.n_actions)
                    ],
                    device=self.device,
                )
                q_values = q_values + mask
                return q_values.max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[random.choice(legal_actions)]], device=self.device, dtype=torch.long
            )

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.cat(batch[0]).to(self.device)
        action_batch = torch.cat(batch[1]).to(self.device)
        reward_batch = torch.cat(batch[2]).to(self.device)
        next_state_batch = torch.cat(batch[3]).to(self.device)
        done_batch = torch.tensor(batch[4], dtype=torch.bool).to(self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_actions = (
                self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            )
            next_state_values = (
                self.target_net(next_state_batch)
                .gather(1, next_state_actions)
                .squeeze(1)
            )
            next_state_values = next_state_values * (~done_batch)

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_best_policy(self, avg_reward):
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.best_policy = self.policy_net.state_dict()


def log_results(log_file, episode, agent1, agent2, agent1_stats, agent2_stats):
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                episode,
                agent1.epsilon,
                agent2.epsilon,
                agent1_stats["avg_loss"],
                agent2_stats["avg_loss"],
                agent1_stats["avg_reward"],
                agent2_stats["avg_reward"],
                agent1_stats["win_rate"],
                agent2_stats["win_rate"],
            ]
        )


def create_log_file(log_dir="logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Episode",
                "Agent1_Epsilon",
                "Agent2_Epsilon",
                "Agent1_Avg_Loss",
                "Agent2_Avg_Loss",
                "Agent1_Avg_Reward",
                "Agent2_Avg_Reward",
                "Agent1_Win_Rate",
                "Agent2_Win_Rate",
            ]
        )
    return log_file


def self_play_training(
    env, n_episodes=10000, max_steps=200, log_interval=200, render=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    print(f"State shape: {state_shape}")
    print(f"Number of actions: {n_actions}")

    agent1 = DQNAgent(state_shape, n_actions, device)
    agent2 = DQNAgent(state_shape, n_actions, device)

    log_file = create_log_file()

    agent1_stats = {"total_reward": 0, "total_loss": 0, "wins": 0, "games": 0}
    agent2_stats = {"total_reward": 0, "total_loss": 0, "wins": 0, "games": 0}

    for episode in range(n_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward1 = 0
        episode_reward2 = 0

        for step in range(max_steps):
            current_agent = agent1 if env.turn == 1 else agent2
            legal_actions = env.get_legal_actions()
            action = current_agent.select_action(state, legal_actions)
            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.tensor(
                next_state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            done = torch.tensor([done], device=device, dtype=torch.bool)

            if env.turn == 1:
                episode_reward1 += reward.item()
            else:
                episode_reward2 += reward.item()

            current_agent.memory.push(state, action, reward, next_state, done)
            state = next_state

            loss = current_agent.optimize_model()

            if loss is not None:
                if env.turn == 1:
                    agent1_stats["total_loss"] += loss
                else:
                    agent2_stats["total_loss"] += loss

            if done:
                if env.winner == 1:
                    agent1_stats["wins"] += 1
                elif env.winner == 2:
                    agent2_stats["wins"] += 1
                break

        agent1_stats["total_reward"] += episode_reward1
        agent2_stats["total_reward"] += episode_reward2
        agent1_stats["games"] += 1
        agent2_stats["games"] += 1

        agent1.update_epsilon()
        agent2.update_epsilon()

        if episode % agent1.target_update == 0:
            agent1.update_target_network()
            agent2.update_target_network()

        if (episode + 1) % log_interval == 0:
            agent1_avg_loss = (
                agent1_stats["total_loss"] / log_interval if log_interval > 0 else 0
            )
            agent2_avg_loss = (
                agent2_stats["total_loss"] / log_interval if log_interval > 0 else 0
            )
            agent1_avg_reward = (
                agent1_stats["total_reward"] / log_interval if log_interval > 0 else 0
            )
            agent2_avg_reward = (
                agent2_stats["total_reward"] / log_interval if log_interval > 0 else 0
            )
            agent1_win_rate = (
                agent1_stats["wins"] / agent1_stats["games"]
                if agent1_stats["games"] > 0
                else 0
            )
            agent2_win_rate = (
                agent2_stats["wins"] / agent2_stats["games"]
                if agent2_stats["games"] > 0
                else 0
            )

            log_results(
                log_file,
                episode + 1,
                agent1,
                agent2,
                {
                    "avg_loss": agent1_avg_loss,
                    "avg_reward": agent1_avg_reward,
                    "win_rate": agent1_win_rate,
                },
                {
                    "avg_loss": agent2_avg_loss,
                    "avg_reward": agent2_avg_reward,
                    "win_rate": agent2_win_rate,
                },
            )

            print(f"Episode {episode + 1}")
            print(
                f"Agent 1 - Epsilon: {agent1.epsilon:.2f}, Avg Loss: {agent1_avg_loss:.4f}, Avg Reward: {agent1_avg_reward:.2f}, Win Rate: {agent1_win_rate:.2f}"
            )
            print(
                f"Agent 2 - Epsilon: {agent2.epsilon:.2f}, Avg Loss: {agent2_avg_loss:.4f}, Avg Reward: {agent2_avg_reward:.2f}, Win Rate: {agent2_win_rate:.2f}"
            )
            print()

            agent1.save_best_policy(agent1_avg_reward)
            agent2.save_best_policy(agent2_avg_reward)

            agent1_stats = {"total_reward": 0, "total_loss": 0, "wins": 0, "games": 0}
            agent2_stats = {"total_reward": 0, "total_loss": 0, "wins": 0, "games": 0}

    return agent1, agent2


def main():
    print("Starting main function...")

    env = FootballGameEnv(height=13, width=13)

    buffer = ReplayBuffer(capacity=1_000_000)

    print("Simulating initial experiences...")
    state = env.reset()
    for _ in range(500_000):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)

        if done:
            state = env.reset()
        else:
            state = next_state

    print(f"Initial buffer size: {len(buffer)}")

    buffer.save("initial_replay_buffer.pkl")
    print("Saved initial replay buffer.")

    print("Starting self-play training...")
    trained_agent1, trained_agent2 = self_play_training(
        env, n_episodes=1500, max_steps=300, log_interval=50, render=False
    )

    torch.save(trained_agent1.policy_net.state_dict(), "trained_agent1.pth")
    torch.save(trained_agent2.policy_net.state_dict(), "trained_agent2.pth")
    print("Training completed and models saved.")

    batch_size = 128
    samples = trained_agent1.memory.sample(batch_size)

    states, actions, rewards, next_states, dones = zip(*samples)

    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.stack(next_states)
    dones = torch.tensor(dones, dtype=torch.bool)

    print("\nFinal buffer statistics:")
    print(f"Sampled batch size: {len(states)}")
    print(f"State shape: {states.shape}")
    print(f"Action shape: {actions.shape}")
    print(f"Reward shape: {rewards.shape}")
    print(f"Next state shape: {next_states.shape}")
    print(f"Done shape: {dones.shape}")


if __name__ == "__main__":
    main()
