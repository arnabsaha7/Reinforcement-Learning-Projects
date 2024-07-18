import torch
import numpy as np
from agent_nn import AgentNN
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage


class Agent:
    def __init__(self, input_dims, num_actions):
        self.num_actions = num_actions
        self.learn_step_counter = 0

        # Hyperparameters
        self.lr = 0.00025
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999975
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.sync_network_rate = 10_000

        # Networks 
        self.online_network = AgentNN(input_dims, num_actions)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True)

        # Optimizer and Loss
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

        # Replay Buffer
        replay_buffer_capacity = 100_000
        storage = LazyMemmapStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)


    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        observation = torch.tensor(np.array(observation), dtype=torch.float32) \
                            .unsqueeze(0) \
                            .to(self.online_network.device)
        
        return self.online_network(observation).argmax().item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(TensorDict({
            "state": torch.tensor(np.array(state), dtype=torch.float32),
            "action": torch.tensor(action),
            "reward": torch.tensor(reward),
            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32),
            "done": torch.tensor(done)
        }, batch_size=[]
        ))

    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        self.sync_networks()
        self.optimizer.zero_grad()
        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)

        keys = ("state", "action", "reward", "next_states", "dones")

        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        predicted_q_values = self.online_network(states)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]

        target_q_values = self.target_network(next_states).max(dim=1)[0]
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1
        self.decay_epsilon()