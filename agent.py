import pickle
import time
import cv2 as cv
import os
import random
import argparse
from datetime import datetime, timedelta
import itertools
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch
from torch import nn

import gymnasium as gym
import miniworld

from dqn import DQCNN
from experience_replay import ReplayMemory


DATE_FORMAT = "%m-%d %H:%M:%S"
CHECKPOINT_FILE_NAME = "checkpoint.pth"

dir_path = os.path.dirname(__file__)

RUNS_DIR = os.path.join(dir_path, 'runs')

os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    def __init__(self, hyperparameter_set):
        file_path = os.path.join(dir_path, 'hyperparameters.yml')
        with open(file_path, 'r') as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc_nodes = hyperparameters['fc_nodes']
        self.input_channels = hyperparameters['input_channels']
        self.env_make_params = hyperparameters.get('env_make_params', {}) #Get Optional enviroment-specific parameters

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.policy_dqn = None
        self.target_dqn = None
        
        self.episode = 0
        self.replay_memory = None
        self.epsilon = 0
        self.step_counter = 0
        self.best_reward = -9999999
        self.epsilon_history = []
        self.rewards_per_episode = []

        # self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        # self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        # self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
        # será inserido dinamicamente
        self.MODEL_FILE = f'{self.hyperparameter_set}.pt'

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gym.make('MiniWorld-OneRoom-v0', render_mode="human" if render else None)

        n_actions = env.action_space.n

        self.policy_dqn = DQCNN(self.input_channels, n_actions, self.fc_nodes).to(device)

        if is_training:
            self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate_a)
            self.target_dqn = DQCNN(self.input_channels, n_actions, self.fc_nodes).to(device)
            self.replay_memory = ReplayMemory(self.replay_memory_size)

            if os.path.isfile(f"{RUNS_DIR}/{CHECKPOINT_FILE_NAME}"):
                
                print("Carregando o checkpoint:")
                self.load_checkpoint()
                print(f"Episodio: {self.episode}\nMemória: {len(self.replay_memory)}\nEpsilon: {self.epsilon}\nStep_counter: {self.step_counter}\nBest_reward:{self.best_reward}")
            else:
                print("Novo modelo")

                self.epsilon = self.epsilon_init

                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        else:
            self.load_model()
            print(f"Loading the model number {self.episode}")

            self.policy_dqn.eval()

        self.rewards_per_episode = []
        #roda indefinidamente
        # for episode in itertools.count():
        while self.episode < 9999999:

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            truncated = False
            episode_reward = 0.0

            while not terminated and not truncated and episode_reward < self.stop_on_reward:

                if is_training and random.random() < self.epsilon:
                    action = env.action_space.sample()  # agent policy that uses the observation and info
                    action = torch.tensor(action, dtype=torch.float, device=device)
                else:
                    with torch.no_grad():
                        # tensor([1,2,3,...]) ==> tensor([[1,2,3,...]])
                        action = self.policy_dqn(state.unsqueeze(1).permute(1, 3, 0, 2)).squeeze().argmax()

                new_state, reward, terminated, truncated, _ = env.step(int(action.item()))
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)


                if is_training:
                    self.replay_memory.append((state, action, new_state, reward, terminated))

                    self.step_counter += 1
                
                state = new_state
                if render:                
                    env.render()

            self.rewards_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > self.best_reward:
                    percentage_gain = f"{(episode_reward - self.best_reward)/self.best_reward*100:+.1f}" if self.best_reward != 0 else f"{100:+.1f}"
                    log_message = f'{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({percentage_gain}%) at episode {self.episode} saving model...'
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    
                        torch.save(self.policy_dqn.state_dict(), f"{RUNS_DIR}/{self.episode}")
                    self.best_reward = episode_reward

                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds = 10):
                    print(f"Recompensa do Episodio {self.episode + 1}: {episode_reward}")
                    print(f"Valor de Epsilon: {self.epsilon}")
                    self.save_graph()
                    last_graph_update_time = current_time
                    
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                self.epsilon_history.append(self.epsilon)


                if(self.episode % 100 == 0 and self.epsilon < 0.5):
                    self.save_checkpoint()

                if(len(self.replay_memory) > self.mini_batch_size):

                    mini_batch = self.replay_memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch)

                    #atualiza a target network depois de alguns passos
                    if(self.step_counter > self.network_sync_rate):
                        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                        self.step_counter=0

            self.episode += 1

    def optimize(self, mini_batch):
        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)
        
        # Reorganiza para [B, C, H, W]
        states = states.permute(0, 3, 1, 2)
        new_states = new_states.permute(0, 3, 1, 2)  

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
                # Calculate target Q values (expected returns)
                target_q = rewards + (1-terminations) * self.discount_factor_g * self.target_dqn(new_states).max(dim=1)[0]
                '''
                    self.target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                        .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                            [0]             ==> tensor([3,6])
                '''

        # Calcuate Q values from current policy
        current_q = self.policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1).long()).squeeze()
        '''
            self.policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases

    def save_graph(self):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(self.rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(self.rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(self.epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)
        
    def save_checkpoint(self, filename="checkpoint.pth"):
        checkpoint = {
            "episode": self.episode,
            "epsilon": self.epsilon,
            "best_reward": self.best_reward,
            "step_counter": self.step_counter,
            "model_state_dict": self.policy_dqn.state_dict(),
            "model_target_dict": self.target_dqn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, f"{RUNS_DIR}/{filename}")
        time.sleep(3)

        with open(os.path.join(RUNS_DIR, 'epsilon-array.pkl'), 'wb') as arquivo:
            pickle.dump(self.epsilon_history, arquivo)
            
        time.sleep(1)

        with open(os.path.join(RUNS_DIR, 'reward-array.pkl'), 'wb') as arquivo:
            pickle.dump(self.rewards_per_episode, arquivo)

        time.sleep(1)
        
        print(f"Checkpoint salvo!")

    def load_checkpoint(self, filename="checkpoint.pth"):
        checkpoint = torch.load(f"{RUNS_DIR}/{filename}")

        self.policy_dqn.load_state_dict(checkpoint["model_state_dict"])
        self.target_dqn.load_state_dict(checkpoint["model_target_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.episode = checkpoint["episode"]
        self.epsilon = checkpoint["epsilon"]
        self.best_reward = checkpoint["best_reward"]
        self.step_counter = checkpoint["step_counter"]

        with open(os.path.join(RUNS_DIR, 'epsilon-array.pkl'), 'rb') as arquivo:
            self.epsilon_history = pickle.load(arquivo)

        with open(os.path.join(RUNS_DIR, 'reward-array.pkl'), 'rb') as arquivo:
            self.rewards_per_episode = pickle.load(arquivo)

    def load_model(self, filename="checkpoint.pth"):
        checkpoint = torch.load(f"{RUNS_DIR}/{filename}")
        self.policy_dqn.load_state_dict(checkpoint["model_state_dict"])
        self.episode = checkpoint["episode"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training Mode', action='store_true')
    args = parser.parse_args()

    dq1 = Agent(hyperparameter_set=args.hyperparameters)
    
    if args.train:
        dq1.run(is_training=True)
    else:
        dq1.run(is_training=False, render=True)