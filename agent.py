
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import minesweeper as env

from argparse import ArgumentParser
from icecream import ic

class AgentPrototype:
    class NN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding = "same")
            self.conv2 = nn.Conv2d(8, 16, 3, padding = "same")
            self.conv3 = nn.Conv2d(16, 32, 3, padding = "same")
            self.tconv1 = nn.ConvTranspose2d(32, 32, 2, 2)
            self.tconv2 = nn.ConvTranspose2d(16, 16, 2, 2)
            self.tconv3 = nn.ConvTranspose2d(8, 8, 2, 2)
            self.conv4 = nn.Conv2d(32, 16, 3, padding = "same")
            self.conv5 = nn.Conv2d(16, 8, 3, padding = "same")
            self.conv6 = nn.Conv2d(8, 3, 3, padding = "same")
            self.output_conv = nn.Conv2d(3, 1, 1, padding = "same")

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)

            x = self.tconv1(x)
            x = F.relu(self.conv4(x))
            x = self.tconv2(x)
            x = F.relu(self.conv5(x))
            x = self.tconv3(x)
            x = F.relu(self.conv6(x))

            return self.output_conv(x)

    class NN2(nn.Module):
        def __init__(self):
            super().__init__()
            filters = [2, 128, 256, 512, 1024]
            self.conv = nn.Sequential()
            for i in range(len(filters) - 1):
                self.conv.append(nn.Conv2d(filters[i], filters[i + 1], 3, padding = "same"))
                self.conv.append(nn.ReLU())
            self.output = nn.Conv2d(filters[-1], 1, 3, padding = "same")

        def forward(self, x):
            x = x[1:, :, :]
            x = self.conv(x)
            o = self.output(x)
            return o
                
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn = self.NN2().to(self.device)
        self.optimizer = torch.optim.SGD(self.nn.parameters())
        self.last_reward = 0
        self.discount = 0.7

    def action(self, state : list, reward_func):
        flag_tensor = torch.tensor([int(s.is_flag) for s in state], dtype = torch.float).reshape((10, 10))
        opened_tensor = torch.tensor([int(s.is_opened) for s in state], dtype = torch.float).reshape((10, 10))
        n_bomb_tensor = torch.tensor([s.n_bomb if s.is_opened else 0 for s in state], dtype = torch.float).reshape((10, 10))

        x = torch.stack((flag_tensor, opened_tensor, n_bomb_tensor), dim = 0).to(self.device)
        y = self.nn(x)
        a = torch.multinomial(F.softmax(y.reshape(-1), -1), 1)[0]
        r = reward_func(a.item())
        self.last_reward = reward_func(a.item()) + self.discount * r
        loss = F.mse_loss(y.reshape(-1)[a], torch.tensor(self.last_reward, dtype = torch.float, device = self.device))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

def main_prototype(args):
    agent = AgentPrototype()

    def reward(i : int):
        n = board.open(board.get_position(i))
        if board.bomb_opened:
            n = -5
        elif n == 0:
            n = -0.5
        elif n == 1:
            n = 3
        else:
            n = 5
        return n

    n_clear = 0
    n_step = 0
    c_loss = 0
    total_step = 0
    total_clear = 0
    total_loss = 0

    result_list = []
    result_split = args.split 
    size_str = args.size.split(',')
    size = (int(size_str[0]), int(size_str[1]))

    epoch = args.epoch
    for i in range(epoch):
        n = 0
        loss = 0
        board = env.Board(size, args.bomb, None, debug = False, color = True)
        
        agent.last_reward = 0
        while True:
            n += 1
            if n == 100:
                print(f"[{i}] Too much step. Play again.")
                break
            loss += agent.action(board.board, reward)
            if board.check():
                print(f"[{i}] Agent cleared all mines. {n} step spent.")
                n_clear += 1
                break
            elif board.bomb_opened:
                print(f"[{i}] Agent failed to clear mines. {n} step spent.")
                break

        n_step += n
        c_loss += loss / n
        board.print()
        if (i + 1) % result_split == 0:
            result_list.append({"clear": n_clear, "step": n_step / result_split, "loss": c_loss / result_split})
            total_step += n_step
            total_clear += n_clear
            total_loss += c_loss
            n_step = 0
            n_clear = 0
            c_loss = 0
    ic(total_clear)
    ic(total_step / epoch)
    ic(total_loss / (epoch / result_split))
    ic(result_list)

class AgentBase:
    def __init__(self, discount = 0.9):
        self.memory = []
        self.discount = discount
        self.last_reward = 0

    def action(self, state : env.Board):
        pass

    def remember(self, board : env.Board, action_index : int, reward : float):
        state_tensor = self.make_state_tensor(board)
        reward_tensor = torch.tensor(reward, dtype = torch.float) + self.discount * self.last_reward
        self.last_reward = reward_tensor
        self.memory.append([state_tensor, action_index, reward_tensor])

    def train(self):
        pass

    def reset(self, memory = True):
        if memory:
            self.memory.clear()
        self.last_reward = 0

    def make_state_tensor(self, board : env.Board):
        flag_tensor = torch.tensor([s.is_flag for s in board.board]).reshape(board.size)
        bomb_tensor = torch.tensor([s.n_bomb if s.is_opened else 0 for s in board.board]).reshape(board.size)
        return torch.stack([flag_tensor, bomb_tensor]).to(dtype = torch.float)
 
    def make_loader(self, batch_size = 1, shuffle = False):
        class __Dataset(torch.utils.data.Dataset):
            def __getitem__(__self, index):
                return self.memory[index]

            def __getitems__(__self, indexes):
                return [self.memory[index] for index in indexes]

            def __len__(__self):
                return len(self.memory)
        return torch.utils.data.DataLoader(__Dataset(), batch_size = batch_size, shuffle = shuffle)

class Agent(AgentBase):
    def __init__(self, board_size, discount, filters = [128, 256, 512, 1024]):
        super().__init__(discount)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ic(self.device)
        self.model = nn.Sequential()
        _filters = [1] + filters
        for i in range(len(_filters) - 1):
            self.model.append(nn.Conv2d(_filters[i], _filters[i + 1], 3, padding = "same"))
            self.model.append(nn.ReLU())
        self.model.append(nn.Conv2d(_filters[-1], 1, 1))
        self.model.append(nn.Flatten())
        self.model.append(nn.Softmax(dim = -1))
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters())

    def action(self, state : env.Board):
        s = super().make_state_tensor(state)[1:2, :, :].to(self.device)
        self.model.eval()
        with torch.no_grad():
            a = torch.multinomial(self.model(s)[0], 1)
        return a.item()

    def train(self):
        loss = 0
        self.optimizer.zero_grad()
        for x, a, r in self.memory:
            loss += r.to(self.device) * torch.log(self.model(x[1:2, :, :].to(self.device))[0][a])
        loss = -loss
        loss.backward()
        ic(loss)
        self.optimizer.step()

def main(args):
    agent = None
    size = args.size.split(',')
    size = (int(size[0]), int(size[1]))
    n_bomb = args.bomb
    step_max = size[0] * size[1]

    if args.version == 1:
        agent = Agent(size, args.discount)
    else:
        raise Exception("version is out of range")

    n_clear = 0
    n_fail = 0
    bad_selection = 0
    split_result = []

    ic(size)
    ic(n_bomb)
    ic(step_max)
    ic(args.discount)

    for i in range(args.epoch):
        board = env.Board(size, n_bomb, None, color = True)
        step = 0
        bad = 0
        agent.reset()
        while True:
            step += 1
            a = agent.action(board)
            n = board.open(board.get_position(a))

            if board.bomb_opened:
                reward = -math.log(step)
                n_fail += 1
            elif board.check():
                reward = math.log(step_max - step) 
                n_clear += 1
            elif n == 0:
                bad += 1
                if bad == step_max:
                    reward = -math.log(step_max)
                    step = step_max
                    bad_selection += 1
                else:
                    continue
            else:
                reward = 1

            agent.remember(board, a, float(reward))
            
            if board.bomb_opened or board.check():
                board.print()
                break
            elif step == step_max:
                board.print()
                print("Too many steps")
                break
        if (i + 1) % args.split == 0:
            print(f"{i} n_clear/n_fail={n_clear / n_fail * 100:>.4f}%")
            split_result.append(n_clear / n_fail)
        
        agent.train()

    ic(n_clear, n_fail, bad_selection)
    ic(f"Last Result n_clear/n_fail={n_clear / n_fail * 100}%")
    ic(split_result)
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--epoch", default = 1000, help = "Number of epoch", type = int)
    parser.add_argument("--split", default = 100, help = "Steps before training", type = int)
    parser.add_argument("-s", "--size", default = "10,10", help = "Board size (x, y)")
    parser.add_argument("-b", "--bomb", default = 10, help = "Number of bombs", type = int)
    parser.add_argument("-v", "--version", default = 0, help = "Agent version", type = int)
    parser.add_argument("-d", "--discount", default = 0.9, help = "Discount rate", type = float)
    args = parser.parse_args()
    if args.version == 0:
        main_prototype(args)
    else:
        main(args)
