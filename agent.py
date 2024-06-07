
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import minesweeper as env

from icecream import ic

class Agent:
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

def main():
    agent = Agent()

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
    result_split = 1000 

    epoch = 25000
    for i in range(epoch):
        n = 0
        loss = 0
        board = env.Board((10, 10), 11, None, debug = False, color = True)
        
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

if __name__ == "__main__":
    main()
