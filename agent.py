
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
            self.conv1 = nn.Conv2d(3, 16, 3, padding = "same")
            self.conv2 = nn.Conv2d(16, 16, 3, padding = "same")
            self.conv3 = nn.Conv2d(16, 16, 3, padding = "same")
            self.tconv1 = nn.ConvTranspose2d(16, 16, 2, 2)
            self.tconv2 = nn.ConvTranspose2d(16, 16, 2, 2)
            self.tconv3 = nn.ConvTranspose2d(16, 16, 2, 2)
            self.conv4 = nn.Conv2d(16, 16, 3, padding = "same")
            self.conv5 = nn.Conv2d(16, 16, 3, padding = "same")
            self.conv6 = nn.Conv2d(16, 3, 3, padding = "same")
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

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn = self.NN().to(self.device)
        self.optimizer = torch.optim.SGD(self.nn.parameters())

    def action(self, state : list, reward_func):
        flag_tensor = torch.tensor([int(s.is_flag) for s in state], dtype = torch.float).reshape((10, 10))
        opened_tensor = torch.tensor([int(s.is_opened) for s in state], dtype = torch.float).reshape((10, 10))
        n_bomb_tensor = torch.tensor([s.n_bomb if s.is_opened else 0 for s in state], dtype = torch.float).reshape((10, 10))

        x = torch.stack((flag_tensor, opened_tensor, n_bomb_tensor), dim = 0).to(self.device)
        y = self.nn(x)
        a = torch.multinomial(F.softmax(y.reshape(-1), -1), 1)[0]
        r = reward_func(a.item())
        loss = F.mse_loss(y.reshape(-1)[a], torch.tensor(r, dtype = torch.float, device = self.device))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

def main():
    agent = Agent()

    def reward(i : int):
        n = board.open(board.get_position(i))
        return n

    n_clear = 0
    total_clear = 0
    result_list = []
    result_split = 100 
    for i in range(10000):
        n = 0
        board = env.Board((10, 10), 5, None, debug = False, color = True)
        
        while True:
            n += 1
            if n == 100:
                print(f"[{i}] Too much step. Play again.")
                break
            try:
                agent.action(board.board, reward)
                if board.check():
                    print(f"[{i}] Agent cleared all mines. {n} step spent.")
                    n_clear += 1
                    break
            except env.BoardExplosionError:
                print(f"[{i}] Agent failed to clear mines. {n} step spent.")
                break
        board.print()
        if (i + 1) % result_split == 0:
            result_list.append(n_clear)
            total_clear += n_clear
            n_clear = 0
    ic(n_clear)
    ic(result_list)

if __name__ == "__main__":
    main()
