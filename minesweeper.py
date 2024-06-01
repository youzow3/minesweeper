
import random
from argparse import ArgumentParser
from icecream import ic

class BoardExplosionErrror(Exception):
    def __init__(self):
        super().__init__("Mine was exploded!")

class Board:
    class State:
        def __init__(self, is_bomb : bool):
            assert type(is_bomb) is bool

            self.is_bomb = is_bomb
            self.is_flag = False
            self.is_opened = False
            self.n_bomb = 0

        def output(self, n_center = 0, color = False, debug = False):
            output = str(self.n_bomb) if self.is_opened else 'F' if self.is_flag else ' '

            if color:
                n_left = max(0, (n_center - len(output)) // 2)
                n_right = max(0, n_center - n_left)
            else:
                output = output.center(n_center)
            if debug:
                output += f":{'B' if self.is_bomb else str(self.n_bomb)}"
            return output

    def __init__(self, size : tuple[int, int], n_bomb : int, safe_pos : tuple[int, int], debug = False):
        assert type(size) is tuple
        assert type(n_bomb) is int
        assert type(safe_pos) is tuple

        self.size = size
        self.size_pad = (len(str(size[0])), len(str(size[1])))
        self.n_bomb = n_bomb
        self.safe_pos = safe_pos
        self.debug = debug

        self.bar = ' ' * self.size_pad[1] + ('+' + '-' * (self.size_pad[0] + (2 if self.debug else 0))) * self.size[0] + '+'
        
        bomb_map = list(range(size[0] * size[1]))
        random.shuffle(bomb_map)
        bomb_indexes = list(reversed(list(range(size[0] * size[1]))))[:self.n_bomb]
        ic(bomb_map)
        ic(bomb_indexes)
        for y in range(-1, 2):
            for x in range(-1, 2):
                if not self.is_valid_index((x, y)):
                    continue
                bomb_map[self.get_index((x, y))] = -1

        self.board = [self.State(bomb_map[i] in bomb_indexes) for i in range(size[0] * size[1])]
        for y in range(size[1]):
            for x in range(size[0]):
                cur = (x, y)
                for yi in range(-1, 2):
                    for xi in range(-1, 2):
                        i = (x + xi, y + yi)
                        if not self.is_valid_index(i):
                            continue

                        self[cur].n_bomb += self[i].is_bomb

    def __getitem__(self, index : tuple[int, int]):
        assert type(index) is tuple
        return self.board[self.get_index(index)]

    def __setitem__(self, index : tuple[int, int], state : State):
        assert type(index) is tuple
        assert type(state) is self.State

        self.board[self.get_index(index)] = state

    def is_valid_index(self, index : tuple[int, int]):
        assert type(index) is tuple
        return (0 <= index[0] and index[0] < self.size[0]) and (0 <= index[1] and index[1] < self.size[1])

    def get_index(self, index : tuple[int, int]):
        if not self.is_valid_index(index):
            raise BoardIndexError(index)
        return index[0] + index[1] * self.size[0]

    def print(self):
        for y in range(self.size[1]):
            if y == 0:
                print(' ' * self.size_pad[1], end = "")
                for x in range(self.size[0]):
                    print(' ' * (1 + (2 if self.debug else 0)) + f"{x}".center(self.size_pad[0]), end = "")
                print()
            print(self.bar)
            for x in range(self.size[0]):
                if x == 0:
                    print(f'{y}'.center(self.size_pad[1]), end = "")
                print('|' + self[(x, y)].output(n_center = self.size_pad[0], debug = self.debug), end = "")
            print('|')
        print(self.bar)

    def open(self, index : tuple[int, int]) -> bool:
        if not self.is_valid_index(index):
            return False
        elif self[index].is_opened:
            return False
        elif self[index].is_bomb:
            raise BoardExplosionError()
        elif self[index].n_bomb > 0:
            self.is_opened = True
        else:
            self[index].is_opened = True
            for xi in range(-1, 2):
                for yi in range(-1, 2):
                    self.open((index[0] + xi, index[1] + yi))
    
    def flag(self, index : tuple[int, int]) -> bool:
        if not self.is_valid_index(index):
            return False

        self[index].is_flag = not self[index].is_flag

def main():
    board = Board((10, 10), 3, (2, 2), debug = True) 
    board.print()

if __name__ == "__main__":
    main()
