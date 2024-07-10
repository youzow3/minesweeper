
import random
from argparse import ArgumentParser
from icecream import ic

class BoardExplosionError(Exception):
    def __init__(self, index : tuple[int, int] = None):
        super().__init__("Mine was exploded!" if index is None else f"Mine at {index} was exploded!")

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
            output = output.center(n_center)

            if color:
                cfflag = 34
                cbflag = 47
                cfbomb = [37, 34, 36, 32, 33, 35, 31, 31, 30]
                cbbomb = [49, 49, 49, 49, 49, 49, 49, 40, 41]
                creset = "\x1b[0m"

                if self.is_opened:
                    output = f"\x1b[1;{cfbomb[self.n_bomb]};{cbbomb[self.n_bomb]}m" + output
                elif self.is_flag:
                    output = f"\x1b[1;{cfflag};{cbflag}m" + output
                output += creset
            if debug:
                output += f":{'B' if self.is_bomb else str(self.n_bomb)}"
            return output

    def __init__(self, size : tuple[int, int], n_bomb : int, safe_pos : tuple[int, int], debug = False, color = False):
        assert type(size) is tuple
        assert type(n_bomb) is int
        assert type(safe_pos) is tuple or safe_pos is None

        self.size = size
        self.size_pad = (len(str(size[0])), len(str(size[1])))
        self.n_bomb = n_bomb
        self.safe_pos = safe_pos
        self.debug = debug
        self.color = color
        self.bomb_opened = False

        self.bar = ' ' * self.size_pad[1] + ('+' + '-' * (self.size_pad[0] + (2 if self.debug else 0))) * self.size[0] + '+'
        
        bomb_map = list(range(size[0] * size[1]))
        random.shuffle(bomb_map)
        if safe_pos is not None:
            for y in range(-1, 2):
                for x in range(-1, 2):
                    if not self.is_valid_index((safe_pos[0] + x, safe_pos[1] + y)):
                        continue
                    bomb_map[self.get_index((safe_pos[0] + x, safe_pos[1] + y))] = -1
        bomb_indexes = list(reversed(sorted(set(bomb_map))))[:n_bomb]

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

    def get_position(self, i : int):
        assert type(i) == int
        return (i % self.size[0], i // self.size[0])

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
                print('|' + self[(x, y)].output(n_center = self.size_pad[0], debug = self.debug, color = self.color), end = "")
            print('|')
        print(self.bar)

    def open(self, index : tuple[int, int]) -> int:
        if not self.is_valid_index(index):
            return 0
        elif self[index].is_opened:
            return 0
        elif self[index].is_bomb:
            self.bomb_opened = True
            return 0
        elif self[index].n_bomb > 0:
            self[index].is_opened = True
            return 1
        else:
            n_opened = 0
            self[index].is_opened = True
            for xi in range(-1, 2):
                for yi in range(-1, 2):
                    n_opened += self.open((index[0] + xi, index[1] + yi))
            return n_opened
    
    def flag(self, index : tuple[int, int]) -> bool:
        if not self.is_valid_index(index):
            return False

        self[index].is_flag = not self[index].is_flag

    def clean(self):
        non_bomb = 0
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                state = self[(x, y)]
                if state.is_bomb and not state.is_flag:
                    raise BoardExplosionError((x, y))
                if not state.is_bomb and state.is_flag:
                    non_bomb += 1
        if non_bomb > 0:
            print(f"You put {non_bomb} flag(s) at non-bomb position")

    def check(self) -> bool:
        for s in self.board:
            if not s.is_bomb and not s.is_opened:
                return False
        return True

def main(args):
    size_str = args.size.split(',')
    size = (int(size_str[0]), int(size_str[1]))
    n_bomb = args.bomb
    if args.position is not None:
        safe_pos_str = args.position.split(',')
        safe_pos = (int(safe_pos_str[0]), int(safe_pos_str[1]))
    else:
        safe_pos = None
    debug = args.debug
    if debug:
        ic.enable()
    else:
        ic.disable()
    color = args.color

    ic(size)
    ic(n_bomb)
    ic(safe_pos)
    ic(debug)
    board = Board(size, n_bomb, safe_pos, debug = debug, color = color) 
    while True:
        board.print()
        cmd = input("Enter the command (empty to show help):").strip().split(' ')
        if cmd[0] == "" or cmd[0] == 'h':
            print("Command List")
            print("h           Show this help")
            print("o [x] [y]   Open the specified position")
            print("f [x] [y]   Put the flag to the specified position")
            print("c           Clean bombs")
            print("q           Quit game")
        elif cmd[0] == 'o' or cmd[0] == 'f':
            try:
                p = (int(cmd[1]), int(cmd[2]))
            except:
                print("Failed to convert value. Please do again.")
                continue
            if not board.is_valid_index(p):
                print("The position {p} is not valid")
                continue
            if cmd[0] == 'o':
                board.open(p)
            elif cmd[0] == 'f':
                board.flag(p)
        elif cmd[0] == 'c':
            board.clean()
        elif cmd[0] == 'q':
            i = input("Do you really quit game? [y/N] ")
            if i.lower() == 'y':
                break

        if board.check():
            print("Clear!")
            break
        elif board.bomb_opened:
            print("Mine was exploded!!")
            break

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--size", default = "10, 10", help = "The board size")
    parser.add_argument("-b", "--bomb", default = 10, help = "Number of bombs", type = int)
    parser.add_argument("-p", "--position", default = None, help = "Safe position (Bombs won't be placed 3x3)")
    parser.add_argument("-d", "--debug", action = "store_true", help = "Start with debug mode")
    parser.add_argument("-c", "--color", action = "store_true", help = "Use colored output if possible")
    main(parser.parse_args())
