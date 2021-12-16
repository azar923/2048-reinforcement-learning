from tkinter import *
from logic import *
from random import *
import threading

# Class that hosts a game grid, that implements the logic of what happens when a particular action is chosen by the bot


SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10


total_score = 0


class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)
        self.total_score = 0

        self.commands = {0: up, 1: down, 2: left, 3: right}

        self.init_matrix()

        self.illegal_moves = 0

    def gen(self):
        return randint(0, GRID_LEN - 1)

    def init_matrix(self):
        self.matrix = new_game(4)
        self.matrix = add_two(self.matrix)
        
    def key_down(self, action_number):
        if action_number in self.commands:
            self.matrix, done, new_adding = self.commands[action_number](self.matrix)

            self.total_score += new_adding

            finished = False

            reward = new_adding

            if done:
                self.matrix = add_two(self.matrix)

                done = False
                if game_state(self.matrix) == 'win':
                    finished = True

                if game_state(self.matrix) == 'lose':
                    finished = True


        new_state = self.matrix

        return new_state, reward, action_number, finished

    def generate_next(self):
        index = (self.gen(), self.gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (self.gen(), self.gen())
        self.matrix[index[0]][index[1]] = 2


