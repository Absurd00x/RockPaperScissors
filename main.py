import tkinter as tk
import shelve
import myNN
from constants import *


class MyGUI:
    def __init__(self):
        # Loading root
        self.root = tk.Tk()
        self.config_root(ROOT_WIDTH, ROOT_HEIGHT, 'Rock paper scissors', False, False)

        # Importing images
        self.images = {'rock': tk.PhotoImage(file="RockSmall.gif"),
                       'paper': tk.PhotoImage(file="PaperSmall.gif"),
                       'scissors': tk.PhotoImage(file="ScissorsSmall.gif")}

        # Reflected images
        self.reflected_images = {'rock': tk.PhotoImage(file="RockSmallReflected.gif"),
                                 'paper': tk.PhotoImage(file="PaperSmallReflected.gif"),
                                 'scissors': tk.PhotoImage(file="ScissorsSmallReflected.gif")}

        #
        # Buttons
        #

        self.UI_buttons = {'Rock': tk.Button(self.root, text='Rock', command=lambda: self.play('rock')),
                           'Paper': tk.Button(self.root, text='Paper', command=lambda: self.play('paper')),
                           'Scissors': tk.Button(self.root, text='Scissors', command=lambda: self.play('scissors'))}

        self.config_buttons()

        #
        # Loading NN and statistics
        #

        try:
            with shelve.open(FILENAME) as file:
                self.nn = file['nn']
                self.nn_input = file['games']
                self.player_score = file['player_score']
                self.nn_score = file['nn_score']
        except:
            with shelve.open(FILENAME, 'c') as file:
                self.nn = myNN.NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)
                self.nn_input = [0.01 for _ in range(30)]

                self.player_score = 0
                self.nn_score = 0

                file['player_score'] = self.player_score
                file['nn_score'] = self.nn_score
                file['nn'] = self.nn
                file['games'] = self.nn_input

        #
        # Statistics
        #

        self.statistics_label = tk.Label(self.root, text=LABEL_TEAMPLATE.format(
            self.player_score, self.nn_score, self.compute_winrate()))

        #
        # Placing
        #

        self.place_buttons()

        self.create_images('rock', 'rock')

        self.statistics_label.place(relx=LABEL_RELX, rely=LABEL_RELY)

    #
    # Methods:
    #

    def config_root(self, width, height, title, resizable_width=True, resizable_height=True):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        self.root.geometry('{:d}x{:d}+{:d}+{:d}'.format(width, height, x, y))
        self.root.title(title)
        self.root.resizable(width=resizable_width, height=resizable_height)

    def config_buttons(self):
        for button in self.UI_buttons.values():
            button.config(width=BUTTON_WIDTH, height=BUTTON_HEIGHT, bd=3)

    def place_buttons(self):
        for button, third in zip(self.UI_buttons.values(), range(1, 4)):
            button.place(relx=(third / 3 - BUTTON_X_INDENT), rely=BUTTON_RELY)

    def compute_winrate(self):
        return self.player_score / max(1, (self.player_score + self.nn_score)) * 100

    def create_images(self, player, nn):
        self.current_player = self.create_canvas(player)
        self.current_nn = self.create_reflected_canvas(nn)

        self.current_player.place(relx=PLAYER_RELX, rely=PLAYER_RELY)
        self.current_nn.place(relx=NN_RELX, rely=NN_RELY)

    def create_canvas(self, argument):
        created = tk.Canvas(self.root, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
        created.create_image((IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2), image=self.images[argument])
        return created

    def create_reflected_canvas(self, argument):
        created = tk.Canvas(self.root, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
        created.create_image((IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2), image=self.reflected_images[argument])
        return created

    def play(self, player_answer):
        # this array displays what beats what
        # rock beats scissors then nn should say rock when it predicts scissors
        answer_output = {'rock': [0.01, 0.99, 0.01], 'paper': [0.01, 0.00, 0.99], 'scissors': [0.99, 0.01, 0.01]}
        output_answer = {0: 'rock', 1: 'paper', 2: 'scissors'}
        nn_answer = output_answer[self.nn.query(self.nn_input).argmax()]

        beats = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}

        if nn_answer != player_answer:
            # player beats nn
            if beats[player_answer] == nn_answer:
                self.player_score += 1
                self.statistics_label['text'] = LABEL_TEAMPLATE.format(
                    self.player_score, self.nn_score, self.compute_winrate())
            else:
                self.nn_score += 1
                self.statistics_label['text'] = LABEL_TEAMPLATE.format(
                    self.player_score, self.nn_score, self.compute_winrate())

        self.current_player.destroy()
        self.current_nn.destroy()

        self.create_images(player_answer, nn_answer)

        correct_answer = answer_output[player_answer]
        self.nn.train(self.nn_input, correct_answer)
        self.nn_input = self.nn_input[3:] + correct_answer


gui = MyGUI()

tk.mainloop()

with shelve.open(FILENAME, 'c') as file:
    file['nn'] = gui.nn
    file['games'] = gui.nn_input
    file['player_score'] = gui.player_score
    file['nn_score'] = gui.nn_score
