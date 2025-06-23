import torch
from model import model

from editor import Editor
from generator import Generator

while True:
    stop_marker = None
    try:
        action = int(input("Press 0 for save and quit. Press 1 for generate new image. Press 2 for edit image: "))
        if action == 1:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            folder = "/Users/maxkucher/face_generator/generations"
            size = (256, 256)

            generator = Generator(device, model, folder, size)
            generator()

        elif action == 2:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dir_to_save = "/Users/maxkucher/face_generator/edits"
            photos_dir = "/Users/maxkucher/face_generator/photos"
            editor = Editor(device, dir_to_save, photos_dir)
            editor()

        elif action == 0:
            stop_marker = True
            break


    except ValueError:
        print("Please enter an integer - 1 or 2.")

    if stop_marker == True:
        break


