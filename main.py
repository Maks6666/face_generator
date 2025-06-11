import torch
from model import model

from editor import Editor
from generator import Generator

while True:
    try:
        action = int(input("Press 1 for generate new image. Press 2 for edit image: "))
        if action == 1:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            folder = "/Users/maxkucher/opencv/face_generator/generations"
            size = (256, 256)

            generator = Generator(device, model, folder, size)
            generator()
            break
        elif action == 2:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dir_to_save = "/Users/maxkucher/opencv/face_generator/edits"
            photos_dir = "/Users/maxkucher/opencv/face_generator/photos"
            editor = Editor(device, dir_to_save, photos_dir)
            editor()
            break

    except ValueError:
        print("Please enter an integer - 1 or 2.")
