from model import model
import torch
import time
import cv2
import numpy as np
import random
import os



class Editor:
    def __init__(self, device, dir_to_save, photos_dir):
        self.device = device

        self.dir_to_save = dir_to_save
        self.photos_dir = photos_dir

        self.dct = {}
        self.values = []
        self.atr_list = ["Five o'Clock Shadow", 'Arched Eyebrows', 'Attractive',
                         'Bags Under Eyes', 'Bald', 'Bangs', 'Big Lips', 'Big Nose',
                         'Black Hair', 'Blond Hair', 'Blurry', 'Brown Hair', 'Bushy Eyebrows',
                         'Chubby', 'Double Chin', 'Eyeglasses', 'Goatee', 'Gray Hair',
                         'Heavy Makeup', 'High Cheekbones', 'Male', 'Mouth Slightly Open',
                         'Mustache', 'Narrow Eyes', 'No Beard', 'Oval Face', 'Pale Skin',
                         'Pointy Nose', 'Receding Hairline', 'Rosy Cheeks', 'Sideburns',
                         'Smiling', 'Straight_Hair', 'Wavy Hair', 'Wearing Earrings',
                         'Wearing Hat', 'Wearing Lipstick', 'Wearing Necklace',
                         'Wearing Necktie', 'Young']

    def choose_photo(self):
        lst = os.listdir(self.photos_dir)
        print(lst)

        stop_marker = None
        while True:
            img = input("Input image name (only file name): ")
            if img in lst:
                full_path = os.path.join(self.photos_dir, img)
                break

            elif img == "q":
                stop_marker = True
                break

            else:
                print("This photo does not exist. Please, try one more time")

        if stop_marker is None:
            print("Accepted!")
            return full_path
        else:
            return None



    def prepare_image(self, link):
        img = cv2.imread(link)
        height, width = img.shape[:2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0

        img = torch.tensor(img, dtype=torch.float).to(self.device)
        img = img.permute(2, 0, 1)
        return img, height, width

    def input_features(self):
        stop_marker = None
        for obj in self.atr_list:
            while True:
                print("Please, input 1 for 'Yes' and 0 for 'No'")
                atr = input(f"Attribute {obj}: ")
                if atr in ['0', '1']:
                    self.dct[obj] = atr
                    break

                elif atr == "q":
                    stop_marker = True
                    break

                else:
                    print("Please, input 1 for 'Yes' and 0 for 'No'")


            if stop_marker == True:
                break

        if stop_marker is None:
            for value in self.dct.values():
                self.values.append(int(value))

            y = torch.tensor(self.values, dtype=torch.float)
            y = y.unsqueeze(0).to(self.device)
            return y

        else:
            return None


    def prepare_result(self, res, height, width):
        res = (res * 255).astype(np.uint8)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        res = cv2.resize(res, (height, width))
        return res


    def save_result(self, res):
        num = random.randint(0, 1000)
        path = os.path.join(self.dir_to_save, f"{num}.jpeg")
        cv2.imwrite(path, res)

    def __call__(self):
        while True:
            link = self.choose_photo()
            if link is None:
                print("Program ended.")
                break

            x, height, width = self.prepare_image(link)

            y = self.input_features()
            if y is None:
                print("Program ended.")
                break

            t_res = model.predict(x, y)
            res = self.prepare_result(t_res, width, height)
            self.save_result(res)



# device = "mps" if torch.backends.mps.is_available() else "cpu"
# dir_to_save = "/Users/maxkucher/opencv/face_generator/edits"
# photos_dir = "/Users/maxkucher/opencv/face_generator/photos"
# editor = Editor(device, dir_to_save, photos_dir)
# editor()
