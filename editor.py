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
        self.atr_list = ['Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
       'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
       'Double_Chin', 'Eyeglasses', 'Gray_Hair', 'High_Cheekbones', 'Male',
       'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
       'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
       'Rosy_Cheeks', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
       'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
       'Wearing_Necklace', 'Wearing_Necktie', 'Young']

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

        link = self.choose_photo()


        x, height, width = self.prepare_image(link)

        y = self.input_features()


        t_res = model.predict(x, y)
        res = self.prepare_result(t_res, width, height)
        self.save_result(res)



# device = "mps" if torch.backends.mps.is_available() else "cpu"
# dir_to_save = "/Users/maxkucher/opencv/face_generator/edits"
# photos_dir = "/Users/maxkucher/opencv/face_generator/photos"
# editor = Editor(device, dir_to_save, photos_dir)
# editor()
