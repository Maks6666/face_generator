from model import model
import torch
import time
import cv2
import numpy as np
import random
import os

device = "mps" if torch.backends.mps.is_available() else "cpu"


class Generator:
    def __init__(self, device, model, folder, size):
        self.device = device
        self.model = model
        self.folder = folder
        self.size = size

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

    def save(self, folder, image):
        num = random.randint(1, 1000)
        path = os.path.join(folder, f"{num}.jpeg")
        cv2.imwrite(path, image)

    def prepare_image(self, img):
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.size)
        return img

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

            if stop_marker is True:
                break


        if stop_marker is None:
            for value in self.dct.values():
                self.values.append(int(value))

            tensor_values = torch.tensor(self.values, dtype=torch.float).unsqueeze(0).to(self.device)
            return tensor_values

        elif stop_marker == True:
            return None

    def __call__(self):

            z = torch.randn(1, 128).to(self.device)

            y = self.input_features()
            if y is not None:

                start = time.time()
                t_res = model.sample(z, y)
                end = time.time()

                fin_res = self.prepare_image(t_res)


                res_time = end - start
                print(f"Time: {res_time:.2f} seconds")

                self.save(self.folder, fin_res)
                print("Saved successfully!")


# device = "mps" if torch.backends.mps.is_available() else "cpu"
# folder = "/Users/maxkucher/opencv/face_generator/generations"
# size = (256, 256)
#
# generator = Generator(device, model, folder, size)
# generator()