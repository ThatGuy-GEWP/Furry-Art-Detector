from PIL import Image, ImageFilter
from os import listdir
from os import getcwd

import random as rnd
import numpy as np
import os


toSize = (256, 256)


def main():
    working = getcwd()

    import sys, time

    if(len(sys.argv) <= 1):
        print("Drop a file/files or a folder to convert.")
        input('Enter to confirm:')
        return

    if (len(sys.argv) > 2):
        for i in range(1, len(sys.argv)):
            droppedFile = sys.argv[i]
            im = Image.open(droppedFile).convert("RGB")
            im = im.resize((128, 128))
            im.save(working+f"\\img{i}_{im.size[0]}x{im.size[1]}.png")
        return


    working = os.path.dirname(sys.argv[0]) + "\\"

    inputDir = "nil"
    outputDir = "converter_out"

    finalName = ""

    if(len(sys.argv) == 2):
        if(os.path.isdir(sys.argv[1])):
            inputDir = sys.argv[1]
            finalName = os.path.basename(sys.argv[1])

    outputDir += f"_[{finalName}]"

    if(not os.path.exists(working+outputDir)):
        os.mkdir(working+outputDir)

    files = listdir(inputDir)

    def img_resized(path):
        im = Image.open(path).convert("RGB")
        return im.resize(toSize)

    def img_flipped(path):
        im = Image.open(path).convert("RGB")
        return im.transpose(Image.FLIP_LEFT_RIGHT).resize(toSize)

    def img_randomized_path(path : str):
        im = Image.open(path).convert("RGB").resize(toSize)
        return add_salt_and_pepper(im, 0.05)

    def img_randomized(im : Image):
        return add_salt_and_pepper(im, 0.05)

    def add_salt_and_pepper(image : Image, amount):

        pixels = image.load()

        for i in range(image.size[0]): # for every pixel:
            for j in range(image.size[1]):
                if(rnd.random() < amount):
                    pixels[i,j] = (int(pixels[i,j][0] - rnd.uniform(0, 80)), int(pixels[i,j][1] - rnd.uniform(0, 80)), int(pixels[i,j][2] - rnd.uniform(0, 80)))

        return image


    cnt = 0

    from tqdm import tqdm

    print(f"Converting \"{inputDir}\"...")

    for i in tqdm (range (len(files)), desc="Converting Images..."):
        f = files[i]

        rf = inputDir+"\\"+f

        img_resized(rf).save(working+f"\\{outputDir}\\"+str(cnt)+".png")
        cnt += 1

        img_flipped(rf).save(working+f"\\{outputDir}\\"+str(cnt)+".png")
        cnt += 1

        img_randomized_path(rf).save(working+f"\\{outputDir}\\"+str(cnt)+".png")
        cnt += 1

        #print(f"{cnt}")

    print("Done!")

main()