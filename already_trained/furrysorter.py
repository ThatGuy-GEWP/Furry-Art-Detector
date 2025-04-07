import argparse

parser = argparse.ArgumentParser(
    prog="FurSorter", 
    description="Sorts furry images from non-furry images",
    epilog="Uhhhh"
)

parser.add_argument("filename", help="the file or directory to sort.")
parser.add_argument('-s', '--sensitivity', help="Sets the cutoff for an image being furry.\n defaults to 0.75 meaning only images the network is more then 75 percent sure are counted as furry.")

args = parser.parse_args()


filename = args.filename
sens = args.sensitivity or 0.85 # cutoff for furry sorting

print("Loading Dependencies....")

import torch.nn as nn
import torch.utils.data.dataloader
import time
import torch

import warnings

from PIL import Image
from os import listdir
from os import getcwd
from os import path
from os import mkdir

from torchvision import transforms

from tqdm import tqdm



dir = filename
files = listdir(dir)


def is_valid_image_pillow(file_name):
    try:
        with Image.open(file_name) as img:
            img.verify()
            return True
    except (IOError, SyntaxError):
        return False

toProcess = []
toProcessFiles = []


for i in tqdm (range (len(files)), desc="Find images in directory..."):
    file_name = files[i]
    file = dir + "\\" + file_name

    with warnings.catch_warnings(action="ignore"):
        if path.exists(file):  # Check if the file exists
            if(is_valid_image_pillow(file)):
                img = Image.open(file).convert("RGB").resize((256, 256))
                toProcess.append(img)
                toProcessFiles.append(file)
                #print("Added "+file_name)
        else:
            #print(f"File '{file_name}' not found.")
            #print()
            continue

def CreateOrLoadDir(dir):
    if(not path.exists(dir)):
        mkdir(dir)

print(f"Testing {len(toProcess)} Images found...")

time.sleep(0.8)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Network(nn.Module): # the network itself
    def __init__(self, num_classes=1):
        super(Network, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 256 -> 128
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 128 -> 64
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 64 -> 32
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 32 -> 16
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5: 16 -> 8
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = Network().to(device) # send to GPU

model.load_state_dict(torch.load("model_130_robust.pt", map_location=device, weights_only=True))

is_furry_cutoff = sens

def getLabel(num):
    if(num < is_furry_cutoff): return "furry"
    if(num > is_furry_cutoff): return "not_furry"


tran = transforms.ToTensor()

model.eval()

i = 0

isFurryImage = []
testResults = []


import shutil

sig = nn.Sigmoid()


model.eval()
for i in tqdm (range (len(toProcess)), desc="Testing Images..."):
    image = toProcess[i]
    asTensor = tran(image).to(device)
    asTensor = torch.unsqueeze(asTensor, 0)

    res = model(asTensor)

    del asTensor

    isFurryImage.append(getLabel(sig(res).item()) == "furry")
    testResults.append(sig(res).item())

    res = res.item()
    i+=1
    


furCount = 0
for i in range(len(isFurryImage)):
    if(isFurryImage[i] == True):
        furCount += 1

print(f"Got {furCount} Furry images out of {len(isFurryImage)} images checked.")
time.sleep(0.15)

furryImagesFolder = getcwd()

targPath = getcwd()

CreateOrLoadDir(furryImagesFolder)

for i in tqdm (range (len(isFurryImage)), desc="Copying Picked Images..."):
    toPath = targPath

    if(isFurryImage[i] == True):
        toPath += "\\furry_images"
    else:
        toPath += "\\nonfurry_images"

    CreateOrLoadDir(toPath)

    _, file_extension = path.splitext(toProcessFiles[i])
    shutil.copy(toProcessFiles[i], toPath+"\\"f"{i}__{round(testResults[i]*100, 2)}%{file_extension}")
