import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.adam
import torch.utils.data.dataloader

import os

import torch
import matplotlib.pyplot as plt

from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader

###-------- Variables start

batch_size = 40 # batch size for training and testing
image_res = 256 # resolution of images before passed into network
shuffle = True # should dataset be shuffled?

learn_rate=0.001 # rate of learning
weight_decay=0.0001 # rate of weight decay (should be 3-4 zeros more then learn_rate)

testing_period = 5 # how many epochs to wait before testing
saving_period = 5 # how many epochs to wait before saving
checkpoint_saving_period = 5

save_as_checkpoint = False # if true, save will include checkpoint data
saving_directory = "models_v4_Low_Decay" # self explainitory

epochs = 500 # how many epochs to train

resume_checkpoint = True #if true the checkpoint will be loaded from checkpoint_name
checkpoint_name = f"models_checkpoint_v4_Low_Decay\\modelcheckpoint_20.cp" # checkpoint to load

checkpoint_directory = "models_checkpoint_v4_Low_Decay" # directory to save checkpoints too

###--------- Variables end



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])
dataset = ImageFolder('dataset', transform=transform) # loads dataset

train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True) # training data
test_dataloader = DataLoader(ImageFolder('dataset_furry_notseen', transform=transform), batch_size=batch_size, shuffle=shuffle, pin_memory=True)


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

loss_fn = nn.BCEWithLogitsLoss() # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay) # optimizer

def CreateOrLoadDir(dir):
    if(not os.path.exists(dir)):
        os.mkdir(dir)
    return dir


# For actual usage of the model, you should do > 0.85 ish instead of 0.5
# The model picks up on non-furry images that include furry ears of lots triangles some times,
# but isnt certain and will output in the 0.6-0.8 range (mostly on icons that include furry characters)

def getLabel(num): # returns the string label of the output neuron
    if(num > 0.5): return "furry"
    if(num < 0.5): return "not_furry"

def getChoice(num): # returns the numerical label of the output neuron
    if(num > 0.5): return 0
    if(num < 0.5): return 1

sig = nn.Sigmoid()

def test(loader, model): # tests the model
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for img, label in tqdm.tqdm(loader, desc="Testing "):
            modelChoice = 1.0 - sig(model(img.to(device))[0][0])
            
            if(getChoice(modelChoice) == getChoice(label[0])):
                correct += 1
            total += 1

    return (correct / total) * 100.0, (correct / total)


import tqdm

#disable debugging APIS
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-tensor-cores
torch.set_float32_matmul_precision("medium")

# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner
torch.backends.cudnn.benchmark = True

def train(dataloader, model, loss_fn, optimizer):
    """Train the model"""
    model.train()
    running_loss = 0.0

    correct = 1
    total = 1
    
    for X, y in tqdm.tqdm(dataloader, desc="Training"):
        # Forward pass
        pred = model(X.to(device))
        loss = loss_fn(torch.flatten(pred), 1.0 - y.to(device))
        
        # uncomment to get accuracy
        # modelChoice = 1.0 - sig(pred[0][0]).detach() .item()

        # if(getChoice(modelChoice) == getChoice(y[0].detach().item())):
        #     correct += 1
        # total += 1

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
    
    
    # Return average loss
    return running_loss / len(dataloader), (correct / total) * 100.0


def save_checkpoint(model, epoch, optimizer, loss, path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, CreateOrLoadDir(os.getcwd() + "\\" + checkpoint_directory) + "\\" + path)


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch']


CreateOrLoadDir(os.getcwd() + "\\" + saving_directory)


cepochs = 0

if(resume_checkpoint == True):
    cepochs = load_checkpoint(checkpoint_name, model, optimizer)


#save_checkpoint(model, epochs, optimizer, loss_fn, checkpoint_name)

for epoch in range(epochs):
    if((epoch) % testing_period == 0):
        test_accuracy, realDiv = test(test_dataloader, model)
        print(f"Test Accuracy: {round(test_accuracy, 2)}, {round(realDiv, 4)}%")

    if((epoch) % checkpoint_saving_period == 0):
        save_checkpoint(model, epoch+cepochs, optimizer, loss_fn, f"modelcheckpoint_{epoch+cepochs}.cp")
        print(f"Saved checkpoint as 'modelcheckpoint_{epoch+cepochs}.cp' to current working directory")

    if((epoch) % saving_period == 0):
        torch.save(model.state_dict(), saving_directory + "\\" + f'model_out_E{epoch+cepochs}.pt')
        print(f"Saved model as 'model_out_E{epoch+cepochs}.pt' to {saving_directory}")

    loss, train_accuracy = train(train_dataloader, model, loss_fn, optimizer)
    print(f"Epoch:{epoch+cepochs} Loss:{loss} Train Accuracy:{round(train_accuracy, 2)}%")

#save_checkpoint(model, epochs, optimizer, loss_fn, checkpoint_name)