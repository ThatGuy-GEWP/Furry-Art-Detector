import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader
import torchvision


import torch
import matplotlib.pyplot as plt



model_to_test = "model_out_E10_90ac.safetensors"



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = self.fc_layers(x)
        return x

model = Network().to(device)

model.load_state_dict(torch.load(model_to_test, map_location=device, weights_only=True))

from torchvision.transforms import transforms

transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
dataset = torchvision.datasets.ImageFolder('dataset_furry_notseen', transform=transform)
dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=1, shuffle=True)

model.eval()


import torch
import matplotlib.pyplot as plt

def show_tensor_as_image(tensor, title=None, fig=None, ax=None):
    """
    Display a PyTorch tensor as an image using matplotlib.
    
    Args:
        tensor (torch.Tensor): The tensor to be displayed as an image.
                               Expected shapes:
                               - (H, W) for grayscale
                               - (H, W, 3) for RGB
                               - (3, H, W) for RGB in PyTorch's channel-first format
        title (str, optional): Title to display above the image
    """
    # Convert tensor to numpy for matplotlib compatibility
    if tensor.device != torch.device('cpu'):
        tensor = tensor.cpu()  # Move tensor to CPU if it's on GPU
    
    # Make a copy to avoid modifying the original tensor
    img = tensor.clone().detach()
    
    # Handle different tensor shapes
    if len(img.shape) == 3 and img.shape[0] == 3:
        # Convert from (3, H, W) to (H, W, 3)
        img = img.permute(1, 2, 0)
    
    # Convert to numpy array
    img = img.numpy()
    

    if(fig == None or ax == None):
        # Create a figure and axis for plotting
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Display the image
    # imshow automatically handles grayscale (H, W) and RGB (H, W, 3) arrays
    im = ax.imshow(img)
    
    # Add a colorbar for grayscale images
    if len(img.shape) == 2:
        fig.colorbar(im, ax=ax)
    
    # Add title if provided
    if title:
        ax.set_title(title)
    
    # Hide axis ticks for cleaner visualization
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Display the plot
    plt.tight_layout()
    #plt.show()
    
    return fig, ax  # Return figure and axis for further customization if needed



def getLabel(num):
    if(num > 0.5): return "furry"
    if(num < 0.5): return "not_furry"

plt.ion()

import time as T

lf, la = None, None

def getChoice(num):
    if(num > 0.5): return 0
    if(num < 0.5): return 1

def test(loader, model,):
    correct = 0
    total = 0

    with torch.no_grad():
        for img, label in loader:
            modelChoice = model(img.to(device))[0][0].detach().item()
            
            if(getChoice(modelChoice) == getChoice(label[0].detach().item())):
                correct += 1
            total += 1

    return (correct / total) * 100.0, (correct / total)

unseenAcc, tot = test(dataloader, model)

print(f"Test Accuracy: {round(unseenAcc, 2)}%")

while True:
    image, label = next(iter(dataloader))
    modelChoice = model(image.to(device))[0][0].detach().item()
    lf, la = show_tensor_as_image(image[0], f"Model thinks this is {getLabel(modelChoice)}, it is {getLabel(label)}", lf, la)




    plt.pause(3.0)
    plt.show()
    