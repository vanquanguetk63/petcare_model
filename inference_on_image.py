import torch
from torchvision.models.resnet import BasicBlock, Bottleneck
from model import ResNet
import cv2
import json
import numpy as np

with open("label.json", "rb") as f:
    label = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=120)
model.load_state_dict(torch.load("checkpoint.pt"))
model.to(device)
model.eval()

def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = (img.transpose((2, 0, 1)) - 127.5) * 0.0078125
    img = torch.from_numpy(img.astype(np.float32))
    img = img.unsqueeze(0).to(device)
    return img

def inference(img):
    return(label[str(np.argmax(model(img)[0].detach().cpu().numpy()))])

