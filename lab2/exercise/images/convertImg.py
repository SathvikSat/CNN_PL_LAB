import torch
from PIL import Image
from torchvision import transforms
import sys
sys.path.append("utils")
from test_utils import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#converts image from jpg to cpp tensor format

def convertImg(infile):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = Image.open(infile)
    input_tensor = preprocess(input_image)
    
    img = mpimg.imread(infile)
    imgplot = plt.imshow(img)
    plt.show()
    f = open("image.tensor","wb")
    writeTensor(input_tensor,f)
    f.close()


if __name__ == "__main__":
    if(len(sys.argv) > 1):
        convertImg(sys.argv[1])
    else:
        print("Input image required !")
