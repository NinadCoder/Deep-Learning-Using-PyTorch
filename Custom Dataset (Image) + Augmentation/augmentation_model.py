import torch
from torchvision import transforms, datasets 
from torchvision.utils import save_image

root_dir = r"C:\Users\ninad\Desktop\Deep Learning\Custom Dataset (Image)\cats_dogs_resized"

my_transforms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0,0.0,0.0], std=[1.0,1.0,1.0])
])

dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)

img_num = 0
for _ in range(1):
    for img, label in dataset:
        save_image(img , 'img' + str(img_num)+'.png')
        img_num += 1