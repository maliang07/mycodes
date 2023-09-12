
import pandas as pd
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from models.Resnet import resnet50
import h5py
import numpy as np
from dataset import FE_EXT_Dataset
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transf = transforms.Compose(
    [
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
)

h5_dir ='E:/256x256-256-2/'

def h5file_save(file_name_list, feats_list):
    for image_name, feature in zip(file_name_list, feats_list):
        feature = feature.astype(np.float32)
        h5file_name = os.path.join(h5_dir, image_name.split('_')[0] +'.h5')
        h5key = image_name.split('.')[0]
        with h5py.File(h5file_name, 'a') as f:
            groups =[key for key in f.keys()]
            if h5key not in groups:
                f.create_dataset(h5key, data=feature)
            else:
                print(h5key)
DEBUG=False
if __name__ == '__main__':

    image_dir = './256x256-3/JPG_Files/'
    images_lst = []
    files = os.listdir(image_dir)
    if DEBUG: files = files[-100:]
    print('files',len(files))

    for file in files:
        if file.endswith('.jpg'):
            images_lst.append(file)

    test_dataset = FE_EXT_Dataset(image_dir, images_lst, transf)
    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, \
                                              num_workers=8, pin_memory=False,drop_last=False)

    # model = resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True).cuda()
    # pretext_model = torch.load(r'./weights/best_ckpt.pth')
    # pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    # model = efficientnet_b2(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True).cuda()
    pretext_model = torch.load(r'./weights/best_ckpt.pth')
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=False)

    file_name_list = []
    feats_list = []
    model.eval()
    with torch.no_grad():
        for filenames, batch in tqdm(data_loader):
            batch = batch.cuda()
            features = model(batch)
            features = features.cpu().numpy()
            file_name_list.extend(filenames)
            feats_list.extend(features)

            #if len(file_name_list)>300000:
                #h5file_save(file_name_list,feats_list)
                #file_name_list = []
                #feats_list = []

        h5file_save(file_name_list, feats_list)

