from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import glob
from os.path import exists
import PIL
import torchvision 
import pyexr
import cv2 
import simplejson as json 

class nvisii_mvs_dataset(Dataset):
    def __init__(self, 
        path,
        segmentation_overlay=False,
        overlay_keypoints=False
        ):
        
        self.segmentation_overlay = segmentation_overlay
        self.overlay_keypoints = overlay_keypoints

        def loadimages(root):
            """
            Find all the images in the path and folders, return them in imgs. 
            """
            imgs = []

            def add_json_files(path,):
                for imgpath in sorted(glob.glob(path+"/*.png")):
                    # if 'rgb' in imgpath:
                    imgs.append([imgpath])
                for imgpath in sorted(glob.glob(path+"/*.jpg")):
                    # imgs.append([imgpath])
                    # if 'rgb' in imgpath:
                    imgs.append([imgpath])
                for imgpath in sorted(glob.glob(path+"/*.exr")):
                    # print(imgpath)
                    # if "depth" in imgpath or "seg" in imgpath or 'nerf' in imgpath:
                    if "depth" in imgpath or "seg" in imgpath:
                        continue
                    # imgs.append([imgpath])
                    # if 'rgb' in imgpath:
                    imgs.append([imgpath,imgpath.replace('exr','seg.exr')])
                    

            def explore(path):
                if not os.path.isdir(path):
                    return
                folders = [os.path.join(path, o) for o in os.listdir(path) 
                                if os.path.isdir(os.path.join(path,o))]
                if len(folders)>0:
                    for path_entry in folders:                
                        explore(path_entry)
                    # add_json_files(path)
                # else:
                add_json_files(path)


            explore(root)

            return imgs

        def load_data(path):
            '''Recursively load the data.  This is useful to load all of the FAT dataset.'''
            imgs = loadimages(path)
            
            # Check all the folders in path 
            for name in os.listdir(str(path)):
                imgs += loadimages(path +"/"+name)
            print(f'dataset: {len(imgs)} images')
            return imgs


        self.transform = T.Compose([
                # T.ToPILImage(),
                T.Resize(400),
                T.ToTensor()])

        self.imgs = load_data(path)
        print(len(self.imgs))

    def __len__(self):
        # When limiting the number of data
        return len(self.imgs)     

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.imgs[index][0]
        # print(image)
        # print(image)
        if 'exr' in image:
            def linear_to_srgb(img):
                limit = 0.0031308
                return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)


            image = pyexr.open(image).get()
            # image = cv2.imread(image,  
            #             cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
            image = image[:,:,:3]
            image[image>1]=1
            image[image<0]=0
            image = linear_to_srgb(image)*255

            if self.segmentation_overlay:
                from skimage.color import label2rgb 
                seg = pyexr.open(self.imgs[index][1]).get()
                # set background to 0
                seg[seg>3.4028235e+37] = 0 
                seg[seg<-3.4028235e+37] =0 
                seg = np.uint8(seg[:,:,0])
                image = np.uint8(image)

                print(seg.shape,seg.min(),seg.max())
                print(image.shape,image.min(),image.max())
                arr = label2rgb(
                    image = image,
                    label=seg.astype(int),
                    # alpha = 0.5,
                    bg_label=0
                )
                print(arr.shape,arr.min(),arr.max())
                image = PIL.Image.fromarray(np.uint8(arr*255))
                # image.save("tmp.png")
                # raise()

            else:
                image = PIL.Image.fromarray(np.uint8(image))
        else:
            image = PIL.Image.open(image)

        # raise()

        if self.overlay_keypoints:
            from PIL import Image, ImageDraw

            draw = ImageDraw.Draw(image)
            r = 10
            colors = [
                (255,0,0),
                (0,255,0),
                (0,0,255),
                (255,255,0),
                (255,0,255),
                (0,255,255),
                (255,255,255),
                (0,0,0),
                (155,155,155),

            ]
            # load the json file 
            json_path = self.imgs[index][0].replace("png",'json').replace('exr','json')
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # draw lines


            for obj in data['objects']:

                draw.line((tuple(obj['projected_cuboid'][0]), tuple(obj['projected_cuboid'][1])), (0,0,0),r)
                draw.line((tuple(obj['projected_cuboid'][1]), tuple(obj['projected_cuboid'][2])), (0,0,0),r)
                draw.line((tuple(obj['projected_cuboid'][3]), tuple(obj['projected_cuboid'][2])), (0,0,0),r)
                draw.line((tuple(obj['projected_cuboid'][3]), tuple(obj['projected_cuboid'][0])), (0,0,0),r)

                # draw back
                draw.line((tuple(obj['projected_cuboid'][4]), tuple(obj['projected_cuboid'][5])), (0,0,0),r)
                draw.line((tuple(obj['projected_cuboid'][6]), tuple(obj['projected_cuboid'][5])), (0,0,0),r)
                draw.line((tuple(obj['projected_cuboid'][6]), tuple(obj['projected_cuboid'][7])), (0,0,0),r)
                draw.line((tuple(obj['projected_cuboid'][4]), tuple(obj['projected_cuboid'][7])), (0,0,0),r)

                # draw sides
                draw.line((tuple(obj['projected_cuboid'][0]), tuple(obj['projected_cuboid'][4])), (0,0,0),r)
                draw.line((tuple(obj['projected_cuboid'][7]), tuple(obj['projected_cuboid'][3])), (0,0,0),r)
                draw.line((tuple(obj['projected_cuboid'][5]), tuple(obj['projected_cuboid'][1])), (0,0,0),r)
                draw.line((tuple(obj['projected_cuboid'][2]), tuple(obj['projected_cuboid'][6])), (0,0,0),r)

                for i_key,ij_key in enumerate(obj["projected_cuboid"]):
                    draw.ellipse((ij_key[0]-r, ij_key[1]-r, ij_key[0]+r, ij_key[1]+r), 
                        fill=colors[i_key], 
                        outline=(0, 0, 0))

        X = self.transform(image)
        
        return X 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        default=None,
        help = "folder of images"
    )

    parser.add_argument(
        '--sorted',
        action='store_true',
        help = "folder of images"
    )
    parser.add_argument(
        '--out',
        default = 'grid.png',
        help = "folder of images"
    )
    parser.add_argument(
        '--nrow',
        default = 10,
        help = "folder of images"
    )
    parser.add_argument(
        '--batch',
        default = 50,
        type = int,
        help = "how many images to load"
    )

    parser.add_argument(
        '--overlay_segmentation',
        action='store_true',
        help = "overlay segmentation"
    )

    parser.add_argument(
        '--overlay_keypoints',
        action='store_true',
        help = "overlay cuboid keypoints"
    )


    opt = parser.parse_args()
    
    batch_size = opt.batch
    if opt.path is None:
        path = 'visii_mvs/'
    else:
        path = opt.path            
    
    transformed_dataset = nvisii_mvs_dataset(
        path = path,
        segmentation_overlay = opt.overlay_segmentation,
        overlay_keypoints = opt.overlay_keypoints,
    )

    shuffle = True
    if opt.sorted:
        shuffle = False
    
    train_dl = DataLoader(
        transformed_dataset, 
        batch_size, 
        shuffle=shuffle, 
        num_workers=1, 
        pin_memory=False,
    )
    for batch in train_dl:
        grid = make_grid(batch, nrow=int(opt.nrow))
        torchvision.utils.save_image(grid,opt.out)
        break
