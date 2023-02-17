
# Structure of the data in a zip file
# f1
#   - images 
# f2
#   - images 
# dataset.json

# in json:
# [00002/img00002971.png,
# extrinsics44.flatten,intrinsics_normed.flatten 
# ]

import functools
import gzip
import io
import json
import os
import pickle
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import numpy as np 

import click
import numpy as np
import PIL.Image
from tqdm import tqdm
import cv2 
import glob 
import argparse

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:

    if os.path.dirname(dest) != '':
        os.makedirs(os.path.dirname(dest), exist_ok=True)
    zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
    def zip_write_bytes(fname: str, data: Union[bytes, str]):
        zf.writestr(fname, data)
    return '', zip_write_bytes, zf.close




#####################################################################################################

parser = argparse.ArgumentParser()

parser.add_argument(
    '--path', 
    type = str, 
    default = "/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/valts_eccv/",
)

parser.add_argument(
    '--outf', 
    type = str, 
    default = "/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/valts_eccv_eg3d.zip",
)


opt = parser.parse_args()



archive_root_dir, save_bytes, close_dest = open_dest(opt.outf)
labels = []
from tqdm import tqdm

for i_folder, folder in enumerate(tqdm(sorted(glob.glob(opt.path+"*/")))):
    
    imgs = glob.glob(folder + "*.png") 

    for img_path in imgs:

        image_bits = io.BytesIO()
        # img = PIL.Image.open("output/360_views/00000.png")
        img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        
        name_folder = img_path.split("/")[-2]
        name = img_path.split("/")[-1].replace(".png","")

        white = np.ones((img.shape[0],img.shape[1],3))*255
        alpha = img[:,:,-1]/255.0
        alpha = np.concatenate([
            alpha.reshape((img.shape[0],img.shape[1],1)),
            alpha.reshape((img.shape[0],img.shape[1],1)),
            alpha.reshape((img.shape[0],img.shape[1],1))]
            ,axis=-1)
        img = white *(1-alpha) + alpha * img[:,:,:3]
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = PIL.Image.fromarray(img)
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir,f'{name_folder}/{name}.png'), image_bits.getbuffer())
        # save_bytes(os.path.join(archive_root_dir,f'{str(i_folder).zfill(4)}_{name_folder}/{name}.png'), image_bits.getbuffer())

        # load the data 
        with open(img_path.replace("png","json"), 'r') as f:
            meta = json.load(f) 
        trans = np.array(meta['camera_data']["cam2world"]).T.flatten()
        
        flat_intrinsiscs = [
            meta['camera_data']['intrinsics']['fx']/alpha.shape[0],0,meta['camera_data']['intrinsics']['cx']/alpha.shape[0],
            0,meta['camera_data']['intrinsics']['fy']/alpha.shape[0],meta['camera_data']['intrinsics']['cy']/alpha.shape[0],
        ]
        # print(trans.tolist()+flat_intrinsiscs)
        # raise()
        output = [os.path.join(archive_root_dir,f'{name_folder}/{name}.png'),trans.tolist()+flat_intrinsiscs+[0,0,1]]
        labels.append(output)

#save the meta data
metadata={
    "labels":labels
}
save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))

close_dest()