# convert the json files generated by nvisii_mvs into a nerf transforms.py 
# this is compatible with ngp

import json
import math
import subprocess
import argparse
import cv2
import glob
import numpy as np
import pyexr
import transforms3d 

parser = argparse.ArgumentParser()

parser.add_argument(
    '--path', 
    type = str, 
    default = None,
    required = True,
)

parser.add_argument(
    '--ext', 
    type = str, 
    default = 'png',
)

parser.add_argument(
    '--nb_test', 
    type = int, 
    default = 0,
    help = 'How many images do you want to use as test, counting backwards'
)
parser.add_argument(
    '--opencv', 
    action="store_true",
    help = 'if you ended up generating this with opencv coordinate frame'
)
parser.add_argument(
    '--row', 
    action="store_true",
    help = 'row major vs row major'
)
def visii_camera_frame_to_rdf(T_world_Cv):
    """Rotates the camera frame to "right-down-forward" frame
    Returns:
        T_world_camera: 4x4 numpy array in "right-down-forward" coordinates
    """
    # C = camera frame (right-down-forward)
    # Cv = visii camera frame (right-up-back)
    T_Cv_C = np.eye(4)
    T_Cv_C[:3, :3] = transforms3d.euler.euler2mat(np.pi, 0, 0)
    T_world_C = T_world_Cv @ T_Cv_C
    return T_world_C.tolist()

def make_json_file(json_files,out_file = 'transforms'):
  out = {}
  out['frames']=[]

  for i_file_name, file_name in enumerate(json_files):
    if "transforms" in file_name:
      continue 
    c2w=[]
    # print(file_name)


    with open(file_name) as json_file:
      data = json.load(json_file)
    c2w = data["camera_data"]["cam2world"]        
    # print(data["camera_data"])

    fr={}
    fr["file_path"]= f"{file_name.split('/')[-1].split('.')[0]}.{opt.ext}"
    if opt.opencv: 
      if opt.row:
        fr["transform_matrix"]=visii_camera_frame_to_rdf(np.array([
                            [c2w[0][0],c2w[0][1],c2w[0][2],c2w[0][3]],
                            [c2w[1][0],c2w[1][1],c2w[1][2],c2w[1][3]],
                            [c2w[2][0],c2w[2][1],c2w[2][2],c2w[2][3]],
                            [c2w[3][0],c2w[3][1],c2w[3][2],c2w[3][3]]]))

      else:
        fr["transform_matrix"]=visii_camera_frame_to_rdf(np.array([
                            [c2w[0][0],c2w[1][0],c2w[2][0],c2w[3][0]],
                            [c2w[0][1],c2w[1][1],c2w[2][1],c2w[3][1]],
                            [c2w[0][2],c2w[1][2],c2w[2][2],c2w[3][2]],
                            [c2w[0][3],c2w[1][3],c2w[2][3],c2w[3][3]]]))


    else:
      fr["transform_matrix"]=[[c2w[0][0],c2w[1][0],c2w[2][0],c2w[3][0]],
                            [c2w[0][1],c2w[1][1],c2w[2][1],c2w[3][1]],
                            [c2w[0][2],c2w[1][2],c2w[2][2],c2w[3][2]],
                            [c2w[0][3],c2w[1][3],c2w[2][3],c2w[3][3]]]  
    out['frames'].append(fr)

  out['aabb'] = [
    [
      data['camera_data']['scene_min_3d_box'][0],
      data['camera_data']['scene_min_3d_box'][1],
      data['camera_data']['scene_min_3d_box'][2],
    ],
    [
      data['camera_data']['scene_max_3d_box'][0],
      data['camera_data']['scene_max_3d_box'][1],
      data['camera_data']['scene_max_3d_box'][2],
    ],
  ]

  fx = data['camera_data']['intrinsics']['fx']
  cx = data['camera_data']['intrinsics']['cx']
  # camang = math.atan(cx/fx)*2
  out['fl_x']=data['camera_data']['intrinsics']['fx']
  out['fl_y']=data['camera_data']['intrinsics']['fy']
  out['cx']=data['camera_data']['intrinsics']['cx']
  out['cy']=data['camera_data']['intrinsics']['cy']
  out['w']=data['camera_data']['width']
  out['h']=data['camera_data']["height"]

  with open(f'{opt.path}/{out_file}.json', 'w') as outfile:
      json.dump(out, outfile, indent=2)

opt = parser.parse_args()
scale = 1
json_files = sorted(glob.glob(opt.path+"*.json"))

if opt.nb_test > len(json_files)-1:
  raise('too many test images')

if opt.nb_test > 0: 
  json_files_test = json_files[len(json_files)-opt.nb_test:]
  json_files = json_files[:len(json_files)-opt.nb_test]
  make_json_file(json_files)
  make_json_file(json_files_test,'transforms_test')
else:
  make_json_file(json_files)
