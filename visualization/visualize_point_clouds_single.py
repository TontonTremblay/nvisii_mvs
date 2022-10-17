import numpy as np 
import open3d as o3d
import argparse
import cv2 
import simplejson as json
# import open3d as o3d
import meshcat
import transforms3d
import glob


parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
opt = parser.parse_args()

def convert_from_uvd(u, v, d,fx,fy,cx,cy):
    # d *= self.pxToMetre
    x_over_z = (cx - u) / fx
    y_over_z = (cy - v) / fy
    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z
    return x, y, z


def get_pointcloud(depth, intrinsics, flatten=False,mask=None):
    """Projects depth image to pointcloujd

    Args:
        depth: HxW float array of perspective depth in meters.
        intrinsics: 3x3 float array of camera intrinsics matrix.
        flatten: whether to flatten pointcloud

    Returns:
    points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) / intrinsics[0, 0]
    py = (py - intrinsics[1, 2]) / intrinsics[1, 1]
    z = depth / np.sqrt(1. + px**2 + py**2)
    px *= z
    py *= z
    points = np.float32([px, py, z]).transpose(1, 2, 0)
    if mask is not None:
        points = points[mask]
    if flatten:
        points = np.reshape(points, [-1, 3])
    return points

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
    return T_world_C

def get_pointcloud_from_file(path):
    print(f'loading {path}')
    img_depth_range = cv2.imread(path,  
                        cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
    img_depth_range[img_depth_range<-3.4028235e+37] = img_depth_range.max()
    img_depth_range[img_depth_range>3.4028235e+37] = img_depth_range.min()

    mask = cv2.imread(path.replace('depth','seg'),  
                        cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 

    mask[mask == mask.max()] = 0
    mask[mask > 0] = 255
    mask_color = (mask > 0)
    mask = (mask[:,:,0] > 0)

    with open(path.replace('depth.exr',"json"), 'r') as f:
        meta = json.load(f) 


    intrinsics = meta['camera_data']['intrinsics']
    intrin_mat = np.array([[intrinsics['fx'],0,intrinsics['cx']],
                    [0,intrinsics['fy'],intrinsics['cy']],
                    [0,0,1]])

    xyz2 = get_pointcloud(img_depth_range[:,:,0],intrin_mat,mask=mask,flatten=True)

    # get the color
    im = cv2.imread(path.replace('depth.exr',"png"))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    im = im[mask_color]
    im = np.reshape(im, [-1, 3])/255

    # Apply transform in opencv coordinate frame
    trans = np.array(meta['camera_data']["cam2world"]).T
    trans = visii_camera_frame_to_rdf(trans)

    return {
        "pointcloud":xyz2,
        "colors":im,
        "camera_trans":trans,
    }



vis = o3d.visualization.Visualizer()
vis.create_window()

# 0,0,0 frame
# frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
# vis.add_geometry(frame)


files = glob.glob(opt.path+"*.depth.exr")

for i in range(10):
    file = files[i]
    output = get_pointcloud_from_file(file)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(output['pointcloud'])
    pcd.colors = o3d.utility.Vector3dVector(output['colors'])
    pcd.transform(output['camera_trans'])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    frame.transform(output['camera_trans'])
    frame.scale(0.05, center=frame.get_center())
    vis.add_geometry(frame)
    vis.add_geometry(pcd)

#add the scene.ply
# pcd = o3d.io.read_point_cloud(opt.path+"scene.ply")
# vis.add_geometry(pcd)

view_ctl = vis.get_view_control()
view_ctl.set_front([0.2,0,0.2])
view_ctl.set_up([0,0,1])
view_ctl.set_lookat([0,0,0])
vis.run()
vis.destroy_window()