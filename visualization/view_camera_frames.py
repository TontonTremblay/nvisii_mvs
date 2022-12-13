import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf

import pyrr 
import simplejson as json

import numpy as np 
import glob 


# These functions were provided by Lucas Manuelli 
def create_visualizer(clear=True, zmq_url='tcp://127.0.0.1:6000'):
    """
    If you set zmq_url=None it will start a server
    """

    print('Waiting for meshcat server... have you started a server? Run `meshcat-server` to start a server')
    vis = meshcat.Visualizer(zmq_url=zmq_url)
    if clear:
        vis.delete()
    return vis

def make_frame(vis, name, T=None, h=0.15, radius=0.001, o=1.0):
    """Add a red-green-blue triad to the Meschat visualizer.
    Args:
      vis (MeshCat Visualizer): the visualizer
      name (string): name for this frame (should be unique)
      h (float): height of frame visualization
      radius (float): radius of frame visualization
      o (float): opacity
    """
    vis[name]['x'].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0xff0000, reflectivity=0.8, opacity=o))
    rotate_x = mtf.rotation_matrix(np.pi / 2.0, [0, 0, 1])
    rotate_x[0, 3] = h / 2
    vis[name]['x'].set_transform(rotate_x)

    vis[name]['y'].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x00ff00, reflectivity=0.8, opacity=o))
    rotate_y = mtf.rotation_matrix(np.pi / 2.0, [0, 1, 0])
    rotate_y[1, 3] = h / 2
    vis[name]['y'].set_transform(rotate_y)

    vis[name]['z'].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x0000ff, reflectivity=0.8, opacity=o))
    rotate_z = mtf.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    rotate_z[2, 3] = h / 2
    vis[name]['z'].set_transform(rotate_z)

    if T is not None:
        print(T)
        vis[name].set_transform(T)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        default=None,
        help = "folder of images"
    )
    opt = parser.parse_args()

    vis = create_visualizer()
    for file in glob.glob(opt.path+"/*.json"):
        with open(file, 'r') as f:
            meta = json.load(f) 
        trans = np.array(meta['camera_data']["cam2world"]).T
        make_frame(vis,file,trans)