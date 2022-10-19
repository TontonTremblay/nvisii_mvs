# NVISII MULTIVIEW SYNTHETISER 

![renders_example](https://i.imgur.com/O2BrQ2u.jpg)

This repo is the skeleton code used to generate different dataset, [RTMV](https://www.cs.umd.edu/~mmeshry/projects/rtmv/), [watch it move](https://nvlabs.github.io/watch-it-move/), [graspnerf](https://nerfgrasp.github.io/), etc.


## Making a motion video
Please download the following [file](https://github.com/torresjrjr/Bezier.py/blob/master/Bezier.py) and put it in this folder. 

![EXAMPLE](https://i.imgur.com/WKyz34b.mp4)

In the config file you need to specify the `camera_type: camera_movement` and the number of anchor points you want your motion to be in render, `camera_nb_anchor: 70`. 
This will generate 70 anchor points within the sphere defined by `camera_theta_range` and `camera_elevation_range`. 
The size of the sphere is defined by `camera_fixed_distance_factor`. 
If no `look_at` is defined, then the algorithm randomly picked an object to look at which is also smoothed out over time. 
Finaly, you can defined a translation to be applied to the camera if you think the camera is too close to the scene, or the movement to wide, `to_add_position: [0.3,0,0]`. 



# Citation
If you use this code in your research please cite the following: 
```
@misc{morrical2021nvisii,
      title={NViSII: A Scriptable Tool for Photorealistic Image Generation}, 
      author={Nathan Morrical and Jonathan Tremblay and Yunzhi Lin and Stephen Tyree and Stan Birchfield and Valerio Pascucci and Ingo Wald},
      year={2021},
      eprint={2105.13962},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```