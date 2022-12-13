# NVISII Multi-View Synthetiser 

![renders_example](https://i.imgur.com/O2BrQ2u.jpg)

This repo is the skeleton code used to generate different dataset, [RTMV](https://www.cs.umd.edu/~mmeshry/projects/rtmv/), [watch it move](https://nvlabs.github.io/watch-it-move/), [graspnerf](https://nerfgrasp.github.io/), etc.

## Installation 

This code base needs a special version of NViSII, which you can download [here](https://www.dropbox.com/s/m85v7ts981xs090/nvisii-1.2.dev47%2Bgf122b5b.72-cp36-cp36m-manylinux2014_x86_64.whl?dl=0). From there do the following, 
```
pip install nvisii-1.2.dev47+gf122b5b.72-cp36-cp36m-manylinux2014_x86_64.whl
``` 

# Rendering scenes

The RTMV datasets has 4 types of environment. These can be recreated by using the different configs. Please note that this repo does not have any downloadable content, links a provided for you to visualize the data. 

On top of the RTMV like dataset you can generate, we also offer a config to render a 360 view of a model. You are also welcome to generate your own config file as the scene config driven feel free to mix things up. 

## 360 view of an object or scene



## Falling object scene

## Using hdri map





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