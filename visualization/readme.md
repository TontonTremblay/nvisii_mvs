# VISUALIZATION DATA GENERATED

The scripts in the repo are meant to be used to visualize the 3d data. 

## simple_dataloader.py

This loads a folder data (it can be recursive) and makes a grid of images. 

![grid](https://i.imgur.com/sa1Lh02.png)

```
python visualization/simple_dataloader.py --path /PATH/TO/FOLDER --sorted --nrow 3 --batch 9 --overlay_segmentation --overlay_keypoints
```

This would generate a grid image of 3x3 where each image is overlayed by the segmentation mask and also each object has cuboid keypoints projected. 

## view_camera_frames

You need to have a [mescat-server](https://github.com/rdeits/meshcat-python#starting-a-server) running.

![camera poses](https://i.imgur.com/o5kq3PE.png)

```
python visualization/view_camera_frames.py --path /PATH/TO/FOLDER
```

## Point Cloud

This shows how the depths exported by nvisii can be put together to generate a point cloud. 

![pcd](https://i.imgur.com/vtKUyZj.png)

```
python visualization/visualize_point_cloud.py --path /PATH/TO/FOLDER
```

## Make an animation

If the frames you generated are sequential, then you can animate them using the following. This is a wrap around ffmpeg. 
```
python make_video_folder2.py --path /PATH/TO/FOLDER
```