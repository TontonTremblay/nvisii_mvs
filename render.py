import os
import time
import glob
import nvisii as visii
import numpy as np
import random
import pybullet as p 
import argparse
import colorsys
import subprocess
import transforms3d 
import trimesh 
import yaml 
import cv2

# from utils import *
from nvisiijohn.utils import *

parser = argparse.ArgumentParser()

parser.add_argument(
    '--config', 
    type = str, 
    default = 'configs/moustafa_chair.yaml' 
)
parser.add_argument(
    '--outf', 
    type = str, 
    default = None,
    help = 'overwrite the config output file'
)
parser.add_argument(
    '--model_path', 
    type = str, 
    default = None,
    help = 'overwrite the config output file'
)
parser.add_argument(
    '--interactive',
    action = 'store_true',
    default = False,
    help = "visualize the rendering process - need x server\
            no rendering/metadata will be saved"
)
opt = parser.parse_args()


with open(opt.config, 'r') as f:
    cfg = DotDict(yaml.safe_load(f))

if opt.outf is not None:
    cfg.outf = opt.outf

if opt.model_path is not None:
    cfg.model_path = opt.model_path


if cfg.file_format is None:
    cfg.file_format = "exr"

# # # # # # # # # # # # # # # # # # # # # # # # #

# UTILS FUNCTIONS

def sphere_renders(
        nb_planes,
        nb_circle,
        elevation_range = [0,180],
        tetha_range = [0,360]
    ):

    positions_to_render = []
    for i_plane in range(nb_planes):
        elevation = np.deg2rad(  elevation_range[0] + \
                                ((i_plane+1) * (elevation_range[1]-elevation_range[0])/(nb_planes+1)))
        for i_circle in range(nb_circle):
            azimuth = np.deg2rad(tetha_range[0]+((i_circle+1) * (tetha_range[1]-tetha_range[0])/(nb_circle+1)))
            eye_position = [
                np.sin(elevation)*np.cos(azimuth),
                np.sin(elevation)*np.sin(azimuth),
                np.cos(elevation),
            ]
            positions_to_render.append(eye_position)
    return positions_to_render

def random_sample_sphere(
        elevation_range = [0,180],
        tetha_range = [0,360],
        nb_frames = 10,
    ):
    to_return = []
    outside = True
    max_radius = 1.00001
    min_radius = 0.99999
    min_max_x = [0,0]
    min_max_y = [0,0]
    min_max_z = [0,0]
    for i_degree in range(tetha_range[0],tetha_range[1],1):
        v = np.cos(np.deg2rad(i_degree))
        if v < min_max_x[0]:
            min_max_x[0] = v
        if v > min_max_x[1]:
            min_max_x[1] = v

    for i_degree in range(tetha_range[0],tetha_range[1],1):
        v = np.sin(np.deg2rad(i_degree))
        if v < min_max_y[0]:
            min_max_y[0] = v
        if v > min_max_y[1]:
            min_max_y[1] = v

    for i_degree in range(elevation_range[0],elevation_range[1],1):
        v = np.cos(np.deg2rad(i_degree))
        if v < min_max_z[0]:
            min_max_z[0] = v
        if v > min_max_z[1]:
            min_max_z[1] = v

    for i in range(nb_frames):
        outside = True
        while outside:

            x = random.uniform(min_max_x[0], min_max_x[1])
            y = random.uniform(min_max_y[0], min_max_y[1])
            z = random.uniform(min_max_z[0], min_max_z[1])

            if  (x**2 + y**2 + z**2) * max_radius < max_radius + 0.0001 \
            and (x**2 + y**2 + z**2) * max_radius > min_radius:
                outside = False
        to_return.append([x,y,z])
    return to_return

# # # # # # # # # # # # # # # # # # # # # # # # #
if os.path.isdir(f'{cfg.outf}'):
    print(f'folder {cfg.outf}/ exists')
else:
    os.mkdir(f'{cfg.outf}')
    print(f'created folder {cfg.outf}/')
# # # # # # # # # # # # # # # # # # # # # # # # #

if opt.interactive:
    visii.initialize(headless=False)
    visii.resize_window(1000,1000)
else:
    visii.initialize(headless=True,
        max_transforms=1000000, 
        max_entities = 1000000,
        max_meshes=10000, 
        max_materials=1000000, 
        max_lights=100, 
        max_textures=1000, max_volumes=1000)
# visii.set_max_bounce_depth(1)

if cfg.denoiser is True: 
    visii.enable_denoiser()
else:
    visii.disable_denoiser()

camera = visii.entity.create(
    name = "camera",
    transform = visii.transform.create("camera"),
    camera = visii.camera.create_from_fov(
        name = "camera", 
        field_of_view = float(cfg.camera_fov), 
        aspect = float(cfg.width)/float(cfg.height)
    )
)
visii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

# LIGHTING


if cfg.dome_light_intensity:
    visii.set_dome_light_intensity(cfg.dome_light_intensity)
else:
    visii.set_dome_light_intensity(1)

if 'render_alpha' in cfg and cfg['render_alpha'] == True:
    visii.set_dome_light_color(visii.vec3(1,1,1),0)
else:
    visii.set_dome_light_color(visii.vec3(1,1,1),1)

# # # # # # # # # # # # # # # # # # # # # # # # #

# LOADING OBJECTS

# for falling scenes
# exporting meta data
cuboids = {}
export_names = []

if not type(cfg.falling_nb_objects) == int:
    cfg.falling_nb_objects = random.randint(cfg.falling_nb_objects[0],cfg.falling_nb_objects[1]+1)

if cfg.scene_type == 'falling':
    # assume google objects
    
    output = create_falling_scene(
        cfg.model_path,
        nb_objects_to_load = cfg.falling_nb_objects,
        model_source = cfg.model_source,
        specific_models = cfg.model_names,
        box_size = cfg.falling_cube_size,
        interactive = opt.interactive,
    )
    entity_list = output['nvisii_entities']
    pybullet_ids = output['pybullet_ids']

    # for exporting meta deta
    for entity in entity_list:
        export_names.append(entity.get_name())
        cuboid = add_cuboid(entity.get_name(), debug=False)
        cuboids[entity.get_name()] = cuboid

elif cfg.model_source == "google_scanned":
    entity_visii = load_google_scanned_objects(cfg.model_path)
    
    # might want to put that in the loading function
    # entity_visii.get_transform().set_angle_axis(visii.pi()/2,visii.vec3(1,0,0))

    cuboid = add_cuboid(entity_visii.get_name(), debug=False)
    # print(cuboid)
    entity_visii.get_transform().add_position(visii.vec3(0,0,-cuboid[0][2]/2))

    cuboids[entity_visii.get_name()] = cuboid
    export_names.append(entity_visii.get_name())

elif cfg.model_source == "amazon_berkeley":
    entity_visii = load_amazon_object(cfg.model_path)
    
    # might want to put that in the loading function
    entity_visii.get_transform().set_angle_axis(visii.pi()/2,visii.vec3(1,0,0))

    cuboid = add_cuboid(entity_visii.get_name(), debug=False)
    cuboids[entity_visii.get_name()] = cuboid
    export_names.append(entity_visii.get_name())

elif cfg.model_source == "lego":
    entity_visii_name_list = load_lego_object(cfg.model_path)
    # print(entity_visii_name_list)
    # raise()
    export_names = entity_visii_name_list
    export_names = [visii.entity.get(entity_visii_name_list[0]).get_transform().get_parent().get_name()]
    # cuboid = add_cuboid(export_names[0], debug=False)
    # cuboids[entity_visii.get_name()] = cuboid
    
elif cfg.model_source == "low_poly_car":
    entity_visii_name_list = load_poly_cars(cfg.model_path)
    export_names = entity_visii_name_list
    # print(export_names)

elif cfg.model_source == "obj":
    entity_visii_name_list = visii.import_scene(
        cfg.model_path,
        visii.vec3(0,0,0),
        visii.vec3(1,1,1), # the scale
        visii.angleAxis(1.57, visii.vec3(1,0,0))
        # visii.angleAxis(1.57, visii.vec3(1,0,0))
    )
    print(entity_visii_name_list)
    parent = entity_visii_name_list.entities[0].get_transform().get_parent()
    print(parent.get_name())
    parent.set_position([0,0,0])
    # texture = visii.texture.create_from_file('tex_obj',cfg.texture_path)
    for m in entity_visii_name_list.entities:
        cuboid = add_cuboid(m.get_name(), debug=False)

        # print(m.get_name())
        # if 'geo' in m.get_name():
        #     cuboid = add_cuboid(m.get_name(), debug=False)
        # # rgb = colorsys.hsv_to_rgb(
        #     random.uniform(0,1),
        #     random.uniform(0.8,1),
        #     random.uniform(0.9,1)
        # )
    for m in entity_visii_name_list.materials:
        print(m.get_name())
        rgb = colorsys.hsv_to_rgb(
            random.uniform(0,1),
            random.uniform(0.8,1),
            random.uniform(0.9,1)
        )

        m.set_base_color(
            visii.vec3(
                1,
                1,
                1,
            )
        )  
        m.set_roughness(0.7)
        m.set_metallic(1)
    # raise()
elif cfg.model_source == "ply":
    if 'obj'in cfg.model_path:
        entity_visii_name_list = visii.import_scene(
            cfg.model_path,
            visii.vec3(0,0,0),
            visii.vec3(.1,.1,.1), # the scale
            # visii.angleAxis(1.57, visii.vec3(1,0,0))
            # visii.angleAxis(1.57*2, visii.vec3(1,0,0))
        )
    else:
        entity_visii_name_list = visii.import_scene(
            cfg.model_path,
            visii.vec3(0,0,0),
            visii.vec3(.1,.1,.1), # the scale
            visii.angleAxis(1.57, visii.vec3(1,0,0))
            # visii.angleAxis(1.57*2, visii.vec3(1,0,0))
        )
    # for i_e in range(len(entity_visii_name_list.entities)):
    #     entity_visii_name_list.entities[i_e].set_material(
    #         visii.material.create(f'hey_{str(i_e)}')
    #             # base_color = visii.vec3(0.2,0.2,0.2))
    #     )
    #     entity_floor = visii.entity.create(
    #         name = "floor",
    #         mesh = visii.mesh.create_plane("mesh_floor"),
    #         transform = visii.transform.create("transform_floor"),
    #         material = visii.material.create("material_floor")
    #     )
    #     entity_floor.get_transform().set_scale(visii.vec3(.1,0.2,1))


elif cfg.model_source == "abc":
    entity_visii_name_list = load_abc_object(cfg.model_path)
    export_names = entity_visii_name_list
    # print(export_names)

elif cfg.model_source == "ycb":
    entity_visii = load_grocery_object(cfg.model_path)
    entity_visii.get_transform().set_scale(visii.vec3(0.01))
    entity_visii.get_transform().set_rotation(visii.angleAxis(-1.57, visii.vec3(1,0,0)))
    export_names = [entity_visii.get_name()]
    # print(export_names)
    
    cuboid = add_cuboid(entity_visii.get_name(), debug=False)
    cuboids[entity_visii.get_name()] = cuboid
    

elif cfg.model_source == "cube":
    entity_visii = visii.entity.create(
            name = "cube",
            mesh = visii.mesh.create_box('cube',[0.5,0.5,0.5]),
            transform = visii.transform.create('cube'),
            material = visii.material.create('cube')
        )
    export_names.append('cube')

elif cfg.model_source == 'nvidia_scanned':
    entity_visii = load_nvidia_scanned(cfg.model_path)
    if entity_visii is None:
        raise 'PROBLEM LOADING OBJECT'
    cuboid = add_cuboid(entity_visii.get_name(), debug=False)
    # print(cuboid)
    entity_visii.get_transform().add_position(visii.vec3(0,0,-cuboid[0][2]/2))

    cuboids[entity_visii.get_name()] = cuboid
    export_names.append(entity_visii.get_name())    # raise()

elif cfg.model_source == 'xyz':
    # render a color point cloud
    import nvisii 
    data = np.load(cfg.model_path)
    transform_papa = nvisii.transform.create("papa")
    cube_mesh = nvisii.mesh.create_box(f"cube")



    for i_p in range(len(data['xyz'])):
        # print(data['xyz'][i_p],data['rgb'][i_p])
        i_pp = random.randint(0,len(data['xyz'])-1)
        cube_t = nvisii.transform.create(
            f"cube_{i_p}",
            position = (data['xyz'][i_pp][0],data['xyz'][i_pp][1],data['xyz'][i_pp][2]),
            scale = (0.003,0.003,0.003)
        )
        cube_t.set_parent(transform_papa)
        cube_m = nvisii.material.create(
            f"cube_{i_p}",
            base_color = (data['rgb'][i_pp][0],data['rgb'][i_pp][1],data['rgb'][i_pp][2]),
            roughness = 0.7
            )
        cube_e = nvisii.entity.create(
            f"cube_{i_p}",
            transform = cube_t, 
            material = cube_m, 
            mesh = cube_mesh
            )
        if i_p > 1000: 
            break
    # add the gripper
    gripper = nvisii.entity.create("gripper",
        transform=nvisii.transform.create("gripper"),
        material = nvisii.material.create('gripper',
            base_color=(118/255,185/255,0)))

    gripper.set_mesh(
        nvisii.mesh.create_from_file(
            "gripper",
            "../baxter_gripper.obj"
            )
        )
    data = np.load(cfg.model_path.replace('pointcloud','grasps'))
    import pyrr
    pose = pyrr.Matrix44(data['poses'][1])
    print(pose[:,-1][:3])
    gripper.get_transform().set_position((pose[:,-1][0],pose[:,-1][1],pose[:,-1][2]))
    quat = nvisii.quat(pose.quaternion.w,pose.quaternion.x,pose.quaternion.y,pose.quaternion.z)
    gripper.get_transform().set_rotation(quat * visii.angleAxis(-np.pi/2, visii.vec3(0,-1,0)))

    # gripper.get_transform().set_rotation(([pose.quaternion.x,pose.quaternion.y,pose.quaternion.z,pose.quaternion.w]))
    # raise()
#import pdb; pdb.set_trace()

# # # # # # # # # # # # # # # # # # # # # # # # #

# SET UP THE SCENE

if cfg.scene_type == 'centered':
    center_scene_model = visii.get_scene_aabb_center()
    bb_max = visii.get_scene_max_aabb_corner()
    bb_min = visii.get_scene_min_aabb_corner()

    distance = np.sqrt((bb_max[0]-bb_min[0])**2+(bb_max[1]-bb_min[1])**2+(bb_max[2]-bb_min[2])**2)    

elif cfg.scene_type == 'falling':
    # compute the volume using the cuboids 3d
    bb_max = entity_list[0].get_min_aabb_corner()
    bb_min = entity_list[0].get_max_aabb_corner()
    center_scene_model = entity_list[0].get_aabb_center()

    for i_entity in range(1,len(entity_list)):
        bb_min_entity = entity_list[i_entity].get_min_aabb_corner()
        bb_max_entity = entity_list[i_entity].get_max_aabb_corner()
        center = entity_list[i_entity].get_aabb_center()

        for k_p in range(3):
            if bb_min_entity[k_p]<bb_min[k_p]:
                bb_min[k_p] = bb_min_entity[k_p]
            if bb_max_entity[k_p]>bb_max[k_p]:
                bb_max[k_p] = bb_max_entity[k_p]
            center_scene_model[k_p] = (center_scene_model[k_p] + center[k_p])/2
    distance = np.sqrt((bb_max[0]-bb_min[0])**2+(bb_max[1]-bb_min[1])**2+(bb_max[2]-bb_min[2])**2)
    center_scene_model = visii.get_scene_aabb_center()


# get the scene information
scene_aabb = [
    [
        visii.get_scene_min_aabb_corner()[0],
        visii.get_scene_min_aabb_corner()[1],
        visii.get_scene_min_aabb_corner()[2],
    ],
    [
        visii.get_scene_max_aabb_corner()[0],
        visii.get_scene_max_aabb_corner()[1],
        visii.get_scene_max_aabb_corner()[2],
    ],
    [
        visii.get_scene_aabb_center()[0],
        visii.get_scene_aabb_center()[1],
        visii.get_scene_aabb_center()[2],
    ]
]

if cfg.point_light:

    # create a random point light. 
    name_point_light = 'point_light'
    light_entity = visii.entity.create(
        name = name_point_light,
        transform = visii.transform.create(name_point_light),
        light = visii.light.create(name_point_light),

    )
    if "light_scale" in cfg:
        light_entity.set_mesh(visii.mesh.create_sphere("light"))
        light_entity.get_transform().set_scale(cfg.light_scale)


    light_entity.get_light().set_intensity(cfg.light_intensity)
    light_entity.get_light().set_falloff(100)
    
    # visii.set_dome_light_intensity(0)

    export_names.append(light_entity.get_name())

    # light_position: [1,1,1]
    # light_temperature: 1000
    if "light_temperature" in cfg:
        light_entity.get_light().set_temperature(cfg.light_temperature)
    if "light_position" in cfg:
        light_positions=[cfg.light_position]
        # print(light_positions)
        light_entity.get_transform().set_position(
            visii.vec3(
                cfg.light_position[0],
                cfg.light_position[1],
                cfg.light_position[2]
            )
        )
    elif cfg.light_movement: 
        light_positions = random_sample_sphere(
                                nb_frames = cfg.camera_nb_frames,
                                elevation_range = [15,85],
                                tetha_range = [0,360]
                            )        
    else:
        light_positions = random_sample_sphere(
                                nb_frames = 1,
                                elevation_range = [15,85],
                                tetha_range = [0,360]
                            )        

        pos_light = light_positions[0]
        light_entity.get_transform().set_position(
            visii.vec3(
                pos_light[0]*cfg.light_distance,
                pos_light[1]*cfg.light_distance,
                pos_light[2]*cfg.light_distance
            )
        )



print('scene distance:',distance)


if len(cfg.camera_look_at) > 0:
    center_scene_model = cfg.camera_look_at


# # # # # # # # # # # # # # # # # # # # # # # # #
# scene extra
# RANDOM DOME LIGHT


if cfg.add_dome_hdri is True:
    visii.set_dome_light_intensity(float(cfg.dome_light_intensity))
    # visii.set_dome_light_rotation(
    #     # visii.angleAxis(visii.pi()/2,visii.vec3(1,0,0)) \
    #     # * visii.angleAxis(visii.pi()/2,visii.vec3(0,0,1))\
    #     visii.angleAxis(random.uniform(-visii.pi(),visii.pi()),visii.vec3(0,0,1))\
    #     * visii.angleAxis(random.uniform(-visii.pi()/6,visii.pi()/6),visii.vec3(0,1,0))\
    #     * visii.angleAxis(random.uniform(-visii.pi()/8,visii.pi()/8),visii.vec3(1,0,0))\
    # )
    # load a random skybox 
    skyboxes = glob.glob(f'{cfg.path_dome_hdri_haven}/*.hdr')
    skybox_random_selection = skyboxes[random.randint(0,len(skyboxes)-1)]

    dome_tex = cv2.imread(skybox_random_selection,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  
    # print(dome_tex.shape)
    import imageio
    dome_tex = imageio.imread(skybox_random_selection)
    # dome_tex = cv2.cvtColor(dome_tex, cv2.COLOR_BGR2RGB)
    dome_tex = np.flipud(dome_tex)
    dome_tex = np.concatenate([dome_tex,np.zeros([dome_tex.shape[0],dome_tex.shape[1],1])],2)
    # print(dome_tex.shape)
    imageio.imwrite('tmp.exr',dome_tex.astype(np.float32))
    # raise()
    # dome_tex = visii.texture.create_from_file('dome_tex',skybox_random_selection)
    dome_tex = visii.texture.create_from_data(
        'dome_tex',
        dome_tex.shape[1],
        dome_tex.shape[0],
        (dome_tex.flatten()).astype(np.float32),
        hdr = True
    )
    visii.set_dome_light_texture(dome_tex)

elif "visii_sun" in cfg and cfg.visii_sun:
    # rgb = colorsys.hsv_to_rgb(
    #     random.uniform(0,1),
    #     random.uniform(0.4,1),
    #     random.uniform(0.8,1)
    # )
    # rgb = visii.vec3(
    #             rgb[0],
    #             rgb[1],
    #             rgb[2],
    #         )
    # pos = visii.vec3(
    #             random.uniform(-10,10),
    #             random.uniform(-10,10),
    #             random.uniform(1,10)
    #         )
    # visii.set_dome_light_sky(
    #     sun_position = pos,
    #     sky_tint = rgb,
    #     atmosphere_thickness = random.uniform(0.5,2),
    #     saturation = random.uniform(0.1,1),
    #     )

    # create a random point light. 
    name_point_light = 'point_light'
    light_entity = visii.entity.create(
        name = name_point_light,
        transform = visii.transform.create(name_point_light),
        light = visii.light.create(name_point_light),
        mesh = visii.mesh.create_sphere("light",),
        material = visii.material.create('light')
        )
    light_tmp = random.uniform(1000,4000)
    light_tmp = 1066.8517017762358
    light_entity.get_light().set_temperature(light_tmp)
    light_entity.get_light().set_intensity(8)
    # light_entity.get_light().set_falloff(0)
    light_entity.get_material().set_alpha(0)
    light_entity.set_visibility(camera= False)
    # export_names.append(light_entity.get_name())


    light_positions = random_sample_sphere(
                            nb_frames = 1,
                            elevation_range = [30,75],
                            tetha_range = [0,360]
                        )        

    pos_light = light_positions[0]
    # pos_light = [0.8075491469781921, 0.0018738600208743073, 0.5898446662654822]

    with open("light.txt",'w')as f:
        f.write(f"pos:{str(pos_light)},temperature:{light_tmp}")
    light_entity.get_transform().set_position(
        visii.vec3(
            pos_light[0]*2,
            pos_light[1]*2,
            pos_light[2]*2
        )
    )


# FLOOR - white or random

if cfg.add_table is True:
    entity_floor = visii.entity.create(
        name = "floor",
        mesh = visii.mesh.create_plane("mesh_floor"),
        transform = visii.transform.create("transform_floor"),
        material = visii.material.create("material_floor")
    )
    entity_floor.get_transform().set_scale(visii.vec3(cfg.table_size))
    mat = visii.material.get("material_floor")
    if cfg.table_color == 'white':
        # white rough floor
        mat.set_base_color(visii.vec3(1))
        mat.set_roughness(1)
    elif cfg.table_color == 'brushed':
        mat.set_base_color(visii.vec3(1))
        mat.set_roughness(0.2)
        mat.set_metallic(0.8)

    elif cfg.table_color == 'metallic':
        mat.set_base_color(visii.vec3(1))
        mat.set_roughness(0)
        mat.set_metallic(0.98)

    else:
        textures_floor = glob.glob(cfg.path_cco_textures + "*/")
        texture_random_selection = textures_floor[random.randint(0,len(textures_floor)-1)]

        entity_floor.set_material(material_from_cco(texture_random_selection,scale=cfg.table_texture_scale))
    if cfg.scene_type == "centered":
        print(bb_min)
        entity_floor.get_transform().add_position(visii.vec3(0,0,bb_min[2]))
    # export_names.append('floor')
    # cuboid_floor = add_cuboid('floor')
    # cuboids['floor'] = cuboid_floor
# # # # # # # # # # # # # # # # # # # # # # # # #


# RENDERING


# # # # # # # # # # # # # # # # # # # # # # # # #


i_frame = -1

if cfg.camera_type == 'fixed':
    # generate the position_to_render using the hemisphere code
    positions_to_render = sphere_renders(
                                nb_planes=cfg.camera_elevation_shots,
                                nb_circle=cfg.camera_theta_shots,
                                elevation_range = cfg.camera_elevation_range,
                                tetha_range = cfg.camera_theta_range
                            )
    look_at_trans = []

    for pos in positions_to_render:
        look_at_trans.append({
            'at': center_scene_model,
            'up': [0,0,1],
            'eye': [pos[0]*float(cfg.camera_fixed_distance_factor),
                    pos[1]*float(cfg.camera_fixed_distance_factor),
                    pos[2]*float(cfg.camera_fixed_distance_factor)]              
            })
    if "scaling_zoom" in cfg and cfg.scaling_zoom:
        # raise()
        look_at_trans = []
        for pos in positions_to_render:
            look_at_trans.append({
                'at': center_scene_model,
                'up': [0,0,1],
                'eye': [pos[0]*float(distance)*cfg.camera_fixed_distance_factor,
                        pos[1]*float(distance)*cfg.camera_fixed_distance_factor,
                        pos[2]*float(distance)*cfg.camera_fixed_distance_factor]              
                })
elif cfg.camera_type == 'random':
    positions_to_render = random_sample_sphere(
                                nb_frames = cfg.camera_nb_frames,
                                elevation_range = cfg.camera_elevation_range,
                                tetha_range = cfg.camera_theta_range
                            )
    look_at_trans = []
    for pos in positions_to_render:

        if cfg.camera_look_at == "random":
            # look inside the -1,-1,0 to 1,1,1 volume
            look_at = [ 
                np.random.uniform(low=-0.1,high=0.1),
                np.random.uniform(low=-0.1,high=0.1),
                np.random.uniform(low=0,high=0.1),
            ]
            # raise()
            # print(look_at)
        else:
            look_at = [0,0,0]
        look_at_trans.append({
            'at': look_at,
            'up': [0,0,1],
            'eye': [pos[0]*float(cfg.camera_fixed_distance_factor),
                    pos[1]*float(cfg.camera_fixed_distance_factor),
                    pos[2]*float(cfg.camera_fixed_distance_factor)]              
            })
                            
elif cfg.camera_type == 'free':
    positions_to_render = find_camera_positions_collision_free(
                            pybullet_ids,
                            nb_positions = cfg.camera_nb_frames,
                            volume = cfg.camera_fixed_distance_factor,
                            )
    # add more randomness with the look at.
    look_at_trans = []
    for pos in positions_to_render:
        # random object for look at
        at = entity_list[np.random.randint(
                0,
                len(entity_list)-1)
            ].get_transform().get_position()
        look_at_trans.append({
            'at': at,
            'up': [0,0,1],
            'eye': [pos[0]*float(cfg.camera_fixed_distance_factor),
                    pos[1]*float(cfg.camera_fixed_distance_factor),
                    pos[2]*float(cfg.camera_fixed_distance_factor)]              
            }
        )
elif cfg.camera_type == 'camera_movement':
    import torch
    # camera moving around a point with distances boundaries. 
    anchor_points = random_sample_sphere(
                            nb_frames = cfg.camera_nb_anchor,
                            elevation_range = cfg.camera_elevation_range,
                            tetha_range = cfg.camera_theta_range
                        )
    anchor_points[-1] = anchor_points[0]
    from Bezier import Bezier

    # from scipy.interpolate import interpn
    # t_points = np.arange(0, 1, 0.01) #................................. Creates an iterable list from 0 to 1.
    # points1 = np.array([[0, 0], [0, 8], [5, 10], [9, 7], [4, 3]]) #.... Creates an array of coordinates.
    # curve1 = Bezier.Curve(t_points, points1) #......................... Returns an array of coordinates.

    t_points = np.arange(0, 1, 1/cfg.camera_nb_frames) #................................. Creates an iterable list from 0 to 1.
    anchor_points = np.array(anchor_points)
    positions_to_render = Bezier.Curve(t_points, anchor_points) #......................... Returns an array of coordinates.


    # # pick an object for center
    at = entity_list[np.random.randint(
            0,
            len(entity_list))
        ].get_transform().get_position()

    # # interpolate between the positions 
    # i_interval = 0
    # size_interval = cfg.camera_nb_frames//(cfg.camera_nb_anchor-1)
    # positions_to_render = []
    # for i_photo in range(cfg.camera_nb_frames):
    #     # find which interval 
    #     if i_photo > (i_interval+1)*size_interval and i_interval<len(anchor_points)-2:
    #         i_interval+=1
    #     # find positions
    #     p0 = torch.tensor(anchor_points[i_interval])
    #     p1 = torch.tensor(anchor_points[i_interval+1])
    #     if not i_photo == 0:
    #         pos = (((i_interval+1)*size_interval)-i_photo)
    #         va = 1- (pos / size_interval)
    #     else:
    #         va = 0
    #     v = torch.lerp(p0,p1,va)
    #     # print(i_photo,va,v)
    #     positions_to_render.append(v.numpy())

    at = entity_list[np.random.randint(
        0,
        len(entity_list))
    ].get_transform().get_position()
    at = [at[0],at[1],at[2]]

    at_all = []
    for i_at in range(len(anchor_points)):
        if random.random() < 0.5:
            at = entity_list[np.random.randint(
                0,
                len(entity_list))
            ].get_transform().get_position()
            at = [at[0],at[1],at[2]]
        at_all.append(at)
    at_all[-1] = at_all[0]    
    at_all = np.array(at_all)
    at_all = Bezier.Curve(t_points, at_all) #......................... Returns an array of coordinates.


    # generate the camera poses 
    look_at_trans = []
    to_add = [0,0,0]
    if "to_add_position" in cfg:
        to_add = cfg.to_add_position
    for i_pos, pos in enumerate(positions_to_render):
        look_at_trans.append({
            'at': at_all[i_pos],
            'up': [0,0,1],
            'eye': [pos[0]*float(cfg.camera_fixed_distance_factor)+to_add[0],
                    pos[1]*float(cfg.camera_fixed_distance_factor)+to_add[1],
                    pos[2]*float(cfg.camera_fixed_distance_factor)+to_add[2]]              
            }
        )


if cfg.model_source == "cube":
    look_at_trans[0] = {
        'at': [0,0,0],
        'up': [0,0,1],
        'eye': [cfg.camera_fixed_distance_factor,0,0]
        }

# if cfg.model_source == 'lego':
#     export_names = []

if opt.interactive:
    # TODO
    # raise "no implemented yet"
    camera.get_transform().clear_motion()

    cursor = visii.vec4()
    speed_camera = 4.0
    rot = visii.vec2(visii.pi() * 1.25, 0)
    visii.register_pre_render_callback(interact)

    while True:
        pass

# main rendering loop
for i_trans,trans_look_at in enumerate(look_at_trans):
    i_frame +=1

    

    camera_struct = {
        'at': trans_look_at['at'],
        'up': trans_look_at['up'],
        'eye': trans_look_at['eye']
    }

    camera.get_transform().look_at(
        # visii.transform.get(entry['visii_id']).get_position(),
        at = visii.vec3(
            camera_struct['at'][0],
            camera_struct['at'][1],
            camera_struct['at'][2]
        ), # look at (world coordinate)
        up = visii.vec3(
            camera_struct['up'][0],
            camera_struct['up'][1],
            camera_struct['up'][2]
        ),
        eye = visii.vec3(
            camera_struct['eye'][0],
            camera_struct['eye'][1],
            camera_struct['eye'][2]
        )
    )

    if cfg.point_light and cfg.light_movement: 
        pos_light = light_positions[i_trans]
        light_entity.get_transform().set_position(
            visii.vec3(
                pos_light[0]*cfg.light_distance,
                pos_light[1]*cfg.light_distance,
                pos_light[2]*cfg.light_distance
            )
        )

    visii.sample_pixel_area(
        x_sample_interval = (.5,.5), 
        y_sample_interval = (.5,.5)
    )
    
    segmentation_array = visii.render_data(
        width=int(cfg.width), 
        height=int(cfg.height), 
        start_frame=0,
        frame_count=1,
        bounce=int(0),
        options="entity_id",
    )
    export_to_ndds_file(
        f"{cfg.outf}/{str(i_frame).zfill(5)}.json",
        obj_names = export_names,
        width = cfg.width,
        height = cfg.height,
        camera_name = 'camera',
        cuboids = cuboids,
        camera_struct = camera_struct,
        segmentation_mask = np.array(segmentation_array).reshape(cfg.height,cfg.width,4)[:,:,0],
        scene_aabb = scene_aabb,
    )
    depth_array = visii.render_data_to_file(
        width=int(cfg.width), 
        height=int(cfg.height), 
        start_frame=0,
        frame_count=1,
        bounce=int(0),
        options="depth",
        file_path=f"{cfg.outf}/{str(i_frame).zfill(5)}.depth.exr"
    )

    segmentation_array = visii.render_data_to_file(
        width=int(cfg.width), 
        height=int(cfg.height), 
        start_frame=0,
        frame_count=1,
        bounce=int(0),
        options="entity_id",
        file_path=f"{cfg.outf}/{str(i_frame).zfill(5)}.seg.exr"
    )

    visii.sample_pixel_area(
        x_sample_interval = (0,1), 
        y_sample_interval = (0,1)
    )

    print(f"{cfg.outf}/{str(i_frame).zfill(5)}.{cfg.file_format}")
    visii.render_to_file(
        width=int(cfg.width), 
        height=int(cfg.height), 
        samples_per_pixel=int(cfg.sample_per_pixel),
        file_path=f"{cfg.outf}/{str(i_frame).zfill(5)}.{cfg.file_format}"
    )

scene_names_to_point_cloud(export_names,cfg.outf)




# CHANGE THE OBJECTS TO BE RE RENDER. 

# positions_to_render = random_sample_sphere(
#                             nb_frames = cfg.testing_views,
#                             elevation_range = cfg.camera_elevation_range,
#                             tetha_range = cfg.camera_theta_range
#                         )
# look_at_trans = []
# for pos in positions_to_render:
#     look_at = [0,0,0]
#     look_at_trans.append({
#         'at': look_at,
#         'up': [0,0,1],
#         'eye': [pos[0]*float(cfg.camera_fixed_distance_factor),
#                 pos[1]*float(cfg.camera_fixed_distance_factor),
#                 pos[2]*float(cfg.camera_fixed_distance_factor)]              
#         })




# for i_trans,trans_look_at in enumerate(look_at_trans):
#     i_frame +=1

#     camera_struct = {
#         'at': trans_look_at['at'],
#         'up': trans_look_at['up'],
#         'eye': trans_look_at['eye']
#     }

#     camera.get_transform().look_at(
#         # visii.transform.get(entry['visii_id']).get_position(),
#         at = visii.vec3(
#             camera_struct['at'][0],
#             camera_struct['at'][1],
#             camera_struct['at'][2]
#         ), # look at (world coordinate)
#         up = visii.vec3(
#             camera_struct['up'][0],
#             camera_struct['up'][1],
#             camera_struct['up'][2]
#         ),
#         eye = visii.vec3(
#             camera_struct['eye'][0],
#             camera_struct['eye'][1],
#             camera_struct['eye'][2]
#         )
#     )

#     if cfg.point_light and cfg.light_movement: 
#         pos_light = light_positions[i_trans]
#         light_entity.get_transform().set_position(
#             visii.vec3(
#                 pos_light[0]*cfg.light_distance,
#                 pos_light[1]*cfg.light_distance,
#                 pos_light[2]*cfg.light_distance
#             )
#         )

#     visii.sample_pixel_area(
#         x_sample_interval = (.5,.5), 
#         y_sample_interval = (.5,.5)
#     )
    
#     segmentation_array = visii.render_data(
#         width=int(cfg.width), 
#         height=int(cfg.height), 
#         start_frame=0,
#         frame_count=1,
#         bounce=int(0),
#         options="entity_id",
#     )
#     export_to_ndds_file(
#         f"{cfg.outf}/{str(i_frame).zfill(5)}.json",
#         obj_names = export_names,
#         width = cfg.width,
#         height = cfg.height,
#         camera_name = 'camera',
#         cuboids = cuboids,
#         camera_struct = camera_struct,
#         segmentation_mask = np.array(segmentation_array).reshape(cfg.height,cfg.width,4)[:,:,0],
#         scene_aabb = scene_aabb,
#     )
#     depth_array = visii.render_data_to_file(
#         width=int(cfg.width), 
#         height=int(cfg.height), 
#         start_frame=0,
#         frame_count=1,
#         bounce=int(0),
#         options="depth",
#         file_path=f"{cfg.outf}/{str(i_frame).zfill(5)}.depth.exr"
#     )

#     segmentation_array = visii.render_data_to_file(
#         width=int(cfg.width), 
#         height=int(cfg.height), 
#         start_frame=0,
#         frame_count=1,
#         bounce=int(0),
#         options="entity_id",
#         file_path=f"{cfg.outf}/{str(i_frame).zfill(5)}.seg.exr"
#     )

#     visii.sample_pixel_area(
#         x_sample_interval = (0,1), 
#         y_sample_interval = (0,1)
#     )

#     print(f"{cfg.outf}/{str(i_frame).zfill(5)}.{cfg.file_format}")
#     visii.render_to_file(
#         width=int(cfg.width), 
#         height=int(cfg.height), 
#         samples_per_pixel=int(cfg.sample_per_pixel),
#         file_path=f"{cfg.outf}/{str(i_frame).zfill(5)}.{cfg.file_format}"
#     )






print('over')

visii.deinitialize()


