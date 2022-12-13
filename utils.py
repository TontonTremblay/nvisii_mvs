import random 
import nvisii as visii
# import visii
import randomcolor
import math 
import colorsys
import glob 
import os 
import numpy as np

class DotDict(dict):
    """Access dictionary elements using dot notation"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def add_random_obj(name = "name",
    x_lim = [-1,1],
    y_lim = [-1,1],
    z_lim = [-1,1],
    scale_lim = [0.01,1],
    obj_id = None,
    ):
    
    # obj = visii.entity.get(name)
    # if obj is None:
    obj= visii.entity.create(
        name = name,
        transform = visii.transform.create(name),
        material = visii.material.create(name)
    )
    if obj_id is None:
        obj_id = random.randint(0,15)

    mesh = None
    if obj_id == 0:
        if add_random_obj.create_sphere is None:
            add_random_obj.create_sphere = visii.mesh.create_sphere(name) 
        mesh = add_random_obj.create_sphere
    if obj_id == 1:
        if add_random_obj.create_torus_knot is None:
            add_random_obj.create_torus_knot = visii.mesh.create_torus_knot(name, 
                random.randint(2,6),
                random.randint(4,10))
        mesh = add_random_obj.create_torus_knot
    if obj_id == 2:
        if add_random_obj.create_teapotahedron is None:
            add_random_obj.create_teapotahedron = visii.mesh.create_teapotahedron(name) 
        mesh = add_random_obj.create_teapotahedron
    if obj_id == 3:
        if add_random_obj.create_box is None:
            add_random_obj.create_box = visii.mesh.create_box(name)
             
        mesh = add_random_obj.create_box
    if obj_id == 4:
        if add_random_obj.create_capped_cone is None:
            add_random_obj.create_capped_cone = visii.mesh.create_capped_cone(name) 
        mesh = add_random_obj.create_capped_cone
    if obj_id == 5:
        if add_random_obj.create_capped_cylinder is None:
            add_random_obj.create_capped_cylinder = visii.mesh.create_capped_cylinder(name) 
        mesh = add_random_obj.create_capped_cylinder
    if obj_id == 6:
        if add_random_obj.create_capsule is None:
            add_random_obj.create_capsule = visii.mesh.create_capsule(name) 
        mesh = add_random_obj.create_capsule
    if obj_id == 7:
        if add_random_obj.create_cylinder is None:
            add_random_obj.create_cylinder = visii.mesh.create_cylinder(name) 
        mesh = add_random_obj.create_cylinder
    if obj_id == 8:
        if add_random_obj.create_disk is None:
            add_random_obj.create_disk = visii.mesh.create_disk(name) 
        mesh = add_random_obj.create_disk
    if obj_id == 9:
        if add_random_obj.create_dodecahedron is None:
            add_random_obj.create_dodecahedron = visii.mesh.create_dodecahedron(name) 
        mesh = add_random_obj.create_dodecahedron
    if obj_id == 10:
        if add_random_obj.create_icosahedron is None:
            add_random_obj.create_icosahedron = visii.mesh.create_icosahedron(name) 
        mesh = add_random_obj.create_icosahedron
    if obj_id == 11:
        if add_random_obj.create_icosphere is None:
            add_random_obj.create_icosphere = visii.mesh.create_icosphere(name) 
        mesh = add_random_obj.create_icosphere
    if obj_id == 12:
        if add_random_obj.create_rounded_box is None:
            add_random_obj.create_rounded_box = visii.mesh.create_rounded_box(name) 
        mesh = add_random_obj.create_rounded_box
    if obj_id == 13:
        if add_random_obj.create_spring is None:
            add_random_obj.create_spring = visii.mesh.create_spring(name) 
        mesh = add_random_obj.create_spring
    if obj_id == 14:
        if add_random_obj.create_torus is None:
            add_random_obj.create_torus = visii.mesh.create_torus(name) 
        mesh = add_random_obj.create_torus
    if obj_id == 15:
        if add_random_obj.create_tube is None:
            add_random_obj.create_tube = visii.mesh.create_tube(name) 
        mesh = add_random_obj.create_tube
    if obj_id == 16:
        if add_random_obj.create_surface is None:
            add_random_obj.create_surface = visii.mesh.create_plane(name) 
        mesh = add_random_obj.create_tube

    obj.set_mesh(mesh)

    obj.get_transform().set_position(
        visii.vec3(
            random.uniform(x_lim[0],x_lim[1]),
            random.uniform(y_lim[0],y_lim[1]),
            random.uniform(z_lim[0],z_lim[1])        
        )
    )
    
    obj.get_transform().set_rotation(
        visii.quat(1.0 ,random.random(), random.random(), random.random()) 
    )    

    obj.get_transform().set_scale(
        visii.vec3(
            random.uniform(scale_lim[0],scale_lim[1])
        )
    )
    return obj
add_random_obj.rcolor = randomcolor.RandomColor()
add_random_obj.create_sphere = None
add_random_obj.create_torus_knot = None
add_random_obj.create_teapotahedron = None
add_random_obj.create_box = None
add_random_obj.create_capped_cone = None
add_random_obj.create_capped_cylinder = None
add_random_obj.create_capsule = None
add_random_obj.create_cylinder = None
add_random_obj.create_disk = None
add_random_obj.create_dodecahedron = None
add_random_obj.create_icosahedron = None
add_random_obj.create_icosphere = None
add_random_obj.create_rounded_box = None
add_random_obj.create_spring = None
add_random_obj.create_torus = None
add_random_obj.create_tube = None



def random_material(
    obj_id,
    color = None, # list of 3 numbers between [0..1]
    just_simple = False,
    ):

    if random_material.textures is None:
        textures = []

        path = 'content/materials_omniverse/'
        for folder in glob.glob(path + "/*/"):
            for folder_in in glob.glob(folder+"/*/"):
                name = folder_in.replace(folder,'').replace('/','').replace('\\', '')
                # print (folder_in)
                # print (name)
                # print (folder_in + "/" + name + "_BaseColor.png")
                if os.path.exists(folder_in + "/" + name + "_BaseColor.png"):
                    if os.path.exists(folder_in + "/" + name + "_Normal.png"):
                        normal = folder_in + "/" + name + "_Normal.png"
                    else:
                        normal = folder_in + "/" + name + "_N.png"

                    textures.append({
                        'base':folder_in + "/" + name + "_BaseColor.png",
                        'normal':folder_in + "/" + name + "_Normal.png",
                        'orm':folder_in + "/" + name + "_ORM.png",
                        })

            if 'glass' in folder.lower():
                #read the mtl 
                print("---")
                for mdl in glob.glob(folder + "/*.mdl"):    
                    
                    with open(mdl,'r') as f:
                        data = {'glass':1}
                        for line in f.readlines():
                            if "transmission_color" in line and not 'transmission_color_texture' in line:
                                text = line.replace("transmission_color","")\
                                    .replace(" ","").replace('color','').replace(":","")\
                                    .replace("f",'')
                                text = eval(text)[0]
                                # print(text)
                                data['color'] = text
                            if 'roughness' in line and not 'roughness_texture_influence' in line\
                                and not "frosting_roughness" in line and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                # print(line)
                                text = line.replace("roughness","")\
                                    .replace(" ","").replace('color','').replace(":","")\
                                    .replace("f",'').replace(',','')
                                # print(text)
                                text = eval(text)
                                data['roughness'] = text
                            if 'frosting_roughness' in line and not 'roughness_texture_influence' in line\
                                and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                # print(line)
                                text = line.replace("frosting_roughness","")\
                                    .replace(" ","").replace('color','').replace(":","")\
                                    .replace("f",'').replace(',','')
                                # print(text)
                                text = eval(text)

                                data['transmisive_roughness'] = text
                            if 'ior' in line and not 'stop' in line\
                                and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                text = line.replace("ior","")\
                                    .replace(" ","").replace('glass_ior','').replace(":","")\
                                    .replace("f",'').replace(',','').replace('glass','')\
                                    .replace("_",'')
                                text = eval(text)

                                data['ior'] = text
                            if 'specular_level' in line and not 'stop' in line\
                                and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                # print(text)
                                text = line.replace("specular_level","")\
                                    .replace(" ","").replace('glass_ior','').replace(":","")\
                                    .replace("f",'').replace(',','').replace('glass','')\
                                    .replace("_",'')
                                text = eval(text)
                                # print(text)
                                data['specular'] = text

                            if 'metallic_constant' in line and not 'stop' in line\
                                and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                text = line.replace("metallic_constant","")\
                                    .replace(" ","").replace('glass_ior','').replace(":","")\
                                    .replace("f",'').replace(',','').replace('glass','')\
                                    .replace("_",'')
                                text = eval(text)
                                data['metallic'] = text
                                
                        textures.append(data)
        random_material.textures = textures

    # obj_mat = visii.material.get(str(obj_id))
    
    obj_mat = obj_id.get_material()

    if color is None: 
        rgb = colorsys.hsv_to_rgb(
            random.uniform(0,1),
            random.uniform(0.4,1),
            random.uniform(0.5,1)
        )

        obj_mat.set_base_color(
            visii.vec3(
                rgb[0],
                rgb[1],
                rgb[2],
            )
        )          
    else:
        obj_mat.set_base_color(
            visii.vec3(
                color[0],color[1],color[2]
            )  
        )

    if just_simple is False:
        texture = random_material.textures[random.randint(0,len(random_material.textures)-1)]

    if random.uniform(0,1) < 0.25 or just_simple is True:
        r = random.randint(0,2)
        r = random.uniform(0,1)
        if r >=0.3:  
            # Plastic / mat
            obj_mat.set_metallic(0)  # should 0 or 1      
            obj_mat.set_transmission(0)  # should 0 or 1      
            obj_mat.set_roughness(random.uniform(0,1)) # default is 1  
        

        if r<0.3:  
            # metallic
            obj_mat.set_metallic(random.uniform(0.9,1))  # should 0 or 1      
            obj_mat.set_transmission(0)  # should 0 or 1      
            obj_mat.set_roughness(random.uniform(0,.1)) # default is 1  

        if r>0.3:  
            obj_mat.set_sheen(random.uniform(0,0.5))  # degault is 0     
            obj_mat.set_clearcoat(random.uniform(0,0.5))  # degault is 0     
            obj_mat.set_specular(random.uniform(0,0.5))  # degault is 0     

            r = random.randint(0,1)
            if r == 0:
                obj_mat.set_anisotropic(random.uniform(0,0.1))  # degault is 0     
            else:
                obj_mat.set_anisotropic(random.uniform(0.9,1))  # degault is 0     

        # if r<0.01:
        #     obj_mat.set_metallic(0)  # should 0 or 1      
        #     obj_mat.set_transmission(random.uniform(0.9,1))  # should 0 or 1      

        # else: 
        #     # glass

        # if r <0.3: # for metallic and glass
        #     r2 = random.randint(0,1)
        #     if r2 == 1: 
        #         obj_mat.set_roughness(random.uniform(0,.1)) # default is 1  
        #     else:
        #         obj_mat.set_roughness(random.uniform(0.9,1)) # default is 1  




    elif 'glass' in texture:
        obj_mat.set_transmission(1)
        if 'metallic' in texture.keys(): 
            obj_mat.set_transmission(0)
            obj_mat.set_metallic(texture['metallic'])
        if 'specular' in texture.keys():
            obj_mat.set_specular(texture['specular'])
        if 'transmisive_roughness' in texture.keys():
            obj_mat.set_transmission_roughness(texture['transmisive_roughness'])
        if 'roughness' in texture.keys(): 
            obj_mat.set_roughness(texture['roughness'])
        if 'color' in texture.keys():
            obj_mat.set_base_color(visii.vec3(
                texture['color'][0],
                texture['color'][1],
                texture['color'][2],
            ))

    else:
        # print(texture['base'])
        tex = visii.texture.get(texture['base'])
        tex_r = visii.texture.get(texture['base']+"_r")
        tex_m = visii.texture.get(texture['base']+"_m")
        tex_n = visii.texture.get(texture['base']+"_n")
        # tex_n = None
        
        if tex is None:
            tex = visii.texture.create_from_image(texture['base'],texture['base'])
            
            # load metallic and roughness 
            from PIL import Image
            im = np.array(Image.open(texture['orm']))            
            
            im_r = np.concatenate(
                [
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1)
                ],
                2
            ) 
            # im_r = Image.fromarray(im_r)
            im_r = (im_r/255.0)

            tex_r = visii.texture.create_from_data(texture['base']+'_r',
                im_r.shape[0],
                im_r.shape[1],
                im_r.reshape(im_r.shape[0]*im_r.shape[1],4).astype(np.float32).flatten().tolist()
            )

            # im_r.save('tmp.png')
            # tex_r = visii.texture.create_from_image(texture['base']+'_r',"tmp.png")

            im_m = np.concatenate(
                [
                    im[:,:,2].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,2].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,2].reshape(im.shape[0],im.shape[1],1)
                ],
                2
            ) 
            im_m = Image.fromarray(im_m)
            im_m.save('tmp.png')

            tex_m = visii.texture.create_from_image(texture['base']+'_m',"tmp.png")

            if os.path.exists(texture['normal']):
                tex_n = visii.texture.create_from_image(
                    texture['base']+"_n",
                    texture['normal'],
                    linear=True
                )
            else:
                tex_n = None
        obj_mat.set_base_color_texture(tex)
        obj_mat.set_metallic_texture(tex_m)
        obj_mat.set_roughness_texture(tex_r)
        if not tex_n is None:
            obj_mat.set_normal_map_texture(tex_n)


####


random_material.textures = None

########################################
# ANIMATION RANDOMIZATION 
########################################



def distance(v0,v1=[0,0,0]):
    l2 = 0
    try:
        for i in range(len(v0)):
            l2 += (v0[i]-v1[i])**2
    except:
        for i in range(3):
            l2 += (v0[i]-v1[i])**2
    return math.sqrt(l2)

def normalize(v):
    l2 = distance(v)
    return [v[0]/l2,v[1]/l2,v[2]/l2]

def random_translation(obj_id,
    x_lim = [-1,1],
    y_lim = [-1,1],
    z_lim = [-1,1],
    speed_lim = [0.01,0.05],
    sample_method = None,
    motion_blur = False,
    strenght = 1,
    ):
    # return
    trans = visii.transform.get(str(obj_id))

    # Position    
    if not str(obj_id) in random_translation.destinations.keys() :
        if sample_method is None:
            random_translation.destinations[str(obj_id)] = [
                random.uniform(x_lim[0],x_lim[1]),
                random.uniform(y_lim[0],y_lim[1]),
                random.uniform(z_lim[0],z_lim[1])
            ]
        else:
            random_translation.destinations[str(obj_id)] = sample_method()
        random_translation.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
    else:
        goal = random_translation.destinations[str(obj_id)]
        pos = trans.get_position()

        if distance(goal,pos) < min(speed_lim)*2:
            if sample_method is None:
                random_translation.destinations[str(obj_id)] = [
                    random.uniform(x_lim[0],x_lim[1]),
                    random.uniform(y_lim[0],y_lim[1]),
                    random.uniform(z_lim[0],z_lim[1])
                ]
            else:
                random_translation.destinations[str(obj_id)] = sample_method()            

            random_translation.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])    
            goal = random_translation.destinations[str(obj_id)]

        dir_vec = normalize(
            [
                goal[0] - pos[0],
                goal[1] - pos[1],
                goal[2] - pos[2]
            ]   
        )
        
        trans.add_position(
            visii.vec3(
                dir_vec[0] * random_translation.speeds[str(obj_id)],
                dir_vec[1] * random_translation.speeds[str(obj_id)],
                dir_vec[2] * random_translation.speeds[str(obj_id)]
            )
        )
        if motion_blur:
            trans.set_linear_velocity(
                visii.vec3(

                    dir_vec[0] * random_translation.speeds[str(obj_id)] * strenght,
                    dir_vec[1] * random_translation.speeds[str(obj_id)] * strenght,
                    dir_vec[2] * random_translation.speeds[str(obj_id)] * strenght
                )
            )

random_translation.destinations = {}
random_translation.speeds = {}


def circle_path(camera_distance = 0.5, camera_height=0.3):
    global opt 

    if circle_path.locations is None:
        circle_path.locations = []
        for i in range(0,210,10):
            circle_path.locations.append([ 
                math.cos(i*0.01745) * camera_distance,
                math.sin(i*0.01745) * camera_distance,
                camera_height
            ])
        circle_path.last = circle_path.locations[-1]
    if len(circle_path.locations) == 0:
        return circle_path.last 

    pos = circle_path.locations[0]
    circle_path.locations = circle_path.locations[1:]
    return pos

circle_path.locations = None

def random_sample_hemispher(
    max_radius = 2.5, 
    min_radius = 0.2
    ):
    if min_radius > max_radius:
        print('min bigger than max')
    outside = True
    while outside:

        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        z = random.uniform( 0, 0.95)
        if  (x**2 + y**2 + z**2) * max_radius < max_radius \
        and (x**2 + y**2 + z**2) * max_radius > min_radius:
            outside = False

    return [x*max_radius,y*max_radius,z*max_radius]



def random_rotation(obj_id,
    speed_lim = [0.01,0.05],
    motion_blur = False,
    strenght = 1,
    ):
    # return
    from pyquaternion import Quaternion

    trans = visii.transform.get(str(obj_id))    

    # Rotation
    if not str(obj_id) in random_rotation.destinations.keys() :
        random_rotation.destinations[str(obj_id)] = Quaternion.random()
        random_rotation.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])

    else:
        goal = random_rotation.destinations[str(obj_id)]
        rot = trans.get_rotation()
        rot = Quaternion(w=rot.w,x=rot.x,y=rot.y,z=rot.z)
        if Quaternion.sym_distance(goal, rot) < 0.1:
            random_rotation.destinations[str(obj_id)] = Quaternion.random()    
            random_rotation.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
            goal = random_rotation.destinations[str(obj_id)]
        dir_vec = Quaternion.slerp(rot,goal,random_rotation.speeds[str(obj_id)])
        q = visii.quat()
        q.w,q.x,q.y,q.z = dir_vec.w,dir_vec.x,dir_vec.y,dir_vec.z
        trans.set_rotation(q)
        # if motion_blur: 
        #     dir_vec = Quaternion.slerp(rot,goal,random_rotation.speeds[str(obj_id)] * strenght)

        #     q.w,q.x,q.y,q.z = dir_vec.w,dir_vec.x,dir_vec.y,dir_vec.z
        #     trans.set_angular_velocity(q)

random_rotation.destinations = {}
random_rotation.speeds = {}

def random_scale(obj_id,
    scale_lim = [0.01,0.2],
    speed_lim = [0.01,0.02],
    x_lim = None,
    y_lim = None,
    z_lim = None,
    motion_blur = False,
    strenght = 1,
    ):
    # return
    # This assumes only one dimensions gets scale

    trans = visii.transform.get(str(obj_id))    

    limit = min(speed_lim)*2
    
    if not x_lim is None:
        limit = [min(y_lim)*2,min(x_lim)*2,min(z_lim)*2]
    
    if not str(obj_id) in random_scale.destinations.keys() :
        if not x_lim is None:
            random_scale.destinations[str(obj_id)] = [
                                        random.uniform(x_lim[0],x_lim[1]),
                                        random.uniform(y_lim[0],y_lim[1]),
                                        random.uniform(z_lim[0],z_lim[1])
                                        ]
        else:    
            random_scale.destinations[str(obj_id)] = random.uniform(scale_lim[0],scale_lim[1])

        random_scale.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])

    else:
        goal = random_scale.destinations[str(obj_id)]

        if x_lim is None:
            current = trans.get_scale()[0]
        else:
            current = trans.get_scale()
        
        if x_lim is None:

            if abs(goal-current) < limit:
                random_scale.destinations[str(obj_id)] = random.uniform(scale_lim[0],scale_lim[1])
                random_scale.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
                goal = random_scale.destinations[str(obj_id)]
            if goal>current:
                q = random_scale.speeds[str(obj_id)]
            else:
                q = -random_scale.speeds[str(obj_id)]
            trans.set_scale(current + q)

        else:
            limits  = [x_lim,y_lim,z_lim]
            q = [0,0,0]
            for i in range(3):
                if abs(goal[i]-current[i]) < limit[i]:
                    random_scale.destinations[str(obj_id)][i] = random.uniform(limits[i][0],limits[i][1])
                    random_scale.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])                    
                    goal = random_scale.destinations[str(obj_id)]
                if goal[i]>current[i]:
                    q[i] = random_scale.speeds[str(obj_id)]
                else:
                    q[i] = -random_scale.speeds[str(obj_id)]
            trans.set_scale(current + visii.vec3(q[0],q[1],q[2]))
        # if motion_blur:
        #     trans.set_scalar_velocity(visii.vec3(q[0]*strenght,q[1]*strenght,q[2]*strenght))

random_scale.destinations = {}
random_scale.speeds = {}


def random_color(obj_id,
    speed_lim = [0.01,0.1]
    ):

    # color
    if not str(obj_id) in random_color.destinations.keys() :
        c = eval(str(random_color.rcolor.generate(luminosity='bright',format_='rgb')[0])[3:])
        random_color.destinations[str(obj_id)] = visii.vec3(c[0]/255.0, c[1]/255.0, c[2]/255.0)
        random_color.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])

    else:
        goal = random_color.destinations[str(obj_id)]
        current = visii.material.get(str(obj_id)).get_base_color()

        if distance(goal,current) < 0.1:
            random_color.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
            c = eval(str(random_color.rcolor.generate(luminosity='bright',format_='rgb')[0])[3:])
            random_color.destinations[str(obj_id)] = visii.vec3(c[0]/255.0, c[1]/255.0, c[2]/255.0)
            goal = random_color.destinations[str(obj_id)]

        target = visii.mix(current,goal,
            visii.vec3( random_color.speeds[str(obj_id)],
                        random_color.speeds[str(obj_id)],
                        random_color.speeds[str(obj_id)]
                        )
        ) 

        visii.material.get(str(obj_id)).set_base_color(target)

random_color.destinations = {}
random_color.speeds = {}
random_color.rcolor = randomcolor.RandomColor()

######## RANDOM LIGHTS ############

def random_light(obj_id,
    intensity_lim = [5000,10000],
    color = None,
    temperature_lim  = [10,10000],
    exposure_lim = [0,0],
    ):

    obj = visii.entity.get(str(obj_id))
    obj.set_light(visii.light.create(str(obj_id)))

    obj.get_light().set_intensity(random.uniform(intensity_lim[0],intensity_lim[1]))
    obj.get_light().set_exposure(random.uniform(exposure_lim[0],exposure_lim[1]))
    # obj.get_light().set_temperature(np.random.randint(100,9000))


    if not color is None:
        obj.get_material().set_base_color(color[0],color[1],color[2])  
        # c = eval(str(rcolor.generate(luminosity='bright',format_='rgb')[0])[3:])
        # obj.get_light().set_color(
        #     c[0]/255.0,
        #     c[1]/255.0,
        #     c[2]/255.0)  
       
    else:
        obj.get_light().set_temperature(random.uniform(temperature_lim[0],temperature_lim[1]))

def random_intensity(obj_id,
    intensity_lim = [5000,10000],
    speed_lim = [0.1,1]
    ):

    obj = visii.entity.get(str(obj_id)).get_light()

    if not str(obj_id) in random_intensity.destinations.keys() :
        random_intensity.destinations[str(obj_id)] = random.uniform(intensity_lim[0],intensity_lim[1])
        random_intensity.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
        random_intensity.current[str(obj_id)] = random_intensity.destinations[str(obj_id)]
        obj.set_intensity(random_intensity.current[str(obj_id)]) 
    else:
        goal = random_intensity.destinations[str(obj_id)]
        current = random_intensity.current[str(obj_id)]
        
        if abs(goal-current) < min(speed_lim)*2:
            random_intensity.destinations[str(obj_id)] = random.uniform(intensity_lim[0],intensity_lim[1])
            random_intensity.speeds[str(obj_id)] = random.uniform(speed_lim[0],speed_lim[1])
            goal = random_intensity.destinations[str(obj_id)]
        if goal>current:
            q = random_intensity.speeds[str(obj_id)]
        else:
            q = -random_intensity.speeds[str(obj_id)]
        obj.set_intensity(random_intensity.current[str(obj_id)] + q)
        random_intensity.current[str(obj_id)] = random_intensity.current[str(obj_id)] + q

random_intensity.destinations = {}
random_intensity.current = {}
random_intensity.speeds = {}


def random_texture_material(entity):
    # select a random texture
    textures = random_texture_material.textures
    if len(textures) == 0: 
        "load the textures"
        path = 'content/materials_omniverse/'
        import glob, os
        for folder in glob.glob(path + "/*/"):
            for folder_in in glob.glob(folder+"/*/"):
                name = folder_in.replace(folder,'').replace('/','').replace('\\', '')
                # print (folder_in)
                # print (name)
                # print (folder_in + "/" + name + "_BaseColor.png")
                if os.path.exists(folder_in + "/" + name + "_BaseColor.png"):
                    if os.path.exists(folder_in + "/" + name + "_Normal.png"):
                        normal = folder_in + "/" + name + "_Normal.png"
                    else:
                        normal = folder_in + "/" + name + "_N.png"

                    textures.append({
                        'base':folder_in + "/" + name + "_BaseColor.png",
                        'normal':folder_in + "/" + name + "_Normal.png",
                        'orm':folder_in + "/" + name + "_ORM.png",
                        })

            if 'glass' in folder.lower():
                #read the mtl 
                # print("---")
                for mdl in glob.glob(folder + "/*.mdl"):    
                    
                    with open(mdl,'r') as f:
                        data = {'glass':1}
                        for line in f.readlines():
                            if "transmission_color" in line and not 'transmission_color_texture' in line:
                                text = line.replace("transmission_color","")\
                                    .replace(" ","").replace('color','').replace(":","")\
                                    .replace("f",'')
                                text = eval(text)[0]
                                # print(text)
                                data['color'] = text
                            if 'roughness' in line and not 'roughness_texture_influence' in line\
                                and not "frosting_roughness" in line and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                # print(line)
                                text = line.replace("roughness","")\
                                    .replace(" ","").replace('color','').replace(":","")\
                                    .replace("f",'').replace(',','')
                                # print(text)
                                text = eval(text)
                                data['roughness'] = text
                            if 'frosting_roughness' in line and not 'roughness_texture_influence' in line\
                                and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                # print(line)
                                text = line.replace("frosting_roughness","")\
                                    .replace(" ","").replace('color','').replace(":","")\
                                    .replace("f",'').replace(',','')
                                # print(text)
                                text = eval(text)

                                data['transmisive_roughness'] = text
                            if 'ior' in line and not 'stop' in line\
                                and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                text = line.replace("ior","")\
                                    .replace(" ","").replace('glass_ior','').replace(":","")\
                                    .replace("f",'').replace(',','').replace('glass','')\
                                    .replace("_",'')
                                text = eval(text)

                                data['ior'] = text
                            if 'specular_level' in line and not 'stop' in line\
                                and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                # print(text)
                                text = line.replace("specular_level","")\
                                    .replace(" ","").replace('glass_ior','').replace(":","")\
                                    .replace("f",'').replace(',','').replace('glass','')\
                                    .replace("_",'')
                                text = eval(text)
                                # print(text)
                                data['specular'] = text

                            if 'metallic_constant' in line and not 'stop' in line\
                                and not 'roughness_texture' in line\
                                and not "reflection_roughness_constant" in line:
                                # print(text)
                                text = line.replace("metallic_constant","")\
                                    .replace(" ","").replace('glass_ior','').replace(":","")\
                                    .replace("f",'').replace(',','').replace('glass','')\
                                    .replace("_",'')
                                text = eval(text)
                                
                                data['metallic'] = text
                                
                        textures.append(data)



    texture = random_texture_material.textures[random.randint(0,len(textures)-1)]
    if random.uniform(0,1) < 0.25:
        random_material(entity)

    elif 'glass' in texture:

        mat = visii.material.get(entity)

        mat.set_transmission(1)
        if 'metallic' in texture.keys(): 
            mat.set_transmission(0)
            mat.set_metallic(texture['metallic'])
        if 'specular' in texture.keys():
            mat.set_specular(texture['specular'])
        if 'transmisive_roughness' in texture.keys():
            mat.set_transmission_roughness(texture['transmisive_roughness'])
        if 'roughness' in texture.keys(): 
            mat.set_roughness(texture['roughness'])
        if 'color' in texture.keys():
            mat.set_base_color(visii.vec3(
                texture['color'][0],
                texture['color'][1],
                texture['color'][2],
            ))

    else:
        import numpy as np
        print(texture['base'])
        tex = visii.texture.get(texture['base'])
        tex_r = visii.texture.get(texture['base']+"_r")
        tex_m = visii.texture.get(texture['base']+"_m")

        if tex is None:
            tex = visii.texture.create_from_image(texture['base'],texture['base'])
            
            # load metallic and roughness 
            from PIL import Image
            im = np.array(Image.open(texture['orm']))            
            
            im_r = np.concatenate(
                [
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1)
                ],
                2
            ) 
            # im_r = Image.fromarray(im_r)
            im_r = (im_r/255.0)

            tex_r = visii.texture.create_from_data(texture['base']+'_r',
                im_r.shape[0],
                im_r.shape[1],
                im_r.reshape(im_r.shape[0]*im_r.shape[1],4).astype(np.float32).flatten().tolist()
            )

            # im_r.save('tmp.png')
            # tex_r = visii.texture.create_from_image(texture['base']+'_r',"tmp.png")

            im_m = np.concatenate(
                [
                    im[:,:,2].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,2].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,2].reshape(im.shape[0],im.shape[1],1)
                ],
                2
            ) 
            im_m = Image.fromarray(im_m)
            im_m.save('tmp.png')
            tex_m = visii.texture.create_from_image(texture['base']+'_m',"tmp.png")
            import os
            if os.path.exists(texture['normal']):
                tex_n = visii.texture.create_from_image(
                    texture['normal'],
                    texture['normal'],
                    linear=True
                )
            else:
                tex_n = None
        visii.material.get(entity).set_base_color_texture(tex)
        visii.material.get(entity).set_metallic_texture(tex_m)
        visii.material.get(entity).set_roughness_texture(tex_r)
        if not tex_n is None:
            visii.material.get(entity).set_normal_map_texture(tex_n)


random_texture_material.textures = []



######## NDDS ##########

def add_cuboid(name, debug=False):
    obj = visii.entity.get(name)

    min_obj = obj.get_mesh().get_min_aabb_corner()
    max_obj = obj.get_mesh().get_max_aabb_corner()
    centroid_obj = obj.get_mesh().get_aabb_center()


    cuboid = [
        visii.vec3(max_obj[0], max_obj[1], max_obj[2]),
        visii.vec3(min_obj[0], max_obj[1], max_obj[2]),
        visii.vec3(max_obj[0], min_obj[1], max_obj[2]),
        visii.vec3(max_obj[0], max_obj[1], min_obj[2]),
        visii.vec3(min_obj[0], min_obj[1], max_obj[2]),
        visii.vec3(max_obj[0], min_obj[1], min_obj[2]),
        visii.vec3(min_obj[0], max_obj[1], min_obj[2]),
        visii.vec3(min_obj[0], min_obj[1], min_obj[2]),
        visii.vec3(centroid_obj[0], centroid_obj[1], centroid_obj[2]), 
    ]

    # change the ids to be like ndds / DOPE
    cuboid = [  cuboid[2],cuboid[0],cuboid[3],
                cuboid[5],cuboid[4],cuboid[1],
                cuboid[6],cuboid[7],cuboid[-1]]

    cuboid.append(visii.vec3(centroid_obj[0], centroid_obj[1], centroid_obj[2]))
        
    for i_p, p in enumerate(cuboid):
        # print(f"{name}_cuboid_{i_p}")
        child_transform = visii.transform.create(f"{name}_cuboid_{i_p}")
        child_transform.set_position(p)
        child_transform.set_scale(visii.vec3(0.01))
        child_transform.set_parent(obj.get_transform())
        if debug: 
            visii.entity.create(
                name = f"{name}_cuboid_{i_p}",
                mesh = visii.mesh.create_sphere(f"{name}_cuboid_{i_p}"),
                transform = child_transform, 
                material = visii.material.create(f"{name}_cuboid_{i_p}")
            )
            
    for i_v, v in enumerate(cuboid):
        cuboid[i_v]=[v[0], v[1], v[2]]

    print(cuboid)
    return cuboid

def get_cuboid_image_space(obj_id, camera_name = 'my_camera'):
    # return cubdoid + centroid projected to the image, values [0..1]

    cam_matrix = visii.entity.get(camera_name).get_transform().get_world_to_local_matrix()
    cam_proj_matrix = visii.entity.get(camera_name).get_camera().get_projection()

    points = []
    points_cam = []

    # check if the data exists
    trans = visii.transform.get(f"{obj_id}_cuboid_0")
    
    if trans is None:
        return points, points_cam

    for i_t in range(9):
        trans = visii.transform.get(f"{obj_id}_cuboid_{i_t}")
        mat_trans = trans.get_local_to_world_matrix()
        pos_m = visii.vec4(
            mat_trans[3][0],
            mat_trans[3][1],
            mat_trans[3][2],
            1)
        
        p_cam = cam_matrix * pos_m 

        p_image = cam_proj_matrix * (cam_matrix * pos_m) 
        p_image = visii.vec2(p_image) / p_image.w
        p_image = p_image * visii.vec2(1,-1)
        p_image = (p_image + visii.vec2(1,1)) * 0.5

        points.append([p_image[0],p_image[1]])
        points_cam.append([p_cam[0],p_cam[1],p_cam[2]])
    return points, points_cam

def get_pose_in_camera(nv_trans, camera_name = 'my_camera'):
    # return point in image space and in camera 3d space.

    cam_matrix = visii.entity.get(camera_name).get_transform().get_world_to_local_matrix()
    cam_proj_matrix = visii.entity.get(camera_name).get_camera().get_projection()

    mat_trans = nv_trans.get_local_to_world_matrix()
    pos_m = visii.vec4(
            mat_trans[3][0],
            mat_trans[3][1],
            mat_trans[3][2],
        1)
        
    p_cam = cam_matrix * pos_m 

    p_image = cam_proj_matrix * (cam_matrix * pos_m) 
    p_image = visii.vec2(p_image) / p_image.w
    p_image = p_image * visii.vec2(1,-1)
    p_image = (p_image + visii.vec2(1,1)) * 0.5

    point_image = [p_image[0],p_image[1]]
    point_cam_3d = [p_cam[0],p_cam[1],p_cam[2]]

    return point_image, point_cam_3d


def export_to_ndds_file(
    filename = "tmp.json", #this has to include path as well
    obj_names = [], # this is a list of ids to load and export
    height = 500, 
    width = 500,
    camera_name = 'my_camera',
    cuboids = None,
    camera_struct = None,
    segmentation_mask = None,
    visibility_percentage = False, 
    scene_aabb = None, #min,max,center 
    ):
    # To do export things in the camera frame, e.g., pose and quaternion

    import simplejson as json

    # assume we only use the view camera
    cam_matrix = visii.entity.get(camera_name).get_transform().get_world_to_local_matrix()
    
    # print("get_world_to_local_matrix")
    # print(cam_matrix)
    # print('look_at')
    # print(camera_struct)
    # print('position')
    # print(visii.entity.get(camera_name).get_transform().get_position())
    # # raise()
    cam_matrix_export = []
    for row in cam_matrix:
        cam_matrix_export.append([row[0],row[1],row[2],row[3]])
    
    cam_world_location = visii.entity.get(camera_name).get_transform().get_position()
    cam_world_quaternion = visii.entity.get(camera_name).get_transform().get_rotation()
    # cam_world_quaternion = visii.quat_cast(cam_matrix)

    cam_intrinsics = visii.entity.get(camera_name).get_camera().get_intrinsic_matrix(width, height)

    if camera_struct is None:
        camera_struct = {
            'at': [0,0,0,],
            'eye': [0,0,0,],
            'up': [0,0,0,]
        }
    cam2wold = visii.entity.get(camera_name).get_transform().get_local_to_world_matrix()
    cam2wold_export = []
    for row in cam2wold:
        cam2wold_export.append([row[0],row[1],row[2],row[3]])
    if scene_aabb is None:
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
    dict_out = {
                "camera_data" : {
                    "width" : width,
                    'height' : height,
                    'camera_look_at':
                    {
                        'at': [
                            camera_struct['at'][0],
                            camera_struct['at'][1],
                            camera_struct['at'][2],
                        ],
                        'eye': [
                            camera_struct['eye'][0],
                            camera_struct['eye'][1],
                            camera_struct['eye'][2],
                        ],
                        'up': [
                            camera_struct['up'][0],
                            camera_struct['up'][1],
                            camera_struct['up'][2],
                        ]
                    },
                    'camera_view_matrix':cam_matrix_export,
                    'cam2world':cam2wold_export,
                    'location_world':
                    [
                        cam_world_location[0],
                        cam_world_location[1],
                        cam_world_location[2],
                    ],
                    'quaternion_world_xyzw':[
                        cam_world_quaternion[0],
                        cam_world_quaternion[1],
                        cam_world_quaternion[2],
                        cam_world_quaternion[3],
                    ],
                    'intrinsics':{
                        'fx':cam_intrinsics[0][0],
                        'fy':cam_intrinsics[1][1],
                        'cx':cam_intrinsics[2][0],
                        'cy':cam_intrinsics[2][1]
                    },
                    'scene_min_3d_box':scene_aabb[0],
                    'scene_max_3d_box':scene_aabb[1],
                    'scene_center_3d_box':scene_aabb[2],
                }, 
                "objects" : []
            }

    # Segmentation id to export
    id_keys_map = visii.entity.get_name_to_id_map()

    for obj_name in obj_names: 

        projected_keypoints, _ = get_cuboid_image_space(obj_name, camera_name=camera_name)

        # put them in the image space. 
        for i_p, p in enumerate(projected_keypoints):
            projected_keypoints[i_p] = [p[0]*width, p[1]*height]

        # Get the location and rotation of the object in the camera frame 


        trans = visii.transform.get(obj_name)
        if trans is None: 
            trans = visii.entity.get(obj_name).get_transform()
            
        quaternion_xyzw = visii.inverse(cam_world_quaternion) * trans.get_rotation()

        object_world = visii.vec4(
            trans.get_position()[0],
            trans.get_position()[1],
            trans.get_position()[2],
            1
        ) 
        pos_camera_frame = cam_matrix * object_world
 
        if not cuboids is None and obj_name in cuboids:
            cuboid = cuboids[obj_name]
        else:
            cuboid = None

        #check if the object is visible
        visibility = -1
        bounding_box = [-1,-1,-1,-1]

        if segmentation_mask is None:
            segmentation_mask = visii.render_data(
                width=int(width), 
                height=int(height), 
                start_frame=0,
                frame_count=1,
                bounce=int(0),
                options="entity_id",
            )
            segmentation_mask = np.array(segmentation_mask).reshape(width,height,4)[:,:,0]
            
        if visibility_percentage == True and int(id_keys_map [obj_name]) in np.unique(segmentation_mask.astype(int)): 
            transforms_to_keep = {}
            
            for name in id_keys_map.keys():
                if 'camera' in name.lower() or obj_name in name:
                    continue
                trans_to_keep = visii.entity.get(name).get_transform()
                transforms_to_keep[name]=trans_to_keep
                visii.entity.get(name).clear_transform()

            # Percentatge visibility through full segmentation mask. 
            segmentation_unique_mask = visii.render_data(
                width=int(width), 
                height=int(height), 
                start_frame=0,
                frame_count=1,
                bounce=int(0),
                options="entity_id",
            )

            segmentation_unique_mask = np.array(segmentation_unique_mask).reshape(width,height,4)[:,:,0]

            values_segmentation = np.where(segmentation_mask == int(id_keys_map[obj_name]))[0]
            values_segmentation_full = np.where(segmentation_unique_mask == int(id_keys_map[obj_name]))[0]
            visibility = len(values_segmentation)/float(len(values_segmentation_full))
            
            # bounding box calculation

            # set back the objects from remove
            for entity_name in transforms_to_keep.keys():
                visii.entity.get(entity_name).set_transform(transforms_to_keep[entity_name])
        else:
            # print(np.unique(segmentation_mask.astype(int)))
            # print(np.isin(np.unique(segmentation_mask).astype(int),
            #         [int(name_to_id[obj_name])]))
            try:
                if int(id_keys_map[obj_name]) in np.unique(segmentation_mask.astype(int)): 
                    #
                    visibility = 1
                    y,x = np.where(segmentation_mask == int(id_keys_map[obj_name]))
                    bounding_box = [int(min(x)),int(max(x)),height-int(max(y)),height-int(min(y))]
                else:
                    visibility = 0
            except:
                visibility= -1

        tran_matrix = trans.get_local_to_world_matrix()
    
        trans_matrix_export = []
        for row in tran_matrix:
            trans_matrix_export.append([row[0],row[1],row[2],row[3]])

        # Final export
        dict_out['objects'].append({
            # 'class':obj_name.split('_')[1],
            'name':obj_name,
            'provenance':'visii',
            # TODO check the location
            'location': [
                pos_camera_frame[0],
                pos_camera_frame[1],
                pos_camera_frame[2]
            ],
            'location_world': [
                trans.get_position()[0],
                trans.get_position()[1],
                trans.get_position()[2]
            ],
            'quaternion_xyzw':[
                quaternion_xyzw[0],
                quaternion_xyzw[1],
                quaternion_xyzw[2],
                quaternion_xyzw[3],
            ],
            'quaternion_xyzw_world':[
                trans.get_rotation()[0],
                trans.get_rotation()[1],
                trans.get_rotation()[2],
                trans.get_rotation()[3]
            ],
            'local_to_world_matrix':trans_matrix_export,
            'projected_cuboid':projected_keypoints,
            'local_cuboid': cuboid,
            'visibility':visibility,
            'bounding_box_minx_maxx_miny_maxy':bounding_box,
        })
        try:
            dict_out['objects'][-1]['segmentation_id']=id_keys_map[obj_name]
        except:
            dict_out['objects'][-1]['segmentation_id']=-1

        try:
            dict_out['objects'][-1]['mat_metallic']=visii.entity.get(obj_name).get_material().get_metallic()
            dict_out['objects'][-1]['mat_roughness']=visii.entity.get(obj_name).get_material().get_roughness()
            dict_out['objects'][-1]['mat_transmission']=visii.entity.get(obj_name).get_material().get_transmission()
            dict_out['objects'][-1]['mat_sheen']=visii.entity.get(obj_name).get_material().get_sheen()
            dict_out['objects'][-1]['mat_clearcoat']=visii.entity.get(obj_name).get_material().get_clearcoat()
            dict_out['objects'][-1]['mat_specular']=visii.entity.get(obj_name).get_material().get_specular()
            dict_out['objects'][-1]['mat_anisotropic']=visii.entity.get(obj_name).get_material().get_anisotropic()            
        except:
            pass            
    with open(filename, 'w+') as fp:
        json.dump(dict_out, fp, indent=4, sort_keys=True)
    # return bounding_box


def change_image_extension(path,extension="jpg"):
    import cv2
    import subprocess
    im = cv2.imread(path)
    cv2.imwrite(path.replace("png",extension),im)
    subprocess.call(['rm',path])
    del im
    



#################### BULLET THINGS ##############################





import pybullet as p


def create_obj(
    name = 'name',
    path_obj = "",
    path_tex = None,
    scale = 1, 
    rot_base = None
    ):

    
    # This is for YCB like dataset
    if path_obj in create_obj.meshes:
        obj_mesh = create_obj.meshes[path_obj]
    else:
        # obj_mesh = visii.mesh.create_from_obj(name, path_obj)
        # create_obj.meshes[path_obj] = obj_mesh

        obj_mesh = visii.mesh.create_from_file(name, path_obj)
        create_obj.meshes[path_obj] = obj_mesh


    
    obj_entity = visii.entity.create(
        name = name,
        # mesh = visii.mesh.create_sphere("mesh1", 1, 128, 128),
        mesh = obj_mesh,
        transform = visii.transform.create(name),
        material = visii.material.create(name)
    )

    # should randomize
    obj_entity.get_material().set_metallic(0)  # should 0 or 1      
    obj_entity.get_material().set_transmission(0)  # should 0 or 1      
    obj_entity.get_material().set_roughness(random.uniform(0,1)) # default is 1  

    if not path_tex is None:

        if path_tex in create_obj.textures:
            obj_texture = create_obj.textures[path_tex]
        else:
            obj_texture = visii.texture.create_from_file(name,path_tex)
            create_obj.textures[path_tex] = obj_texture


        obj_entity.get_material().set_base_color_texture(obj_texture)

    obj_entity.get_transform().set_scale(visii.vec3(scale))

    return obj_entity
create_obj.meshes = {}
create_obj.textures = {}

def create_physics(
    name="",
    mass = 1,         # mass in kg
    concave = False,
    ):

    # Set the collision with the floor mesh
    # first lets get the vertices 
    obj = visii.entity.get(name)
    vertices = []
    # print(name)
    # print(obj.get_mesh())
    # print(obj.to_string())
    # print(visii.mesh.get("mesh_floor"))

    for v in obj.get_mesh().get_vertices():
        vertices.append([float(v[0]),float(v[1]),float(v[2])])
    print(np.array(vertices).shape)
    min_z = np.min(np.array(vertices)[:,2])
    max_z = np.max(np.array(vertices)[:,2])
    print(min_z,max_z)
    # raise()
    # get the position of the object
    pos = obj.get_transform().get_position()
    pos = [pos[0],pos[1],pos[2]]
    scale = obj.get_transform().get_scale()
    scale = [scale[0],scale[1],scale[2]]
    rot = obj.get_transform().get_rotation()
    rot = [rot[0],rot[1],rot[2],rot[3]]

    # create a collision shape that is a convez hull

    if concave:
        indices = obj.get_mesh().get_triangle_indices()
        obj_col_id = p.createCollisionShape(
            p.GEOM_MESH,
            vertices = vertices,
            meshScale = scale,
            indices = indices,
        )


    else:
        # try:
        []
        obj_col_id = p.createCollisionShape(
            p.GEOM_MESH,
            vertices = vertices,
            meshScale = scale,
        )
        # except:
        #     return None
    # create a body without mass so it is static
    if not mass is None : 
        obj_id = p.createMultiBody(  
            baseMass = mass, 
            baseCollisionShapeIndex = obj_col_id,
            basePosition = pos,
            baseOrientation= rot,
            # baseInertialFramePosition =[0,0,max_z/2],
        )        
    else:
        obj_id = p.createMultiBody(
            baseCollisionShapeIndex = obj_col_id,
            basePosition = pos,
            baseOrientation= rot,
            # baseInertialFramePosition = [0,0,max_z/2],
        )    
    
    return obj_id


def update_pose(obj_dict):
    pos, rot = p.getBasePositionAndOrientation(obj_dict['bullet_id'])
    # print(pos)

    obj_entity = visii.entity.get(obj_dict['visii_id'])
    obj_entity.get_transform().set_position(visii.vec3(
                                            pos[0],
                                            pos[1],
                                            pos[2]
                                            )
                                        )
    if not obj_dict['base_rot'] is None: 
        obj_entity.get_transform().set_rotation(visii.quat(
                                                rot[3],
                                                rot[0],
                                                rot[1],
                                                rot[2]
                                                ) * obj_dict['base_rot']   
                                            )
    else:
        obj_entity.get_transform().set_rotation(visii.quat(
                                                rot[3],
                                                rot[0],
                                                rot[1],
                                                rot[2]
                                                )   
                                            )


import os 
import warnings
import simplejson as json

def material_from_json(path_json, randomize = False):
    """Load a json file visii material definition. 

    Parameters:
        path_json (str): The path to the json file to load
        randomize (bool): Randomize the material using the file definition (default:False)
    Return: 
        visii.material (visii.material): Returns a material object or None if there is a problem

    """

    if not os.path.isfile(path_json):
        warnings.warn(f"{path_json} does not exist")
        return None

    with open(path_json) as json_file:
        data = json.load(json_file)

    mat = visii.material.create(data['name'])


    # Load rgb image for base color
    if visii.texture.get(data['color']):
        mat.set_base_color_texture(visii.texture.get(data['color']))
    else:
        mat.set_base_color_texture(visii.texture.create_from_file(data['color'],data['color']))
    
    if 'gloss' in data:
        if visii.texture.get(data['gloss']):
            mat.set_roughness_texture(visii.texture.get(data['gloss']))
        else:
            im = cv2.imread(data['gloss'])
            # print(im.shape)
            im = np.power(1-(im/255.0),2)
            im_r = np.concatenate(
                [
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1),
                    im[:,:,1].reshape(im.shape[0],im.shape[1],1)
                ],
                2
            ) 
            roughness_tex = visii.texture.create_from_data(
                data['gloss'],
                im_r.shape[0],
                im_r.shape[1],
                im_r.reshape(im_r.shape[0]*im_r.shape[1],4).astype(np.float32).flatten()
            )
            mat.set_roughness_texture(visii.texture.create_from_file(data['gloss'],roughness_tex))        
    
    if "normal" in data:
        if visii.texture.get(data['normal']):
            mat.set_normal_texture(visii.texture.get(data['normal']))
        else:
            mat.set_normal_texture(visii.texture.create_from_file(data['normal'],data['normal'],linear = True))

    return mat

def load_amazon_object(path,suffix = ""):
    """
        This loads a single entity from a glb file. Assume the scene is one object. 

        path: path to the glb file 
        suffix: string to add to the name
        return: a visii entity
    """
    import trimesh
    scene = trimesh.load(path)

    for node in scene.graph.nodes_geometry:
        transform, node_name = scene.graph[node]

        mesh_tri = scene.geometry[node_name]

        points = np.array(mesh_tri.vertices)
        normals = np.array(mesh_tri.vertex_normals)
        triangles = np.array(mesh_tri.faces)
        # Flip the uvs
        uvs = np.array(mesh_tri.visual.uv)
        uvs[:,1] = 1-uvs[:,1]

        # get the textures
        tex_color_raw = np.array(mesh_tri.visual.material.baseColorTexture)
        tex_metallic_roughness = np.array(mesh_tri.visual.material.metallicRoughnessTexture)
        tex_normal = np.array(mesh_tri.visual.material.normalTexture)

        # print("tex_color_raw",tex_color_raw.shape)
        # print("tex_metallic_roughness",tex_metallic_roughness.shape)
        # print("tex_metallic_roughness[0]",tex_metallic_roughness[:,:,0].min(),tex_metallic_roughness[:,:,0].max())
        # print("tex_metallic_roughness[1]",tex_metallic_roughness[:,:,1].min(),tex_metallic_roughness[:,:,1].max())
        # print("tex_metallic_roughness[2]",tex_metallic_roughness[:,:,2].min(),tex_metallic_roughness[:,:,2].max())
        # print("tex_normal",tex_normal.shape)

        node_name = node_name + suffix

        mesh_visii = visii.mesh.create_from_data(
                        name = node_name,
                        positions = points.flatten(),
                        normals = normals.flatten(), 
                        texcoords = uvs.flatten(),
                        indices = triangles.flatten()
                    )

        ones = np.ones((tex_color_raw.shape[0],tex_color_raw.shape[1],1))

        material = visii.material.create(node_name)

        if tex_color_raw.shape[-1]==3:
            tex_color_raw = np.concatenate([tex_color_raw,ones],axis = 2)

        tex_color = visii.texture.create_from_data(
            node_name + 'color',
            tex_color_raw.shape[0],
            tex_color_raw.shape[1],
            (tex_color_raw.flatten()/255.0).astype(np.float32)
        )

        tex_roughness = visii.texture.create_from_data(
            node_name + 'roughness',
            tex_metallic_roughness.shape[0],
            tex_metallic_roughness.shape[1],
            (np.concatenate(
                [
                    np.expand_dims(tex_metallic_roughness[:,:,1],axis=-1),
                    np.expand_dims(tex_metallic_roughness[:,:,1],axis=-1),
                    np.expand_dims(tex_metallic_roughness[:,:,1],axis=-1),
                    np.ones((tex_color_raw.shape[0],tex_color_raw.shape[1],1))
                ],axis=2
            ).flatten()/255.0).astype(np.float32),
            linear = True
        )

        tex_metallic = visii.texture.create_from_data(
            node_name + 'metallic',
            tex_color_raw.shape[0],
            tex_color_raw.shape[1],
            (np.concatenate(
                [
                    np.expand_dims(tex_metallic_roughness[:,:,0],axis=-1),
                    np.expand_dims(tex_metallic_roughness[:,:,0],axis=-1),
                    np.expand_dims(tex_metallic_roughness[:,:,0],axis=-1),
                    np.ones((tex_color_raw.shape[0],tex_color_raw.shape[1],1))
                ],axis=2
            ).flatten()/255.0).astype(np.float32),
            linear = True
        )
        if len(tex_normal.shape)>2:
            tex_normal = visii.texture.create_from_data(
                node_name + 'normal',
                tex_normal.shape[0],
                tex_normal.shape[1],
                (np.concatenate([tex_normal,ones],axis = 2).flatten()/255.0).astype(np.float32),
                linear=True
            )
            material.set_normal_map_texture(tex_normal)

        material.set_base_color_texture(tex_color)
        material.set_metallic_texture(tex_metallic)
        material.set_roughness_texture(tex_roughness)

        entity_obj = visii.entity.create(
                        name = node_name,
                        transform = visii.transform.create(node_name),
                        mesh = mesh_visii, 
                        material = material
                    )

        ma = entity_obj.get_mesh().get_max_aabb_corner()
        mi = entity_obj.get_mesh().get_min_aabb_corner()
        size = np.sqrt( (ma[0]-mi[0])**2+(ma[1]-mi[1])**2+(ma[2]-mi[2])**2)
        entity_obj.get_transform().clear_parent()
        entity_obj.get_transform().set_scale(visii.vec3(1/size*.45))
        # entity_obj.get_transform().set_scale(visii.vec3(0.1))

        return entity_obj

def load_lego_object(path):
    """
        This loads a single entity from a lego model taken from 
        mecabrick.com. 

        path: path to the head folder for the lego scene 

        return: a list of visii entity names
    """
    # print(path)
    models = glob.glob(path+"/*.obj")
    # print(models)
    if len(models) == 0:
        raise('no models')

    obj_to_load = models[0]    
    name_model = models[0].replace(path,'').replace('.obj','').replace(' ',"_")
    
    print("loading:",name_model)
    name = path.split('/')[-2]

    # load the materials
    with open(obj_to_load.replace('obj','mtl')) as f:
        lines = f.readlines()
    colours = {}
    name_material = None

    for line in lines:
        # print(line)
        color = None
        texture = None
        transparency = 1

        if 'newmtl' in line:
            # print(line)
            name_material = line.split(' ')[-1].replace('\n','')
            # print(name_material)
            colours[name_material] = {
                "texture_path":None
            } 

        if 'Kd' in line and not 'png' in line:
            kd =  eval(",".join(line.split(' ')[1:]))
            # print(name_material,kd)
            # colours[name_material] = [kd[0],kd[1],kd[2]]
            colours[name_material]['rgb'] = [kd[0],kd[1],kd[2]]

        if 'map_Kd' in line:
            texture_name =  line.split(' ')[-1].replace('\n','')
            # print(name_material,kd)
            # colours[name_material] = texture_name
            colours[name_material]['texture_path'] = texture_name

        if 'd' == line.split(' ')[0]:
            value_transparency = line.split(' ')[-1]
            if not value_transparency == '1':
                colours[name_material+'_transparency'] = float(value_transparency)
            colours[name_material]['alpha'] = float(value_transparency)
        
        # print(line)


    toys = visii.import_scene(obj_to_load,
        visii.vec3(0,0,0),
        visii.vec3(0.0001,0.0001,0.0001), # the scale
        visii.angleAxis(1.57, visii.vec3(1,0,0))
        )

    for material in toys.materials:
        if 'DefaultMaterial' == material.get_name():
            continue

        rgb = colours[material.get_name()]['rgb']
        material.set_base_color(visii.vec3(float(rgb[0]),float(rgb[1]),float(rgb[2])))

        if colours[material.get_name()]['alpha']<1:
            material.set_alpha(colours[material.get_name()]['alpha'])

        material.set_roughness(0)
        material.set_metallic(0)
        material.set_transmission(0)
        material.set_specular(0.33)

    # find the entity names
    with open(obj_to_load) as f:
        lines = f.readlines()

    entity_names = []

    for line in lines: 
        if line[0] == 'o':
            name = line.split(' ')[-1].replace("\n","")
            entity_names.append(f"{name}_{name}")
    # print(entity_names)

    for entity_name in entity_names:

        # print(entity_name)
        if entity_name == 'camera':
            continue
        # print('get')
        
        entity = visii.entity.get(entity_name)
        # print('get_mat')

        mat_name = entity.get_material().get_name()
        if colours[mat_name]['alpha']<1:
            entity.set_visibility(shadow = False)
            mat = entity.get_material()
            mat.set_transmission(1)
            mat.set_ior(0.98)
            # print(entity.get_name())
            # entity.set_visibility(False)
            # print(entity.get_transform().get_parent().get_name())
    ma = visii.get_scene_max_aabb_corner()
    mi = visii.get_scene_min_aabb_corner()
    size = np.sqrt( (ma[0]-mi[0])**2+(ma[1]-mi[1])**2+(ma[2]-mi[2])**2)

    # print(size)
    # scale = 
    entity.get_transform().get_parent().set_scale(visii.vec3((0.0001*0.7)/size))

    # entity.get_transform().get_parent().set_scale(visii.vec3(100))
    ma = visii.get_scene_max_aabb_corner()
    mi = visii.get_scene_min_aabb_corner()
    size = np.sqrt( (ma[0]-mi[0])**2+(ma[1]-mi[1])**2+(ma[2]-mi[2])**2)
    # print(size)
    # raise()

    # entity.get_transform().get_parent().set_position(visii.vec3(0,0,0))

    return entity_names


def load_poly_cars(path):
    """
        This loads a single entity low poly from turbosquid. 
        https://www.turbosquid.com/3d-models/3d-model-vehicle-mega-pack-1280734
        path: path to the fbx file

        return: a list of visii entity names
    """
    
    obj_to_load = path    
    name_model = path

    print("loading:",name_model)
    name = path.split('/')[-2]

    toys = visii.import_scene(obj_to_load,
        visii.vec3(0,0,0),
        visii.vec3(0.01,0.01,0.01), # the scale
        visii.angleAxis(1.57, visii.vec3(1,0,0))
        )
    for material in toys.materials:
        # print(material.get_name())
        if 'Glass' in material.get_name():
            # print('changing')
            # material.set_transmission(0.7)
            material.set_metallic(0.7)
            material.set_roughness(0)
    entity_names = []
    for entity in toys.entities:
        entity_names.append(entity.get_name())

    return entity_names

def load_abc_object(path, id_obj=""):
    """
        data from there https://deep-geometry.github.io/abc-dataset/

        return: a list of visii entity names
    """
    
    obj_to_load = path    
    name_model = path

    print("loading:",name_model)
    name = path.split('/')[-2] + id_obj

    # check for obj files
    models = glob.glob(path+"*.obj")
    # print(models)
    if len(models)==0:
        return []

    toys = visii.import_scene(models[0],
        visii.vec3(0,0,0),
        visii.vec3(1,1,1), # the scale
        visii.angleAxis(1.57, visii.vec3(1,0,0))
        )
    for material in toys.materials:
        # print(material.get_name())
        if 'Glass' in material.get_name():
            # print('changing')
            # material.set_transmission(0.7)
            material.set_metallic(0.7)
            material.set_roughness(0)
        rgb = colorsys.hsv_to_rgb(
            random.uniform(0,1),
            random.uniform(0.1,1),
            random.uniform(0.1,1)
        )
        material.set_base_color(
            visii.vec3(
                rgb[0],
                rgb[1],
                rgb[2],
            )
        )
        material.set_roughness(random.uniform(0,1)) # default is 1  

    entity_names = []
    for entity in toys.entities:
        entity_names.append(entity.get_name())
        random_material(entity,just_simple=True)
        ma = entity.get_mesh().get_max_aabb_corner()
        mi = entity.get_mesh().get_min_aabb_corner()
        size = np.sqrt( (ma[0]-mi[0])**2+(ma[1]-mi[1])**2+(ma[2]-mi[2])**2)
        entity.get_transform().clear_parent()
        entity.get_transform().set_scale(visii.vec3(1/size*.2))

    print(entity_names)

    return entity_names


def load_grocery_object(path,suffix = ""):

    # obj_to_load = path + "/meshes/model.obj"
    # texture_to_load = path + "/materials/textures/texture.png"
    obj_to_load = path + "/google_16k/textured.obj"
    texture_to_load = path + "/google_16k/texture_map_flat.png"

    print("loading:",obj_to_load)

    name = path.split('/')[-2] + suffix

    scale = 1
    toy_mesh = visii.mesh.create_from_obj(name,obj_to_load)

    toy = visii.entity.create(
        name = name,
        transform = visii.transform.create(name),
        mesh = toy_mesh,
        material = visii.material.create(name)
    )
    
    # add_cuboid(name)
    toy_rgb_tex = visii.texture.create_from_image(name,texture_to_load)
    toy.get_material().set_base_color_texture(toy_rgb_tex) 
    toy.get_material().set_roughness(random.uniform(0.1,0.7))
    toy.get_material().set_metallic(random.uniform(0,0.3))

    return toy

def load_nvidia_scanned(path,suffix = ""):
    from . import usd_import
    print(path)
    objs = glob.glob(path + "*.usd")
    if len(objs) == 0: 
        return None

    obj_to_load = objs[0]
    node_name = obj_to_load.split("/")[1]
    pngs = glob.glob(path + "/SubUSDs/textures/*")

    
    # get the textures
    # tex_color_raw = np.array(mesh_tri.visual.material.baseColorTexture)
    # tex_metallic_roughness = np.array(mesh_tri.visual.material.metallicRoughnessTexture)
    # tex_normal = np.array(mesh_tri.visual.material.normalTexture)


    node_name = node_name + suffix
    usd_data = usd_import.import_mesh(obj_to_load, time=0,with_normals=True)

    # print("vertices",usd_data.vertices.shape)
    # print("face_normals",usd_data.face_normals.shape)
    # # print(usd_data.face_normals[0])

    # print("uvs",usd_data.uvs.shape)
    # print(usd_data.uvs[0])
    # print("face_uvs_idx",usd_data.face_uvs_idx.shape)
    # print(usd_data.face_uvs_idx[0],np.min(usd_data.face_uvs_idx),np.max(usd_data.face_uvs_idx))
    # print("faces",usd_data.faces.shape)
    # print(usd_data.faces[0])

    positions = [] 
    normals = []
    uvs = []

    for i_face in range(0,len(usd_data.face_uvs_idx)):
        for ii in range(3):
            positions.append(usd_data.vertices[usd_data.faces[i_face,ii]])
            normals.append(usd_data.face_normals[i_face,ii])
            uvs.append(usd_data.uvs[usd_data.face_uvs_idx[i_face,ii]])

    mesh_visii = visii.mesh.create_from_data(
                    name = node_name,
                    positions = np.array(positions).flatten(),
                    # normals = usd_data.face_normals.flatten(), 
                    normals = np.array(normals).flatten(),
                    # texcoords = usd_data.face_uvs_idx.flatten(),
                    texcoords = np.array(uvs).flatten(),
                    # texcoord_dimensions = usd_data.uvs.flatten(),
                    # indices = usd_data.faces.flatten()
                )

    name = path.split('/')[-2] + suffix

    scale = 0.1

    toy = visii.entity.create(
        name = name,
        transform = visii.transform.create(name,
            scale=visii.vec3(scale,scale,scale),
            rotation = visii.angleAxis(np.pi/2,(1,0,0))),
        mesh = mesh_visii,
        material = visii.material.create(name)
    )

    pngs = glob.glob(path + "/SubUSDs/textures/*.png")
    if len(pngs) == 1: 
        toy_rgb_tex = visii.texture.create_from_image(node_name,pngs[0])
        toy.get_material().set_base_color_texture(toy_rgb_tex) 
    if len(pngs) > 1: 
        for png in pngs: 
            if 'BC' in png: 
                toy_rgb_tex = visii.texture.create_from_image(node_name,png)
                toy.get_material().set_base_color_texture(toy_rgb_tex) 
                break                
    print("HLLLLLLLOOOO")
    # else:
    #     return None
    toy.get_material().set_roughness(random.uniform(0.1,0.7))
    toy.get_material().set_metallic(random.uniform(0,0.3))



    return toy

def load_google_scanned_objects(path,suffix = ""):

    obj_to_load = path + "/meshes/model.obj"
    texture_to_load = path + "/materials/textures/texture.png"

    print("loading:",obj_to_load)

    name = path.split('/')[-2] + suffix

    scale = 1
    toy_mesh = visii.mesh.create_from_obj(name,obj_to_load)

    toy = visii.entity.create(
        name = name,
        transform = visii.transform.create(name),
        mesh = toy_mesh,
        material = visii.material.create(name)
    )


    toy_rgb_tex = visii.texture.create_from_image(name,texture_to_load)
    toy.get_material().set_base_color_texture(toy_rgb_tex) 
    toy.get_material().set_roughness(random.uniform(0.1,0.7))
    toy.get_material().set_metallic(random.uniform(0,0.3))

    return toy

def material_from_cco(path_folder,scale=1):
    """Load a json file visii material definition. 

    Parameters:
        path_json (str): The path to the textures location
        scale (float): by how much to scale the texture - default 1.0
    Return: 
        visii.material (visii.material): Returns a material object or None if there is a problem
    """
    # print(path_folder.split("/")[-2])
    mat = visii.material.create(path_folder.split("/")[-2])

    files = glob.glob(path_folder+'/*.jpg')+glob.glob(path_folder+'/*.png')
    # print(files)

    for file in files:
        name_file_visii = file + str(scale)
        if 'color' in file.lower():
            if visii.texture.get(name_file_visii):
                mat.set_base_color_texture(visii.texture.get(name_file_visii))
            else:
                mat.set_base_color_texture(visii.texture.create_from_file(name_file_visii,file))
        if 'normal' in file.lower():
            if visii.texture.get(name_file_visii):
                mat.set_normal_map_texture(visii.texture.get(name_file_visii))
            else:
                mat.set_normal_map_texture(visii.texture.create_from_file(name_file_visii,file,linear=True))
        if 'rough' in file.lower():
            if visii.texture.get(name_file_visii):
                mat.set_roughness_texture(visii.texture.get(name_file_visii))
            else:
                mat.set_roughness_texture(visii.texture.create_from_file(name_file_visii,file,linear=True))
        if 'metal' in file.lower():
            if visii.texture.get(name_file_visii):
                mat.set_metallic_texture(visii.texture.get(name_file_visii))
            else:
                mat.set_metallic_texture(visii.texture.create_from_file(name_file_visii,file,linear=True))

        if visii.texture.get(name_file_visii):
            visii.texture.get(name_file_visii).set_scale((scale,scale))

    return mat


def find_camera_positions_collision_free(
        objects_in_scene,
        distance_max=1,
        nb_positions = 150,
        volume = 1.3,
    ):
    import pybullet as p

    # create a camera volume
    camera_pybullet_col = p.createCollisionShape(p.GEOM_SPHERE,0.01)
    camera_pybullet = p.createMultiBody(
        baseCollisionShapeIndex = camera_pybullet_col,
        basePosition = [0,0,0],
        # baseOrientation= rot,
    ) 

    positions_to_return = []

    while len(positions_to_return)<nb_positions:
        collided = False
        position_sample = [ random.uniform(-volume,volume),
                            random.uniform(-volume,volume),
                            random.uniform(0.02,volume)
                          ]
        p.resetBasePositionAndOrientation(camera_pybullet,position_sample,[1,0,0,0])

        for obj in objects_in_scene:
            # print('obj',obj)
            # print('camera_pybullet',camera_pybullet)
            if obj is None: 
                continue
            contact = p.getClosestPoints(camera_pybullet,obj,0.01)
            if len(contact)>0:
                collided = True
                break
        if collided is False:
            positions_to_return.append(position_sample)

    return positions_to_return


def create_falling_scene(
        objects_path,
        model_source,
        specific_models = None,
        box_size = 1,
        nb_objects_to_load = 1,
        interactive=False,
    ):
    """
    """

    import pybullet as p

    if interactive:
        physicsClient = p.connect(p.GUI) # non-graphical version
    else:
        physicsClient = p.connect(p.DIRECT) # non-graphical version

    p.setGravity(0,0,-10)

    visii_pybullet = []

    entities_loaded = []
    pybullet_ids_loaded = []
    # load the possible meshes 
    content_to_choose_from = []

    # data from grocery and YCB
    for folder in glob.glob(objects_path + "*/"):
        content_to_choose_from.append(folder)

    if model_source == 'amazon_berkeley':
        for folder in glob.glob(objects_path + "*.glb"):
            content_to_choose_from.append(folder)
    

    if not specific_models is None:
        content_to_choose_from = []

        for specific_instance in specific_models:
            content_to_choose_from.append(objects_path + specific_instance + "/")
        # nb_objects_to_load = len(content_to_choose_from)
        # print(content_to_choose_from)
        # raise()
    # print(len(content_to_choose_from))
    for i_obj in range(int(nb_objects_to_load)):
        loading = True

        toy_to_load = content_to_choose_from[random.randint(0,len(content_to_choose_from)-1)]
        # print(toy_to_load)
        # if not specific_models is None:
        #     toy_to_load = content_to_choose_from[i_obj]
        if model_source == 'amazon_berkeley':
            entity_loaded = load_amazon_object(toy_to_load,str(i_obj))    
            # scale = 0.1
            # entity_loaded.get_transform().set_scale(visii.vec3(scale))

        elif model_source == 'ycb':
            entity_loaded = load_grocery_object(toy_to_load,str(i_obj))    
            scale = 0.01
            entity_loaded.get_transform().set_scale(visii.vec3(scale))

        elif model_source == 'google_scanned':
            entity_loaded = load_google_scanned_objects(toy_to_load,str(i_obj))    
            scale = 1
            entity_loaded.get_transform().set_scale(visii.vec3(scale))

        elif model_source == 'abc':
            entity_loaded = load_abc_object(toy_to_load,str(i_obj))    
            if len(entity_loaded) == 0: 
                continue
            entity_loaded = visii.entity.get(entity_loaded[0])
            # scale = 0.1
        elif model_source == 'nvidia_scanned':
            entity_loaded = load_nvidia_scanned(toy_to_load,str(i_obj))    
            if entity_loaded is None:
                continue
            # entity_loaded = visii.entity.get(entity_loaded[0])
            # raise()

        
        entity_loaded.get_transform().set_rotation(
            visii.quat(
                random.uniform(0, 1),
                random.uniform(0, 1),
                random.uniform(0, 1),
                random.uniform(0, 1),
            )
        )

        id_pybullet = create_physics(entity_loaded.get_name(), mass = 1)
        
        # print(id_pybullet)
        # raise()
        pybullet_ids_loaded.append(id_pybullet)
        if id_pybullet is not None:

            visii_pybullet.append(
                {
                    'visii_id':entity_loaded.get_name(),
                    'bullet_id':id_pybullet,
                    'base_rot':None,
                }
            )
        entities_loaded.append(entity_loaded)
    
        # create the physical scene
        plane1 = p.createCollisionShape(p.GEOM_PLANE,planeNormal = [0,0,1])
        plane1_body = p.createMultiBody(
            baseCollisionShapeIndex = plane1,
            basePosition = [0,0,0],
        )    

        plane1 = p.createCollisionShape(p.GEOM_PLANE,planeNormal = [1,0,0])
        plane2_body = p.createMultiBody(
            baseCollisionShapeIndex = plane1,
            basePosition = [-box_size,0,0],
        )    

        plane1 = p.createCollisionShape(p.GEOM_PLANE,planeNormal = [-1,0,0])
        plane3_body = p.createMultiBody(
            baseCollisionShapeIndex = plane1,
            basePosition = [box_size,0,0],
        )    


        plane1 = p.createCollisionShape(p.GEOM_PLANE,planeNormal = [0,1,0])
        plane4_body = p.createMultiBody(
            baseCollisionShapeIndex = plane1,
            basePosition = [0,-box_size,0],
        )    

        plane1 = p.createCollisionShape(p.GEOM_PLANE,planeNormal = [0,-1,0])
        plane5_body = p.createMultiBody(
            baseCollisionShapeIndex = plane1,
            basePosition = [0,box_size,0],
        )    

        # set the objects
        for i_entry, entry in enumerate(visii_pybullet):
            pos_rand = [
                random.uniform(-1,1),
                random.uniform(-1,1),
                random.uniform(2,5),
            ]
            
            rot_random = visii.quat(
                random.uniform(0,1),
                random.uniform(0,1),
                random.uniform(0,1),
                random.uniform(0,1)
            )

            # update physics.
            p.resetBasePositionAndOrientation(
                entry['bullet_id'],
                pos_rand,
                [rot_random[0],rot_random[1],rot_random[2],rot_random[3]]
            )
    # raise()
    # update the physics and the nvisii entities. 
    print('simulating the falling')
    for i in range(10000):
        p.stepSimulation()

    # Update the pose of the objects
    for i_entry, entry in enumerate(visii_pybullet):
        print('update',entry)
        update_pose(entry)

    return {
        "nvisii_entities":entities_loaded,
        "pybullet_ids": pybullet_ids_loaded,
    }
def scene_names_to_point_cloud(exported_names,path_to_output):
    """
        exported_names: names in nvisii to be exported 
        path_to_output: path to the output where to put scene.ply
    """
    array_to_export = []
    points_all = []
    triangles_all = []

    for name in exported_names:
        toy = visii.entity.get(name)
        try:
            toy_mesh = toy.get_mesh()
            triangles = list(toy_mesh.get_triangle_indices())
        except: 
            continue
        # print(triangles)
        # raise()
        vertices = np.array(toy_mesh.get_vertices())
        trans_matrix = toy.get_transform().get_local_to_world_matrix()
        points = []
        previous_object_nb_points = len(points_all)
        # print(triangles)
        for triangle in triangles:
            triangles_all.append(triangle + previous_object_nb_points)    
        for v in vertices:
            pos_m = visii.vec4(
                v[0],
                v[1],
                v[2],
                1)
            
            p_world = trans_matrix * pos_m 
            # print(p_world)
            points.append([p_world[0],p_world[1],p_world[2]])
            points_all.append([p_world[0],p_world[1],p_world[2]])
        # raise()
        array_to_export.append([triangles,points])

    # # # # # # # # # # # # # # # # # # # # # # # # #
    ply_text = f"ply\n\
    format ascii 1.0\n\
    element vertex {len(points_all)}\n\
    property float x\n\
    property float y\n\
    property float z\n\
    element face {int(len(triangles_all)/3)}\n\
    property list uchar uint vertex_indices\n\
    end_header\n"

    for point in points_all:
         ply_text += f"{point[0]} {point[1]} {point[2]}\n"

    for i_tri in range(0,len(triangles_all),3):
         ply_text += f"3 {triangles_all[i_tri]} {triangles_all[i_tri+1]} {triangles_all[i_tri+2]}\n"

    with open(f'{path_to_output}/scene.ply','w+') as f:
        f.write(ply_text)

def interact():
    global speed_camera
    global cursor
    global rot

    # nvisii camera matrix 
    cam_matrix = camera.get_transform().get_local_to_world_matrix()
    dt = nvisii.vec4(0,0,0,0)

    # translation
    if nvisii.is_button_held("W"): dt[2] = -speed_camera
    if nvisii.is_button_held("S"): dt[2] =  speed_camera
    if nvisii.is_button_held("A"): dt[0] = -speed_camera
    if nvisii.is_button_held("D"): dt[0] =  speed_camera
    if nvisii.is_button_held("Q"): dt[1] = -speed_camera
    if nvisii.is_button_held("E"): dt[1] =  speed_camera 

    # control the camera
    if nvisii.length(dt) > 0.0:
        w_dt = cam_matrix * dt
        camera.get_transform().add_position(nvisii.vec3(w_dt))

    # camera rotation
    cursor[2] = cursor[0]
    cursor[3] = cursor[1]
    cursor[0] = nvisii.get_cursor_pos().x
    cursor[1] = nvisii.get_cursor_pos().y
    if nvisii.is_button_held("MOUSE_LEFT"):
        nvisii.set_cursor_mode("DISABLED")
        rotate_camera = True
    else:
        nvisii.set_cursor_mode("NORMAL")
        rotate_camera = False

    if rotate_camera:
        rot.x -= (cursor[0] - cursor[2]) * 0.001
        rot.y -= (cursor[1] - cursor[3]) * 0.001
        init_rot = nvisii.angleAxis(nvisii.pi() * .5, (1,0,0))
        yaw = nvisii.angleAxis(rot.x, (0,1,0))
        pitch = nvisii.angleAxis(rot.y, (1,0,0)) 
        camera.get_transform().set_rotation(init_rot * yaw * pitch)

    # change speed movement
    if nvisii.is_button_pressed("UP"):
        speed_camera *= 0.5 
        print('decrease speed camera', speed_camera)
    if nvisii.is_button_pressed("DOWN"):
        speed_camera /= 0.5
        print('increase speed camera', speed_camera)