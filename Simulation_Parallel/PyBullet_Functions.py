import pybullet as p
import pybullet_data
import numpy as np
import husky_ur5
import time 
id_lookup = None
def create_world():
    global id_lookup
    world = 'jsons/simple_world_home0.json'
    objects = "jsons/simple_objects.json"
    id_lookup = husky_ur5.start(world, object_file= objects)
    return id_lookup

def init_pybullet(flag, time_step, shaddow = False, fpath = "", Path = []):
    """
    flag can be no_gui, gui, or mp4
    """    
    if flag == "no_gui":    
        p.connect(p.DIRECT )
    elif flag == "gui":
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    elif flag =="mp4":
        print("mp4")
        p.connect(p.GUI, options= f"--mp4={fpath}test.mp4 --mp4fps=10")
        # p.connect(p.GUI, options="--width=320 --height=200 --mp4=\"test.mp4\" --mp4fps=240")
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    
    # p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)    
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

    # p.resetDebugVisualizerCamera( cameraDistance=2, cameraYaw=180, cameraPitch=270, cameraTargetPosition=[0,0,0])
    
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
    if shaddow:
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0)
    

    id_lookup = create_world()
    pos,orn = p.getBasePositionAndOrientation(id_lookup['Husky'])
    pos = np.array(pos)
    pos[:2] = Path[0]
    p.resetBasePositionAndOrientation(id_lookup['Husky'], pos, orn)
    print('-------------------------')
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./time_step)
    p.setRealTimeSimulation(0)                                          # switch to real time simulation

    return id_lookup

def update_simulator(car_name, vel_control, num_steps, time_step):    
    global id_lookup
    FWr, FWl =  vel_control
    car = id_lookup[car_name]
    # pass
    ## TurtleBot
    # ## TurtleBot wheel parameters
    for _ in range(num_steps):     
        if car_name == "Turtle":                
            p.setJointMotorControl2(car,0,p.VELOCITY_CONTROL,targetVelocity=FWl)
            p.setJointMotorControl2(car,1,p.VELOCITY_CONTROL,targetVelocity=FWr)    
        
        ## Husky
        if car_name == 'Husky':            
            p.setJointMotorControl2(car,2,p.VELOCITY_CONTROL,targetVelocity=FWl)
            p.setJointMotorControl2(car,3,p.VELOCITY_CONTROL,targetVelocity=FWr)    
            p.setJointMotorControl2(car,4,p.VELOCITY_CONTROL,targetVelocity=FWl)
            p.setJointMotorControl2(car,5,p.VELOCITY_CONTROL,targetVelocity=FWr)    
        
        if car_name == 'Prius':            
            p.setJointMotorControl2(car,3,p.VELOCITY_CONTROL,targetVelocity=FWl)
            p.setJointMotorControl2(car,5,p.VELOCITY_CONTROL,targetVelocity=FWr)    
            p.setJointMotorControl2(car,6,p.VELOCITY_CONTROL,targetVelocity=FWl)
            p.setJointMotorControl2(car,7,p.VELOCITY_CONTROL,targetVelocity=FWr)    
    
        p.stepSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        #print("r2d2 vel=", p.getBaseVelocity(r2d2)[0][2])
        # p.removeUserDebugItem(item)
        time.sleep(1./time_step)     

def place_obstacles(Obstacles):
    """
    flag cylindr, or hen ## TODO random, and others
    """
    p_obs = []
    
    for obs in Obstacles:
        # print(obs)
        obsX, obsY, obsR, flag, obs_rot = obs         
        height = 0.5
        mass = np.pi * (obsR **2) * height
        if flag == "cylindr":
            obs_id = p.createCollisionShape(p.GEOM_CYLINDER, radius = obsR, height = 0.5)
            
            basePosition    = [obsX, obsY, height/2]
            baseOrientation = [0, 0, 0, 1]

            obs_id = p.createMultiBody(mass, obs_id, -1, basePosition, baseOrientation)


        if flag == "hen":
            obs        = p.createCollisionShape(p.GEOM_MESH,
                                                fileName='pybulletData/new_urdf/rooster_1/chicken.obj',
                                                meshScale=np.array([1,1,1])* obsR * 3,
                                                flags = 1)
            visual_obs = p.createVisualShape(p.GEOM_MESH,
                                        fileName='pybulletData/new_urdf/rooster_1/chicken.obj',
                                        meshScale=np.array([1,1,1])* obsR* 3)
            obs_id     = p.createMultiBody(baseCollisionShapeIndex=obs,
                                        baseVisualShapeIndex=visual_obs,
                                        basePosition=[obsX,obsY,0],                                    
                                        baseOrientation=p.getQuaternionFromEuler([0, 0, obs_rot]),
                                        baseMass=0.1)
            rooster_texture = p.loadTexture("pybulletData/new_urdf/rooster_1/chicken.jpg")
            p.changeVisualShape(obs_id, -1, textureUniqueId = rooster_texture, rgbaColor = [1, 1, 1, 1])
            
        if flag == "sofa":
            obs_id = p.loadURDF("pybulletData/urdf/sofa_1/sofa.urdf",globalScaling=obsR *4  ,
                    basePosition= [obsX, obsY, 0], #[-4,4,0],
                    baseOrientation = p.getQuaternionFromEuler([0, 0, obs_rot]) )#[0,0,0,1])


        if flag == 'turtle':
            car_start_pose = [obsX, obsY, height/2] #[x_shifted, y_shifted, 0.1] # shifted position    
            car_orientation = p.getQuaternionFromEuler([0, 0, 0])
            
            obs_id = p.loadURDF("pybulletData/turtlebot.urdf",  car_start_pose, car_orientation) # load car model
            ### Change color of Turtle bot to Green
            for link in range(-1,p.getNumJoints(obs_id)):
                p.changeVisualShape(obs_id, link, rgbaColor=[0.9, 0.2, 0.1, 1])
        p_obs.append(obs_id) 
    
    
    return p_obs
    
def get_image(width = 128, height = 128, fov = 60, near = 0.02, far = 1,
              camTargetPos = [0,0,0],
              camDistance = 2.0,
              angle = [0,0,0],
              upAxisIndex = 2
              ):
    
    aspect = width / height
    yaw, pitch, roll = angle
    view_Matrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Get depth values using the OpenGL renderer
    images = p.getCameraImage(width, height, 
                                    #   view_Matrix, projection_matrix, 
                                      shadow=1,
                                      lightDirection=[1, 1, 1],
                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)

    np_img_arr = np.reshape(images[2], (images[1], images[0], 4))
    np_img_arr = np_img_arr * (1. / 255.)
    return np_img_arr

