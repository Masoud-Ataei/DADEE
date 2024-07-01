import os
import time
import pdb
import pybullet as p
import time
import os, shutil
from src.initialise import initHuskyUR5
import matplotlib.pyplot as plt
from operator import sub
import math
import pickle
import os


wings_file = "jsons/wings.json"
tolerance_file = "jsons/tolerance.json"
goal_file = "jsons/goal.json"

if not os.path.exists("logs"): os.mkdir("logs")

#Number of steps before image capture
COUNTER_MOD = 50

# Enclosures
enclosures = ['fridge', 'cupboard']

# Semantic objects
# Sticky objects
sticky = []
# Fixed objects
fixed = []
# Objects on
on = ['light']
# Has been fueled
fueled = []
# Cut objects
cut = []
# Has cleaner
cleaner = False
# Has stick
stick = False
# Objects cleaned
clean = []
# Objects drilled
drilled = []
# Objects welded
welded = []
# Objects painted
painted = []
names = {}

def initDisplay(display):
    plt.axis('off')
    plt.rcParams["figure.figsize"] = [8,6]
    cam = plt.figure(3)
    plt.axis('off')
    ax = plt.gca()
    return ax, cam

def mentionNames(id_lookup):
    """
    Add labels of all objects in the world
    """
    
    for obj in id_lookup.keys():
        id = p.addUserDebugText(obj, 
                        (0, 0, 0.2),
                        parentObjectUniqueId=id_lookup[obj])
def start(world = None, object_file = "jsons/objects.json", _speed = 1.0):
  # Initialize husky and ur5 model
  global husky, object_lookup, id_lookup, horizontal_list, ground_list,fixed_orientation,tolerances, properties,cons_cpos_lookup,cons_pos_lookup, cons_link_lookup,ur5_dist,states,wings,gotoWing,constraints,constraint,x1, y1, o1
  global imageCount,yaw,ims,dist,pitch,ax,fig,cam,camX, camY, world_states,id1, perspective, wall_id, datapoint
  global light, args, speed

  # Connect to Bullet using GUI mode
  # light = p.connect(p.GUI)

  # Add input arguments  
  speed = _speed
  
  # return print('speed', speed)

  # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
  # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
  
  
  ( husky,     
    object_lookup, 
    id_lookup, 
    horizontal_list, 
    ground_list,
    fixed_orientation,
    tolerances, 
    properties,
    cons_cpos_lookup,
    cons_pos_lookup, 
    cons_link_lookup,
    ur5_dist,
    states) = initHuskyUR5(world, object_file)
    
  print ("The world file is", world)
  
  # Initialize dictionary of wing positions
  # wings = initWingPos(wings_file)

  # Set small gravity
  p.setGravity(0,0,-10)

  # Initialize gripper joints and forces
  # controlJoints, joints = initGripper(robotID)
  # gotoWing = getUR5Controller(robotID)
  # gotoWing(robotID, wings["home"])

  # Position of the robot
  x1, y1, o1 = 0, 0, 0
  constraint = 0

  # List of constraints with target object and constraint id
  constraints = dict()

  # Init camera
  imageCount = 0
  yaw = 50
  ims = []
  dist = 5
  pitch = -35.0

  # Start video recording
  p.setRealTimeSimulation(0) 
  # ax = 0; fig = 0; cam = []
  

  # ax, cam = initDisplay("both")
  
  # camX, camY = 0, 0

  # Mention names of objects
  # mentionNames(id_lookup)

  # Save state
  world_states = []
  id1 = p.saveState()
  world_states.append([id1, x1, y1, o1, constraints])
  print(id_lookup)
  print(fixed_orientation)

  # Default perspective
  perspective = "tp"

  # Wall to make trasparent when camera outside
  wall_id = -1
  # if 'home' in world:
  #   wall_id = id_lookup['walls']
  # if 'factory' in world:
  #   wall_id = id_lookup['wall_warehouse']

  return id_lookup


