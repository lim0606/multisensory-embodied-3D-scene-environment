import os
import sys
import argparse
import time
import copy
from PIL import Image
import pickle as pkl
import numpy as np
import scipy
#import torch

import gym
import envs

from utils import gen_ornstein_uhlenbeck_process
from utils import load_env_info, save_env_info, save_image, load_image, save_observation_action, load_observation_action
from utils import save_scene, load_scene
from utils import r_from_pitch_yaw, r_from_roll_yaw
from utils import compare_part, compare_shape, compare_shape_with_shapelist, get_label
from utils import run_camera_exp, run_hand_exp


# define Scene class
class Scene(object):
    def __init__(self,
                 images,
                 cameras,
                 proprioceptives,
                 hands,
                 hand_images,
                 sampled_dir_cameras,
                 sampled_dir_hands,
                 env_info,
                 ):
        self.images =              images
        self.cameras =             cameras
        self.proprioceptives =     proprioceptives
        self.hands =               hands
        self.hand_images =         hand_images
        self.sampled_dir_cameras = sampled_dir_cameras
        self.sampled_dir_hands =   sampled_dir_hands
        self.env_info =            env_info


# global parameters for this experiment
parser = argparse.ArgumentParser()
parser.add_argument('--n-parts', type=int, default=5,
                    help='# of parts in Shepard Metzler object')
parser.add_argument('--start_ind_experiment', type=int, default=0,
                    help='start index of experiments')
parser.add_argument('--n_experiments', type=int, default=100,
                    help='number of environment instances for a given environment')
parser.add_argument('--nrow', type=int, default=4, help='nrow')
parser.add_argument('--ncol', type=int, default=4, help='ncol')
parser.add_argument('--start_ind_img_episode', type=int, default=0,
                    help='start index of episode')
parser.add_argument('--n_img_episodes', type=int, default=15,
                    help='number of episodes for each env instance')
parser.add_argument('--start_ind_hpt_episode', type=int, default=0,
                    help='start index of episode')
parser.add_argument('--n_hpt_episodes', type=int, default=15,
                    help='number of episodes for each env instance')
parser.add_argument('--split', default='train',
                    choices=['train', 'val', 'test'],
                    help='split: train | val | test')
update_parser = parser.add_mutually_exclusive_group(required=False)
update_parser.add_argument('--update-unique-shapes', dest='update_unique_shapes', action='store_true', help='flag for update unique shapes.')
update_parser.add_argument('--no-update-unique-shapes', dest='update_unique_shapes', action='store_false', help='flag for update unique shapes.')
parser.set_defaults(update_unique_shapes=True)

# parse arguments
opt = parser.parse_args()
opt.camera_environment = 'ShepardMetzlerCameraSingleObjP{}-v0'.format(opt.n_parts)
opt.hand_environment   = 'ShepardMetzlerHandSingleObjP{}-v0'.format(opt.n_parts)
env = gym.make(opt.camera_environment)

# init variables
dt = env.env.dt  # init dt
threshold = 150.
img_pitches = np.linspace(-np.pi/4, np.pi/4, num=opt.nrow, endpoint=True)
img_yaws    = np.linspace(0+np.pi/5., 2.*np.pi-np.pi/5., num=opt.ncol, endpoint=True)

# init unique_shapes
os.system('mkdir -p data/haptix/shepard_metzler_{}_parts'.format(opt.n_parts))
shapes_filename = 'data/haptix/shepard_metzler_{}_parts/shapes.pkl'.format(opt.n_parts)
num_unique_shapes = 0
unique_shapes = [None]*opt.n_experiments
if os.path.exists(shapes_filename):
    # load shapes
    with open(shapes_filename, 'rb') as f:
        _unique_shapes = pkl.load(f)
    num_unique_shapes = len(_unique_shapes)

    # expand container
    if len(unique_shapes) < num_unique_shapes:
        unique_shapes = [None]*(2*num_unique_shapes)

    # load shapes
    for i in range(num_unique_shapes):
        unique_shapes[i] = _unique_shapes[i]

# run experiments
for i_exp in range(opt.start_ind_experiment, opt.start_ind_experiment+opt.n_experiments):
    # init env
    env = gym.make(opt.camera_environment)
    env.env.with_fixed_window_size=True

    # init enviroment info
    info = copy.deepcopy(env.env.info)

    # get shape
    shape = info['obj_infos'][0]['parts_rel']

    # check shape
    is_same = compare_shape_with_shapelist(shape, unique_shapes[:num_unique_shapes])

    # update
    if opt.update_unique_shapes and not is_same:
        unique_shapes[num_unique_shapes] = shape
        num_unique_shapes += 1
        print('num_unique_shapes: ', num_unique_shapes)
    else:
        assert is_same

    # get label
    label = get_label(shape, unique_shapes)

    # save to info
    info['class'] = label

    # run camera experiment
    images, cameras, sampled_dir_cameras = run_camera_exp(opt.camera_environment, info, i_exp, start_ind_img_episode=opt.start_ind_img_episode, n_img_episodes=opt.n_img_episodes)

    # run hand experiment
    hand_images, proprioceptives, hands, sampled_dir_hands = run_hand_exp(opt.hand_environment, info, i_exp, start_ind_hpt_episode=opt.start_ind_hpt_episode, n_hpt_episodes=opt.n_hpt_episodes, dt=dt, threshold=threshold)

    # save scene
    scene = Scene(
            images=images,
            cameras=cameras,
            proprioceptives=proprioceptives,
            hands=hands,
            hand_images=hand_images,
            sampled_dir_cameras=sampled_dir_cameras,
            sampled_dir_hands=sampled_dir_hands,
            env_info=info,
            )
    sys.modules['__main__'].Scene = Scene ## hack because of the pickle saving..
    save_scene(i_exp, scene, suffix='shepard_metzler_{}_parts/{}'.format(opt.n_parts, opt.split))

    # save shapes
    if opt.update_unique_shapes:
        _unique_shapes = unique_shapes[:num_unique_shapes]
        pkl.dump(_unique_shapes, open(shapes_filename, 'wb'))

# save shapes
if opt.update_unique_shapes:
    unique_shapes = unique_shapes[:num_unique_shapes]
    pkl.dump(unique_shapes, open(shapes_filename, 'wb'))
else:
    print('unique_shapes: ', len(unique_shapes))
