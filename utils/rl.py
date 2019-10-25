'''
miscellaneous functions
'''
import os
import errno
import time
import math
import gzip
import numpy as np
import pickle as pkl

from PIL import Image

import gym
import matplotlib.pyplot as plt

from utils.learning import logging
from utils.transformers import quaternion_from_matrix, rotation_matrix


''' for rl general '''
def get_random_variable(probs, dist='categorical'):
    if dist == 'categorical':
        rv = Categorical(probs)
    else:
        rv = Normal(probs[0], probs[1])

    return rv

def get_log_prob(rv, action):
    log_prob = rv.log_prob(action)
    return log_prob

def sample(probs, dist='categorical'):
    # get random variable
    rv = get_random_variable(probs, dist=dist)

    # sample
    action = rv.sample()

    ## measure log prob
    #log_prob = get_log_prob(rv, action)

    return action#, log_prob


''' for random action generation '''
def gen_ornstein_uhlenbeck_process(x, mu=1.0, sig=0.2, dt=1e-2):
    '''
    https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    https://gist.github.com/StuartGordonReid/961cd2b227d023aa51af
    '''
    th = 1
    mu = mu  # 1.2
    sig = sig  # 0.3
    dt = dt  # 1e-2
    dx = th * (mu-x) * dt + sig * math.sqrt(dt) * np.random.normal(size=x.size)
    x_tp1 = x + dx
    return x_tp1

'''
convert angle and axis to rotation matrix⋅
https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py
'''
def r_from_angle_axis(angle, axis):
    """Generate the rotation matrix from the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        c = cos(angle); s = sin(angle); C = 1-c
        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC
        [ x*xC+c   xyC-zs   zxC+ys ]
        [ xyC+zs   y*yC+c   yzC-xs ]
        [ zxC-ys   yzC+xs   z*zC+c ]
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @param axis:    The 3D rotation axis.
    @type axis:     numpy array, len 3
    @param angle:   The rotation angle.
    @type angle:    float
    """

    # Trig factors.
    ca = np.cos(angle)
    sa = np.sin(angle)
    C = 1 - ca

    # Depack the axis.
    x, y, z = axis[0], axis[1], axis[2]

    # Multiplications (to remove duplicate calculations).
    xs = x*sa
    ys = y*sa
    zs = z*sa
    xC = x*C
    yC = y*C
    zC = z*C
    xyC = x*yC
    yzC = y*zC
    zxC = z*xC

    # Update the rotation matrix
    matrix = np.zeros((3,3))
    matrix[0, 0] = x*xC + ca
    matrix[0, 1] = xyC - zs
    matrix[0, 2] = zxC + ys
    matrix[1, 0] = xyC + zs
    matrix[1, 1] = y*yC + ca
    matrix[1, 2] = yzC - xs
    matrix[2, 0] = zxC - ys
    matrix[2, 1] = yzC + xs
    matrix[2, 2] = z*zC + ca

    return matrix

'''
convert pitch, yaw to R
https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
'''
def r_from_pitch_yaw(pitch, yaw):
    # apply global yaw -> R_z
    R_z = rotation_matrix(yaw, np.array([0., 0., 1.]))[:3, :3]

    # infer new rotation axis given R_z
    new_y_axis = R_z.dot(np.array([0., 1., 0.]))
    R_local_y = rotation_matrix(pitch, new_y_axis)[:3, :3]

    # combine rotations
    R = R_local_y.dot(R_z)

    return R

'''
convert pitch, yaw to quat
https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
'''
def quat_from_pitch_yaw(pitch, yaw):
    # convert pitch, yaw to R
    R = r_from_pitch_yaw(pitch, yaw)

    # infer quaternion from rotation matrix
    quat = quaternion_from_matrix(R)

    return quat

'''
convert roll, yaw to R
https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
'''
def r_from_roll_yaw(roll, yaw):
    # apply global yaw -> R_z
    R_z = rotation_matrix(yaw, np.array([0., 0., 1.]))[:3, :3]

    # infer new rotation axis given R_z
    new_x_axis = R_z.dot(np.array([1., 0., 0.]))
    R_local_x = rotation_matrix(roll, new_x_axis)[:3, :3]

    # combine rotations
    R = R_local_x.dot(R_z)

    return R

'''
convert roll, yaw to quat
https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
'''
def quat_from_roll_yaw(roll, yaw):
    # convert roll, yaw to R
    R = r_from_roll_yaw(roll, yaw)

    # infer quaternion from rotation matrix
    quat = quaternion_from_matrix(R)

    return quat


''' for environment settings '''
def load_env_info(env_name, suffix='simple'):
    # init root path
    root = os.path.join('data', 'haptix', suffix)

    # load info
    filename = os.path.join(root, '{}-info.pkl'.format(env_name))
    if os.path.exists(filename):
        info = pkl.load(open(filename, 'rb'))
        return info
    else:
        return False

def save_env_info(info, env_name, suffix='simple'):
    # init root path
    root = os.path.join('data', 'haptix', suffix)

    # make directory
    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # load info
    filename = os.path.join(root, '{}-info.pkl'.format(env_name))
    if os.path.exists(filename):
        return False
    else:
        pkl.dump(info, open(filename, 'wb'))
        return True


''' for generative models '''
def save_image(i_exp, i_episode, t, img, suffix='simple'):
    # init root path
    root = os.path.join('data', 'haptix', suffix)

    if t < 0:
        # init path
        path = os.path.join(root, 'env{}'.format(i_exp))

        # init filename
        filename = os.path.join(path, '{}.png'.format(i_episode))
    else:
        # init path
        path = os.path.join(root, 'env{}'.format(i_exp), 'ep{}'.format(i_episode))

        # init filename
        filename = os.path.join(path, '{}.png'.format(t))

    # make directory
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # save
    #plt.imsave(filename, img)
    img = Image.fromarray(img, mode='RGB')
    img.save(filename, format='PNG')

    # save to file list
    logging('{}.png'.format(t), path, filename='filenames.list', print_=False, log_=True)

def load_image(i_exp, i_episode, t, suffix='simple'):
    # init root path
    root = os.path.join('data', 'haptix', suffix)

    if t < 0:
        # init path
        path = os.path.join(root, 'env{}'.format(i_exp))

        # init filename
        filename = os.path.join(path, '{}.png'.format(i_episode))
    else:
        # init path
        path = os.path.join(root, 'env{}'.format(i_exp), 'ep{}'.format(i_episode))

        # init filename
        filename = os.path.join(path, '{}.png'.format(t))

    # load img
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            img = Image.open(f)
            return np.array(img.convert('RGB'))
    else:
        return False

def save_observation_action(i_exp, i_episode, t, observation, action, suffix='simple'):
    # init root path
    root = os.path.join('data', 'haptix', suffix)

    if t < 0:
        # init path
        path = os.path.join(root, 'env{}'.format(i_exp))

        # init filename
        filename = os.path.join(path, '{}.npy'.format(i_episode))
    else:
        # init path
        path = os.path.join(root, 'env{}'.format(i_exp), 'ep{}'.format(i_episode))

        # init filename
        filename = os.path.join(path, '{}.npy'.format(t))

    # make directory
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # save
    np.save(filename, {'observation': observation, 'action': action})

def load_observation_action(i_exp, i_episode, t, suffix='simple'):
    # init root path
    root = os.path.join('data', 'haptix', suffix)

    if t < 0:
        # init path
        path = os.path.join(root, 'env{}'.format(i_exp))

        # init filename
        filename = os.path.join(path, '{}.npy'.format(i_episode))
    else:
        # init path
        path = os.path.join(root, 'env{}'.format(i_exp), 'ep{}'.format(i_episode))

        # init filename
        filename = os.path.join(path, '{}.npy'.format(t))

    # load observation and action
    if os.path.exists(filename):
        obj = np.load(filename)
        observation = obj.item().get('observation').astype(np.float32)
        action = obj.item().get('action').astype(np.float32)
        return observation, action
    else:
        return False

def save_scene(i_exp, scene, suffix='simple'):
    # init root path
    root = os.path.join('data', 'haptix', suffix)

    # init filename
    filename = os.path.join(root, 'scene{}.pkl.gz'.format(i_exp))

    # make directory
    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # save
    with gzip.open(filename, 'wb') as f:
        pkl.dump(scene, f)

def load_scene(i_exp, suffix='simple', root=None):
    # init root path
    if root is None:
        root = os.path.join('data', 'haptix', suffix)

    # init filename
    filename = os.path.join(root, 'scene{}.pkl.gz'.format(i_exp))

    # load observation and action
    if os.path.exists(filename):
        with gzip.open(filename, 'rb') as f:
            scene = pkl.load(f)
        return scene
    else:
        return False


''' compare objects '''
def compare_part(part1, part2):
    is_same = 0
    for i in range(3):
        is_same += int(abs(part1[i] - part2[i]) < 1e-3)
    return is_same == 3

def compare_shape(shape1, shape2):
    is_same = 0
    for i in range(len(shape1)):
        is_same += 1 if compare_part(shape1[i], shape2[i]) else 0
    return is_same == len(shape1)

def compare_shape_with_shapelist(shape, shapelist):
    is_same = False
    for j in range(len(shapelist)):
        is_same = compare_shape(shape, shapelist[j])
        if is_same:
            break
    return is_same

def get_label(shape, shapelist):
    is_same = False
    for j in range(len(shapelist)):
        is_same = compare_shape(shape, shapelist[j])
        if is_same:
            break
    assert is_same
    return j


''' experiment '''
# init msc funciton
def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx, :]

def normalize_ang(angle):
    return (angle + 2*np.pi) % (2*np.pi)

def sample_camera(x0=0, y0=0, z0=0, i_episode=None, use_predefined=False, nrow=None, ncol=None, img_pitches=None, img_yaws=None):
    # init
    radius = 1.0

    # sample pitch yaw
    if use_predefined:
        col = i_episode % ncol
        row = i_episode // ncol
        pitch = normalize_ang(img_pitches[row])
        yaw   = img_yaws[col]
    else:
        pitch = normalize_ang(np.random.uniform(low=-np.pi/2., high=np.pi/2))
        yaw   = np.random.uniform(low=0., high=2*np.pi)

    # rot_mat
    rot_mat = r_from_pitch_yaw(-pitch, yaw)

    # get camera head and tail
    head = np.array([x0, y0, z0]) - rot_mat.dot(np.array([radius-0.17, 0., 0.]))
    tail = np.array([x0, y0, z0]) - rot_mat.dot(np.array([radius, 0., 0.]))

    return head, tail, pitch, yaw

def sample_wrist(x0=0, y0=0, z0=0):
    # init
    radius = 0.17  # 0.38 - 0.21

    # sample height
    height_low = 0.01 #0.11
    height_high = 0.04 #0.14
    height = 0.05 #0.04 #np.random.uniform(height_low, height_high)

    # sample pitch yaw
    pitch = normalize_ang(np.random.uniform(low=-np.pi/2., high=np.pi/2))
    yaw   = np.random.uniform(low=0., high=2*np.pi)

    # rot_mat
    rot_mat = r_from_roll_yaw(pitch, yaw)

    # get camera head and tail
    head = np.array([x0, y0, z0]) - rot_mat.dot(np.array([0., radius-0.1, -height]))
    tail = np.array([x0, y0, z0]) - rot_mat.dot(np.array([0., radius, -height]))

    # set t horizon
    t_horizon = 25
    return head, tail, pitch, yaw, t_horizon

def run_camera_exp(
        environment, info, i_exp,
        start_ind_img_episode=0, n_img_episodes=100,
        use_predefined=False, nrow=None, ncol=None, img_pitches=None, img_yaws=None,
        ):
    # init
    images = []
    cameras = []

    # init env
    env = gym.make(environment)
    env.env.with_fixed_window_size=True

    # init env
    env.env.init(info)

    # init scene info for detacting collisions
    num_objs = info['num_objs']
    pos_objs = []
    for i in range(num_objs):
        pos = np.mean(info['obj_infos'][i]['parts_abs'], axis=0)
        pos_objs += [pos]

    # plot 2d map
    fig = plt.figure(figsize=(6, 6))
    axis0 = 0 #1
    axis1 = 1 #2
    for i in range(num_objs):
        pos = info['obj_infos'][i]['parts_abs']
        ax = plt.gca()
        ax.plot(pos[:, axis0], pos[:, axis1], 'ks')
        pos = info['obj_infos'][0]['parts_abs'][0]
        ax.text(pos[axis0]+0.02, pos[axis1]+0.02, 'shepard_metzler-{}'.format(i))
    ax.plot(0., 0., 'go')
    ax.axis('equal')
    ax.set_xlim([-0.7, 0.7])
    ax.set_ylim([-0.7, 0.7])

    # run episodes
    i_episode = start_ind_img_episode if not use_predefined else 0
    i_episode_end = start_ind_img_episode+n_img_episodes if not use_predefined else nrow*ncol 
    while i_episode < i_episode_end:
        # reset env
        observation = env.reset()

        # init time
        start_time = time.time()

        # init action
        action = np.zeros_like(env.action_space.sample())

        # select target object
        pos = info['obj_infos'][0]['pos']

        # sample camera positions
        camera_head, camera_tail, camera_pitch, camera_yaw = sample_camera(pos[0], pos[1], pos[2], i_episode, use_predefined, nrow, ncol, img_pitches, img_yaws)

        # apply sampled camera pos
        action[0] = camera_tail[0] # camera pos (x, y)
        action[1] = camera_tail[1] # camera pos (x, y)
        action[2] = camera_tail[2] # camera pos z
        action[3] = camera_pitch # camera direction
        action[4] = camera_yaw # camera direction

        # run multiple step for actuators to converge
        for acf in range(5):
            observation, reward, done, info = env.step(action)

        # render image
        img = env.render(mode='rgb_array')

        # crop 150 x 150 -> resize 256 x 256
        img = crop_center(img, 150, 150)

        # detect blackout
        idx0, idx1, idx2 = np.unravel_index(img.argmax(), img.shape)
        if img[idx0, idx1, idx2] < 1.:
            print("Episode failed (blackout) after {} timesteps ({:.3f} sec)".format(
                acf+1,
                time.time()-start_time))

            # close env
            env.close()

            # init env
            env.env.init(info)
        else:
            # add to data
            images += [img]
            cameras += [action]

            # plot 2d map (camera)
            ax = plt.gca()
            x  = camera_tail[axis0] #camera_tail[0]
            y  = camera_tail[axis1] #camera_tail[1]
            xx = camera_head[axis0] #camera_head[0]
            yy = camera_head[axis1] #camera_head[1]
            ax.plot(x, y, 'ro')
            ax.plot(xx, yy, 'bo')
            ax.quiver(x, y, xx-x, yy-y, angles='xy', scale_units='xy', scale=1)
            ax.text(x+0.02, y+0.02, 'camera-{}'.format(i_episode))
            ax.axis('equal')
            ax.set_xlim([-0.7, 0.7])
            ax.set_ylim([-0.7, 0.7])

            # done
            print("Episode finished after {} timesteps ({:.3f} sec)".format(
                acf+1,
                time.time()-start_time))

            # increase i_episode
            i_episode += 1

    # close env
    env.close()

    # save fig
    fig.canvas.draw()  # draw the canvas, cache the renderer
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    sampled_dir_cameras = img

    # close fig
    plt.close('all')

    # concate results
    images = np.stack(images, axis=0)
    cameras = np.stack(cameras, axis=0)

    return images, cameras, sampled_dir_cameras

def run_hand_exp(
        environment, info, i_exp,
        start_ind_hpt_episode=0, n_hpt_episodes=100,
        dt=1e-2, threshold=150,
        ):
    # init
    proprioceptives = []
    hands = []
    hand_images = []

    # init env
    env = gym.make(environment)
    env.env.with_fixed_window_size=True

    # init env
    env.env.init(info)

    # init scene info for detacting collisions
    num_objs = info['num_objs']
    pos_objs = []
    for i in range(num_objs):
        pos = info['obj_infos'][i]['parts_abs'][0]
        pos_objs += [pos]

    # plot 2d map
    fig = plt.figure(figsize=(6, 6))
    axis0 = 0 #1
    axis1 = 1 #2
    for i in range(num_objs):
        pos = info['obj_infos'][i]['parts_abs']
        ax = plt.gca()
        ax.plot(pos[:, axis0], pos[:, axis1], 'ks')
        pos = info['obj_infos'][0]['parts_abs'][0]
        ax.text(pos[axis0]+0.02, pos[axis1]+0.02, 'shepard_metzler-{}'.format(i))
    ax.plot(0., 0., 'go')
    ax.axis('equal')
    ax.set_xlim([-0.7, 0.7])
    ax.set_ylim([-0.7, 0.7])

    # set counter
    counter = 0

    # run episodes
    i_episode = start_ind_hpt_episode
    while i_episode < start_ind_hpt_episode+n_hpt_episodes:

        # reset env
        observation = env.reset()

        # init time
        start_time = time.time()

        # init action
        action = np.zeros_like(env.action_space.sample())

        # select target object
        pos = info['obj_infos'][0]['pos']

        # select action
        hand_head, hand_tail, hand_pitch, hand_yaw, actuator_convergence_frames = sample_wrist(pos[0], pos[1])

        # run multiple step for actuators to converge
        action = np.zeros_like(env.action_space.sample())
        for acf in range(actuator_convergence_frames):
            # apply heuristic policy
            action = gen_ornstein_uhlenbeck_process(action, dt=dt, sig=0., mu=env.action_space.high)

            # apply wrist pose and direction (pitch, yaw)
            action[13] = hand_tail[0] # hand pos
            action[14] = hand_tail[1] # hand pos
            action[15] = hand_tail[2] # hand pos  0.11 / 0.12 / 0.13 / 0.14 ellipsoid
            action[16] = hand_pitch
            action[17] = hand_yaw # hand pos  0.11 / 0.12 / 0.13 / 0.14 ellipsoid

            # forward (step)
            observation, reward, done, info = env.step(action)

        # render image
        img = env.render(mode='rgb_array')

        # crop 150 x 150
        img = crop_center(img, 150, 150)

        # check condition
        if len(hand_images) == 0:
            is_img_okay = True
        else:
            diff = hand_images[-1] - img
            is_img_okay = np.sum(np.abs(diff)) > 1
        idx = np.argmax(observation)
        if observation[idx] > threshold or not is_img_okay:
            # done
            print("Episode failed after {} timesteps ({:.3f} sec) | counter: {}".format(
                acf+1,
                time.time()-start_time,
                counter))

            # increase counter
            counter += 1

            if counter > 10:
                # close env
                env.close()

                # init env
                env.env.init(info)

                # reset counter
                counter = 0
        else:
            # add to data
            hand_images += [img]

            # add to data
            hands += [action[13:18]]
            proprioceptives += [observation]

            # plot 2d map (hand)
            x  = hand_tail[axis0] #hand_tail[0]
            y  = hand_tail[axis1] #hand_tail[1]
            xx = hand_head[axis0] #hand_head[0]
            yy = hand_head[axis1] #hand_head[1]
            ax = plt.gca()
            ax.plot(x, y, 'gH')
            ax.quiver(x, y, xx-x, yy-y, angles='xy', scale_units='xy', scale=1, color='g')
            ax.text(x+0.02, y+0.02, 'hand-{}'.format(i_episode))
            ax.axis('equal')
            ax.set_xlim([-0.7, 0.7])
            ax.set_ylim([-0.7, 0.7])

            # done
            print("Episode finished after {} timesteps ({:.3f} sec) {}".format(
                acf+1,
                time.time()-start_time,
                i_episode,
                ))

            # increase i_episode
            i_episode += 1

    # close env
    env.close()

    # save fig
    fig.canvas.draw()  # draw the canvas, cache the renderer
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    sampled_dir_hands = img

    # close fig
    plt.close()

    # concate results
    proprioceptives = np.stack(proprioceptives, axis=0)
    hands = np.stack(hands, axis=0)

    return hand_images, proprioceptives, hands, sampled_dir_hands
