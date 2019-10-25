import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding
from envs.haptix import haptix_env
from mujoco_py.generated import const

class BasicEnv(haptix_env.HaptixEnv):
    def __init__(self,
                 model_path,
                 n_substeps,
                 initial_qpos,
                 control_method,
                 control_hand_position):

        # init input arguments
        self.control_method = control_method
        assert self.control_method in ['absolute', 'centered', 'relative']
        self.control_hand_position = control_hand_position

        # set n_actuators
        self.n_actuators = 13

        # set n_positions
        if self.control_hand_position:
            self.n_positions = 3
        else: 
            self.n_positions = 0

        # viewer setup
        self.use_defined_camera = True

        # init class
        super(BasicEnv, self).__init__(
            model_path=model_path,
            n_substeps=n_substeps,
            initial_qpos=initial_qpos)

    # HaptixEnv methods
    # ----------------------------

    def _set_action(self, action):
        #''' split action '''
        #action_actuators = action[:self.n_actuators]  # actuators: 0-12 <- for MPL hand, 13-14 <- for camera
        #action_positions = action[self.n_actuators:]  # pos: 15-17 <- for MPL hand, 18-19 <- for camera

        ''' action_actuators for actuators '''
        action_actuators = action[:self.n_actuators]  # actuators: 0-12 <- for MPL hand, 13-14 <- for camera

        #assert action_actuators.shape == (13,)
        assert action_actuators.shape == (self.n_actuators,)

        if self.control_method == 'absolute':
            self.sim.data.ctrl[:] = action_actuators
        elif self.control_method == 'centered':
            ctrlrange = self.sim.model.actuator_ctrlrange
            actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
            actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
            self.sim.data.ctrl[:] = actuation_center + action_actuators * actuation_range
        elif self.control_method == 'relative':
            raise NotImplementedError()
            #actuation_center = np.zeros_like(action_actuators)
            #for i in range(self.sim.data.ctrl.shape[0]):
            #    actuation_center[i] = self.sim.data.get_joint_qpos(
            #        self.sim.model.actuator_names[i].replace(':A_', ':'))
            #for joint_name in ['FF', 'MF', 'RF', 'LF']:
            #    act_idx = self.sim.model.actuator_name2id(
            #        'robot0:A_{}J1'.format(joint_name))
            #    actuation_center[act_idx] += self.sim.data.get_joint_qpos(
            #        'robot0:{}J0'.format(joint_name))
            #self.sim.data.ctrl[:] = actuation_center + action_actuators * actuation_range
        else:
            raise NotImplementedError()
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

        ''' action for positions (note action = displacement!!!!!!!!!) '''
        def set_pos_diff(action):
            action_positions = action[self.n_actuators:]  # pos: 15-17 <- for MPL hand, 18-19 <- for camera

            assert action_positions.shape == (self.n_positions,)

            pos_diff = np.clip(
                    action_positions,
                    self.position_ctrlrange[:, 0],
                    self.position_ctrlrange[:, 1])
            return pos_diff
        if self.control_hand_position:
            pos_diff = set_pos_diff(action)
            self.sim.data.mocap_pos[0,:] += pos_diff
        else:
            pass

    def _set_ctrlrange(self):
        # actuators
        self.actuator_ctrlrange = self.model.actuator_ctrlrange.copy()  # get actuator control range

        # control for motion
        if self.control_hand_position:
            self.position_ctrlrange = np.array(
                    [[-0.01, 0.01],  # MPL hand x-axis
                     [-0.01, 0.01],  # MPL hand y-axis
                     [-0.01, 0.01],  # MPL hand z-axis
                     ])
        else:
            self.position_ctrlrange = None

        # concat
        self.ctrlrange = np.concatenate([
            self.actuator_ctrlrange,
            self.position_ctrlrange
            ]) if self.position_ctrlrange is not None else self.actuator_ctrlrange

    def _get_ctrlrange(self):
        bounds = self.ctrlrange
        low = bounds[:, 0]
        high = bounds[:, 1]

        return low, high

    def _viewer_setup(self):
        # Manual setting of the camera
        if not self.use_defined_camera: 
            body_id = self.sim.model.body_name2id('forearm')
            #body_id = self.sim.model.body_name2id('rgb_camera_arm')
            lookat = self.sim.data.body_xpos[body_id]
            for idx, value in enumerate(lookat):
                self.viewer.cam.lookat[idx] = value
            self.viewer.cam.distance = 1.5  # 0.5
            self.viewer.cam.azimuth = -100  # 55.
            self.viewer.cam.elevation = -30  # -25.
        else:
            # Use one of the cameras defined in the model as default
            # https://github.com/openai/mujoco-py/issues/10
            self.viewer.cam.type = const.CAMERA_FIXED
            self.viewer.cam.fixedcamid = 1
