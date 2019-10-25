import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding
from envs.haptix import haptix_env
from mujoco_py.generated import const
from envs.haptix.utils import quat_from_angle_and_axis

class RoomsEnv(haptix_env.HaptixEnv):
    def __init__(self,
                 model_path,
                 n_substeps,
                 initial_qpos,
                 control_method,
                 #control_hand_position,
                 control_camera_position):

        # init input arguments
        self.control_method = control_method
        assert self.control_method in ['absolute', 'centered', 'relative']
        #self.control_hand_position = control_hand_position
        self.control_camera_position = control_camera_position

        # set n_actuators
        self.n_actuators = 0 #15

        # set n_positions
        #if self.control_hand_position and self.control_camera_position:
        #    self.n_positions = 6
        #elif self.control_hand_position or self.control_camera_position:
        #    self.n_positions = 3
        #else:
        #    self.n_positions = 0
        if self.control_camera_position:
            self.n_positions = 3
        else:
            self.n_positions = 0

        # set n_angles
        self.n_angles = 1

        # viewer setup
        self.use_defined_camera = True

        # init class
        super().__init__(
            model_path=model_path,
            n_substeps=n_substeps,
            initial_qpos=initial_qpos)

    # HaptixEnv methods
    # ----------------------------

    def _set_action(self, action):
        #''' split action '''
        #action_actuators = action[:self.n_actuators]  # actuators: 0-1 <- for camera
        #action_positions = action[self.n_actuators:]  # pos: 2-4 <- for camera

        #''' action_actuators for actuators '''
        #action_actuators = action[:self.n_actuators]  # actuators: 0-1 <- for camera

        ##assert action_actuators.shape == (13,)
        #assert action_actuators.shape == (self.n_actuators,)

        #ctrlrange = self.sim.model.actuator_ctrlrange
        #if self.control_method == 'absolute':
        #    self.sim.data.ctrl[:] = action_actuators
        #elif self.control_method == 'centered':
        #    actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
        #    actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
        #    self.sim.data.ctrl[:] = actuation_center + action_actuators * actuation_range
        #elif self.control_method == 'relative':
        #    raise NotImplementedError()
        #    #actuation_center = np.zeros_like(action_actuators)
        #    #for i in range(self.sim.data.ctrl.shape[0]):
        #    #    actuation_center[i] = self.sim.data.get_joint_qpos(
        #    #        self.sim.model.actuator_names[i].replace(':A_', ':'))
        #    #for joint_name in ['FF', 'MF', 'RF', 'LF']:
        #    #    act_idx = self.sim.model.actuator_name2id(
        #    #        'robot0:A_{}J1'.format(joint_name))
        #    #    actuation_center[act_idx] += self.sim.data.get_joint_qpos(
        #    #        'robot0:{}J0'.format(joint_name))
        #    #self.sim.data.ctrl[:] = actuation_center + action_actuators * actuation_range
        #else:
        #    raise NotImplementedError()
        #self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

        ''' action for positions '''
        def set_pos(action_positions):
            assert action_positions.shape == (self.n_positions,)

            # clip pos with control range
            pos = np.clip(
                    action_positions,
                    self.position_ctrlrange[:, 0],
                    self.position_ctrlrange[:, 1])

            # relative control
            if self.control_method == 'relative':
                current_pos = self.sim.data.mocap_pos.copy().reshape(self.n_positions, )
                pos += current_pos

            return pos

        ''' action for angles '''
        def set_angle(action_angles):
            #assert action_angles.shape == (self.n_angles,)

            # clip pos with control range
            angle = np.clip(
                    action_angles,
                    self.angle_ctrlrange[:, 0],
                    self.angle_ctrlrange[:, 1])

            # relative control
            if self.control_method == 'relative':
                raise NotImplementedError()

            # original hand angle start from np.pi
            return angle

        # set pos
        #if self.control_hand_position and self.control_camera_position:
        #    pos = set_pos(action)
        #    self.sim.data.mocap_pos[:,:] = pos.reshape(2, 3)
        #elif self.control_hand_position:
        #    pos = set_pos(action)
        #    self.sim.data.mocap_pos[0,:] = pos
        #elif self.control_camera_position:
        #    pos = set_pos(action)
        #    self.sim.data.mocap_pos[1,:] = pos
        #else:
        #    pass
        if self.control_camera_position:
            pos = set_pos(action[self.n_actuators:self.n_actuators+self.n_positions])  # pos: 0-2 <- for camera
            self.sim.data.mocap_pos[0,:] = pos

            # set angle
            angle = set_angle(action[-1])  # pos: 3 <- for MPL hand
            quat = quat_from_angle_and_axis(angle[0], np.array([0., 0., 1.]))
            self.sim.data.mocap_quat[0,:] = quat
        else:
            pass

    def _set_ctrlrange(self):
        ## actuators
        #self.actuator_ctrlrange = self.model.actuator_ctrlrange.copy()  # get actuator control range

        # control for motion
        pos_ctrl_range = 0.01 if self.control_method == 'relative' else 0.7
        yaw_range = 2 * np.pi
        #if self.control_hand_position and self.control_camera_position:
        #    self.position_ctrlrange = np.array(
        #            [[-pos_ctrl_range, pos_ctrl_range],  # MPL hand x-axis
        #             [-pos_ctrl_range, pos_ctrl_range],  # MPL hand y-axis
        #             [-pos_ctrl_range, pos_ctrl_range],  # MPL hand z-axis
        #             [-pos_ctrl_range, pos_ctrl_range],  # camera x-axis
        #             [-pos_ctrl_range, pos_ctrl_range],  # camera y-axis
        #             [-pos_ctrl_range, pos_ctrl_range],  # camera z-axis
        #             ])
        #elif self.control_hand_position or self.control_camera_position:
        #    self.position_ctrlrange = np.array(
        #            [[-pos_ctrl_range, pos_ctrl_range],  # MPL hand or camera x-axis
        #             [-pos_ctrl_range, pos_ctrl_range],  # MPL hand or camera y-axis
        #             [-pos_ctrl_range, pos_ctrl_range],  # MPL hand or camera z-axis
        #             ])
        #else:
        #    self.position_ctrlrange = None
        if self.control_camera_position:
            self.position_ctrlrange = np.array(
                    [[-pos_ctrl_range, pos_ctrl_range],  # camera x-axis
                     [-pos_ctrl_range, pos_ctrl_range],  # camera y-axis
                     [-pos_ctrl_range, pos_ctrl_range],  # camera z-axis
                     ])
            self.angle_ctrlrange = np.array(
                    [[0., yaw_range],  # MPL hand or camera yaw z-axis
                     ])
        else:
            self.position_ctrlrange = None

        # concat
        self.ctrlrange = np.concatenate([
            #self.actuator_ctrlrange,
            self.position_ctrlrange,
            self.angle_ctrlrange,
            ])# if self.position_ctrlrange is not None else self.actuator_ctrlrange

    def _get_ctrlrange(self):
        bounds = self.ctrlrange
        low = bounds[:, 0]
        high = bounds[:, 1]

        return low, high

    def _viewer_setup(self):
        # Manual setting of the camera
        if not self.use_defined_camera: 
            body_id = self.sim.model.body_name2id('obj0')
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
            self.viewer.cam.fixedcamid = 0
