import os
import math
import numpy as np
import xml
import xml.etree.ElementTree as elt
import tempfile

from gym import utils, error
from envs.haptix import cameractrl_env
from envs.haptix.utils import set_model_path

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


# choose random type
TYPES = ['box',
        'cylinder',
        'ellipsoid']

# model_paths
MODEL_PATHS = {'box': 'basic/MPL_CameraCtrl_Random_Box.xml',
               'cylinder': 'basic/MPL_CameraCtrl_Random_Cylinder.xml',
               'ellipsoid': 'basic/MPL_CameraCtrl_Random_Ellipsoid.xml'}


# msc func
def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
    quat /= np.linalg.norm(quat)
    return quat


# task class
class ManipulateEnv(cameractrl_env.CameraCtrlEnv, utils.EzPickle):
    def __init__(
        self,
        obj_type=None,
        #model_path,
        #target_position, target_rotation,
        #target_position_range, 
        #reward_type,
        initial_qpos={},
        #randomize_initial_position=True, randomize_initial_rotation=True,
        #distance_threshold=0.01, rotation_threshold=0.1,
        n_substeps=20,
        control_method='centered',
        #ignore_z_target_rotation=False,
        control_hand_position=False,
        control_camera_position=False,
    ):
        """Initializes a new CameraCtrl manipulation environment.

        Args:
            model_path (string): path to the environments XML file
            target_position (string): the type of target position:
                - ignore: target position is fully ignored, i.e. the object can be positioned arbitrarily
                - fixed: target position is set to the initial position of the object
                - random: target position is fully randomized according to target_position_range
            target_rotation (string): the type of target rotation:
                - ignore: target rotation is fully ignored, i.e. the object can be rotated arbitrarily
                - fixed: target rotation is set to the initial rotation of the object
                - xyz: fully randomized target rotation around the X, Y and Z axis
                - z: fully randomized target rotation around the Z axis
                - parallel: fully randomized target rotation around Z and axis-aligned rotation around X, Y
            ignore_z_target_rotation (boolean): whether or not the Z axis of the target rotation is ignored
            target_position_range (np.array of shape (3, 2)): range of the target_position randomization
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            randomize_initial_position (boolean): whether or not to randomize the initial position of the object
            randomize_initial_rotation (boolean): whether or not to randomize the initial rotation of the object
            distance_threshold (float, in meters): the threshold after which the position of a goal is considered achieved
            rotation_threshold (float, in radians): the threshold after which the rotation of a goal is considered achieved
            n_substeps (int): number of substeps the simulation runs on every call to step
            relative_control (boolean): whether or not the hand is actuated in absolute joint positions or relative to the current state
        """
        #self.target_position = target_position
        #self.target_rotation = target_rotation
        #self.target_position_range = target_position_range
        #self.parallel_quats = [rotations.euler2quat(r) for r in rotations.get_parallel_rotations()]
        #self.randomize_initial_rotation = randomize_initial_rotation
        #self.randomize_initial_position = randomize_initial_position
        #self.distance_threshold = distance_threshold
        #self.rotation_threshold = rotation_threshold
        #self.reward_type = reward_type
        #self.ignore_z_target_rotation = ignore_z_target_rotation

        #assert self.target_position in ['ignore', 'fixed', 'random']
        #assert self.target_rotation in ['ignore', 'fixed', 'xyz', 'z', 'parallel']

        ''' init type '''
        if obj_type is None:
            obj_type = np.random.choice(TYPES, 1)[0]
        self.type = obj_type

        ''' init configs '''
        self.configs = {}
        self.configs['n_substeps']=n_substeps
        self.configs['initial_qpos']=initial_qpos
        self.configs['control_method']=control_method
        self.configs['control_hand_position']=control_hand_position
        self.configs['control_camera_position']=control_camera_position

        ''' init class '''
        self.init()

    #def _get_achieved_goal(self):
    #    # Object position and rotation.
    #    object_qpos = self.sim.data.get_joint_qpos('object:joint')
    #    assert object_qpos.shape == (7,)
    #    return object_qpos

    #def _goal_distance(self, goal_a, goal_b):
    #    assert goal_a.shape == goal_b.shape
    #    assert goal_a.shape[-1] == 7

    #    d_pos = np.zeros_like(goal_a[..., 0])
    #    d_rot = np.zeros_like(goal_b[..., 0])
    #    if self.target_position != 'ignore':
    #        delta_pos = goal_a[..., :3] - goal_b[..., :3]
    #        d_pos = np.linalg.norm(delta_pos, axis=-1)

    #    if self.target_rotation != 'ignore':
    #        quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]

    #        if self.ignore_z_target_rotation:
    #            # Special case: We want to ignore the Z component of the rotation.
    #            # This code here assumes Euler angles with xyz convention. We first transform
    #            # to euler, then set the Z component to be equal between the two, and finally
    #            # transform back into quaternions.
    #            euler_a = rotations.quat2euler(quat_a)
    #            euler_b = rotations.quat2euler(quat_b)
    #            euler_a[2] = euler_b[2]
    #            quat_a = rotations.euler2quat(euler_a)

    #        # Subtract quaternions and extract angle between them.
    #        quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
    #        angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
    #        d_rot = angle_diff
    #    assert d_pos.shape == d_rot.shape
    #    return d_pos, d_rot

    # GoalEnv methods
    # ----------------------------

    #def compute_reward(self, achieved_goal, goal, info):
    #    if self.reward_type == 'sparse':
    #        success = self._is_success(achieved_goal, goal).astype(np.float32)
    #        return (success - 1.)
    #    else:
    #        d_pos, d_rot = self._goal_distance(achieved_goal, goal)
    #        # We weigh the difference in position to avoid that `d_pos` (in meters) is completely
    #        # dominated by `d_rot` (in radians).
    #        return -(10. * d_pos + d_rot)

    # RobotEnv methods
    # ----------------------------

    #def _is_success(self, achieved_goal, desired_goal):
    #    d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
    #    achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
    #    achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
    #    achieved_both = achieved_pos * achieved_rot
    #    return achieved_both

    #def _env_setup(self, initial_qpos, initial_qvel=None):
    #    for name, value in initial_qpos.items():
    #        self.sim.data.set_joint_qpos(name, value)
    #    self.sim.forward()
    def _env_setup(self, initial_qpos, initial_qvel=None, initial_mpos=None):
        # init qpos and qvel
        qpos = initial_qpos
        qvel = initial_qvel
        if initial_qvel is None:
            qvel = self.init_qvel
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)

        # init mpos
        self.sim.data.mocap_pos[:, :] = initial_mpos

        # sim forward
        self.sim.forward()

    def _reset_sim(self):
        #self.sim.set_state(self.initial_state)
        #self.sim.forward()
        #self._env_setup(initial_qpos=self.init_qpos)
        self._env_setup(initial_qpos=self.init_qpos,
                        initial_qvel=self.init_qvel,
                        initial_mpos=self.init_mpos)

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            #self._set_action(np.zeros(13))
            self._set_action(np.zeros(self.n_actuators + self.n_positions))
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False

        return True

        #initial_qpos = self.sim.data.get_joint_qpos('object:joint').copy()
        #initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        #assert initial_qpos.shape == (7,)
        #assert initial_pos.shape == (3,)
        #assert initial_quat.shape == (4,)
        #initial_qpos = None

        ## Randomization initial rotation.
        #if self.randomize_initial_rotation:
        #    if self.target_rotation == 'z':
        #        angle = self.np_random.uniform(-np.pi, np.pi)
        #        axis = np.array([0., 0., 1.])
        #        offset_quat = quat_from_angle_and_axis(angle, axis)
        #        initial_quat = rotations.quat_mul(initial_quat, offset_quat)
        #    elif self.target_rotation == 'parallel':
        #        angle = self.np_random.uniform(-np.pi, np.pi)
        #        axis = np.array([0., 0., 1.])
        #        z_quat = quat_from_angle_and_axis(angle, axis)
        #        parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
        #        offset_quat = rotations.quat_mul(z_quat, parallel_quat)
        #        initial_quat = rotations.quat_mul(initial_quat, offset_quat)
        #    elif self.target_rotation in ['xyz', 'ignore']:
        #        angle = self.np_random.uniform(-np.pi, np.pi)
        #        axis = self.np_random.uniform(-1., 1., size=3)
        #        offset_quat = quat_from_angle_and_axis(angle, axis)
        #        initial_quat = rotations.quat_mul(initial_quat, offset_quat)
        #    elif self.target_rotation == 'fixed':
        #        pass
        #    else:
        #        raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))

        ## Randomize initial position.
        #if self.randomize_initial_position:
        #    if self.target_position != 'fixed':
        #        initial_pos += self.np_random.normal(size=3, scale=0.005)

        #initial_quat /= np.linalg.norm(initial_quat)
        #initial_qpos = np.concatenate([initial_pos, initial_quat])
        #self.sim.data.set_joint_qpos('object:joint', initial_qpos)

        #def is_on_palm():
        #    self.sim.forward()
        #    cube_middle_idx = self.sim.model.site_name2id('object:center')
        #    cube_middle_pos = self.sim.data.site_xpos[cube_middle_idx]
        #    is_on_palm = (cube_middle_pos[2] > 0.04)
        #    return is_on_palm

        ## Run the simulation for a bunch of timesteps to let everything settle in.
        #for _ in range(10):
        #    self._set_action(np.zeros(20))
        #    try:
        #        self.sim.step()
        #    except mujoco_py.MujocoException:
        #        return False
        #return is_on_palm()

    #def _sample_goal(self):
    #    # Select a goal for the object position.
    #    target_pos = None
    #    if self.target_position == 'random':
    #        assert self.target_position_range.shape == (3, 2)
    #        offset = self.np_random.uniform(self.target_position_range[:, 0], self.target_position_range[:, 1])
    #        assert offset.shape == (3,)
    #        target_pos = self.sim.data.get_joint_qpos('object:joint')[:3] + offset
    #    elif self.target_position in ['ignore', 'fixed']:
    #        target_pos = self.sim.data.get_joint_qpos('object:joint')[:3]
    #    else:
    #        raise error.Error('Unknown target_position option "{}".'.format(self.target_position))
    #    assert target_pos is not None
    #    assert target_pos.shape == (3,)

    #    # Select a goal for the object rotation.
    #    target_quat = None
    #    if self.target_rotation == 'z':
    #        angle = self.np_random.uniform(-np.pi, np.pi)
    #        axis = np.array([0., 0., 1.])
    #        target_quat = quat_from_angle_and_axis(angle, axis)
    #    elif self.target_rotation == 'parallel':
    #        angle = self.np_random.uniform(-np.pi, np.pi)
    #        axis = np.array([0., 0., 1.])
    #        target_quat = quat_from_angle_and_axis(angle, axis)
    #        parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
    #        target_quat = rotations.quat_mul(target_quat, parallel_quat)
    #    elif self.target_rotation == 'xyz':
    #        angle = self.np_random.uniform(-np.pi, np.pi)
    #        axis = self.np_random.uniform(-1., 1., size=3)
    #        target_quat = quat_from_angle_and_axis(angle, axis)
    #    elif self.target_rotation in ['ignore', 'fixed']:
    #        target_quat = self.sim.data.get_joint_qpos('object:joint')
    #    else:
    #        raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))
    #    assert target_quat is not None
    #    assert target_quat.shape == (4,)

    #    target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
    #    goal = np.concatenate([target_pos, target_quat])
    #    return goal

    #def _render_callback(self):
    #    # Assign current state to target object but offset a bit so that the actual object
    #    # is not obscured.
    #    goal = self.goal.copy()
    #    assert goal.shape == (7,)
    #    if self.target_position == 'ignore':
    #        # Move the object to the side since we do not care about it's position.
    #        goal[0] += 0.15
    #    self.sim.data.set_joint_qpos('target:joint', goal)
    #    self.sim.data.set_joint_qvel('target:joint', np.zeros(6))

    #    if 'object_hidden' in self.sim.model.geom_names:
    #        hidden_id = self.sim.model.geom_name2id('object_hidden')
    #        self.sim.model.geom_rgba[hidden_id, 3] = 1.
    #    self.sim.forward()

    def _get_obs(self):
        return self.sim.data.sensordata.copy()
        #robot_qpos, robot_qvel = robot_get_obs(self.sim)
        #object_qvel = self.sim.data.get_joint_qvel('object:joint')
        #achieved_goal = self._get_achieved_goal().ravel()  # this contains the object position + rotation
        #observation = np.concatenate([robot_qpos, robot_qvel, object_qvel, achieved_goal])
        #return {
        #    'observation': observation.copy(),
        #    'achieved_goal': achieved_goal.copy(),
        #    'desired_goal': self.goal.ravel().copy(),
        #}

    def _get_model_xml(self, full_path, info):
        ''' get model '''
        # load and parse xml
        model_tree = elt.parse(full_path)
        root = model_tree.getroot()

        ''' update model '''
        # update position
        element = root.findall(".//body[@name='{}']".format(self.type))[0]
        element.set('pos', '{:.5f} {:.5f} {:.5f}'.format(info['pos'][0], info['pos'][1], info['pos'][2]))

        # update rotation
        element = root.findall(".//body[@name='{}']".format(self.type))[0]
        element.set('axisangle', '0.0 0.0 1.0 {:.5f}'.format(float(info['axisangle'])))

        # update size
        geom = element.findall(".//geom[@type='{}']".format(self.type))[0]
        geom.set('size', info['size'])

        # update material
        geom = element.findall(".//geom[@type='{}']".format(self.type))[0]
        geom.set('material', info['material'])

        ''' cleanup xml '''
        model_xml = elt.tostring(model_tree.getroot())

        return model_xml

    def _get_info(self, full_path):
        ''' init info '''
        info = {}

        ''' check type '''
        assert self.type
        info['type'] = self.type

        ''' get model '''
        # load and parse xml
        model_tree = elt.parse(full_path)
        root = model_tree.getroot()

        ''' update model '''
        # update position
        element = root.findall(".//body[@name='{}']".format(self.type))[0]
        init_pos = np.fromstring(element.get('pos'), dtype=float, sep=' ')
        #scale = 0.1
        #modified_pos = scale * np.random.normal(size=init_pos.shape) + init_pos
        #modified_pos = np.array2string(modified_pos, precision=5, separator=' ')
        #element.set('pos', '0.1 0.2 0.1')
        #element.set('pos', modified_pos)
        info['pos'] = init_pos

        # update rotation
        element = root.findall(".//body[@name='{}']".format(self.type))[0]
        #init_axisangle = np.fromstring(element.get('axisangle'), dtype=float, sep=' ')
        modified_axisangle = np.random.uniform(low=-math.pi, high=math.pi, size=(1,))
        #element.set('axisangle', '0.0 0.0 1.0 {:.5f}'.format(float(modified_axisangle)))
        info['axisangle'] = modified_axisangle

        # update size
        geom = element.findall(".//geom[@type='{}']".format(self.type))[0]
        init_size = np.fromstring(geom.get('size'), dtype=float, sep=' ')
        modified_size = np.random.uniform(low=0.8, high=1.2, size=init_size.shape) * init_size
        modified_size = np.array2string(modified_size, precision=5, separator=' ')[1:-1]
        #geom.set('size', modified_size)
        #geom.set('size', '0.02 0.04 0.05')
        info['size'] = modified_size

        # update material
        geom = element.findall(".//geom[@type='{}']".format(self.type))[0]
        materials = ["MatFoil",
                "MatPlane",
                "MatWood",
                "MatSquare",
                "MatWoodR",
                "MatWoodG",
                "MatWoodB"]
        modified_material = np.random.choice(materials, 1)[0]
        #geom.set('material', modified_material)
        info['material'] = modified_material

        return info

    def init(self, info=None):
        ''' check configs '''
        assert self.configs
        n_substeps=self.configs['n_substeps']
        initial_qpos=self.configs['initial_qpos']
        control_method=self.configs['control_method']
        control_hand_position=self.configs['control_hand_position']
        control_camera_position=self.configs['control_camera_position']

        ''' get model_path '''
        if info is not None:
            self.type = info['type']
        model_path = MODEL_PATHS[self.type]

        ''' get model path '''
        full_path, prefix_path = set_model_path(model_path)

        ''' generate info '''
        if info is None: 
            info = self._get_info(full_path)

        ''' add current model_path to info '''
        info['model_path'] = model_path

        ''' get model xml '''
        model_xml = self._get_model_xml(full_path, info)

        ''' set info '''
        self.info = info

        ''' init environment '''
        with tempfile.NamedTemporaryFile(suffix=".xml", dir=os.path.join(prefix_path, 'basic')) as fp:
            # write temporary xml file
            fp.write(model_xml)
            fp.flush()

            # set up modified model path
            modified_model_path=fp.name

            # init environment
            cameractrl_env.CameraCtrlEnv.__init__(
                self,
                modified_model_path,
                n_substeps=n_substeps,
                initial_qpos=initial_qpos,
                control_method=control_method,
                control_hand_position=control_hand_position,
                control_camera_position=control_camera_position)

        utils.EzPickle.__init__(self)


class CameraCtrlBoxEnv(ManipulateEnv):
    #def __init__(self, target_position='random', target_rotation='xyz', reward_type='sparse'):
    def __init__(self):
        super(CameraCtrlBoxEnv, self).__init__(
            obj_type='box',  # model_path='MPL_CameraCtrl_Random_Box.xml',  # model_path='MPL_Basic.xml',
            #target_position=target_position,
            #target_rotation=target_rotation,
            #target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            #reward_type=reward_type
            )

class CameraCtrlEllipsoidEnv(ManipulateEnv):
    def __init__(self):
        super(CameraCtrlEllipsoidEnv, self).__init__(
            obj_type='ellipsoid',  # model_path='MPL_CameraCtrl_Random_Ellipsoid.xml',
            )

class CameraCtrlCylinderEnv(ManipulateEnv):
    def __init__(self):
        super(CameraCtrlCylinderEnv, self).__init__(
            obj_type='cylinder',  # model_path='MPL_CameraCtrl_Random_Cylinder.xml',
            )

class CameraCtrlRandomEnv(ManipulateEnv):
    def __init__(self):
        super(CameraCtrlRandomEnv, self).__init__(
            obj_type=None,  # model_path=model_path,
            )

class CameraCtrlBoxPosCtrlEnv(ManipulateEnv):
    def __init__(self):
        super(CameraCtrlBoxPosCtrlEnv, self).__init__(
            obj_type='box',  # model_path='MPL_CameraCtrl_Random_Box.xml',
            control_hand_position=True,
            control_camera_position=True
            )

class CameraCtrlEllipsoidPosCtrlEnv(ManipulateEnv):
    def __init__(self):
        super(CameraCtrlEllipsoidPosCtrlEnv, self).__init__(
            obj_type='ellipsoid',  # model_path='MPL_CameraCtrl_Random_Ellipsoid.xml',
            control_hand_position=True,
            control_camera_position=True
            )

class CameraCtrlCylinderPosCtrlEnv(ManipulateEnv):
    def __init__(self):
        super(CameraCtrlCylinderPosCtrlEnv, self).__init__(
            obj_type='cylinder',  # model_path='MPL_CameraCtrl_Random_Cylinder.xml',
            control_hand_position=True,
            control_camera_position=True
            )

class CameraCtrlRandomPosCtrlEnv(ManipulateEnv):
    def __init__(self):
        super(CameraCtrlRandomPosCtrlEnv, self).__init__(
            obj_type=None,  # model_path=model_path,
            control_hand_position=True,
            control_camera_position=True
            )
