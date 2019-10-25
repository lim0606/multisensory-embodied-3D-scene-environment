import os
import math
import colorsys
import numpy as np
import xml
import xml.etree.ElementTree as elt
import tempfile
from copy import deepcopy

from gym import utils, error
from envs.haptix import shepard_metzler_hand_camera_env
from envs.haptix.utils import set_model_path, r_from_angle_axis, quat_from_angle_and_axis

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


# (obj) material types
OBJ_MATERIALS = ["MatFoil",
                 "MatPlane",
                 "MatWood",
                 "MatSquare",
                 "MatWoodR",
                 "MatWoodG",
                 "MatWoodB"]

# (wall) material types
WALL_MATERIALS = ["MatFoil",
                  "MatPlane",
                  "MatWood",
                  "MatSquare",
                  "MatWoodR",
                  "MatWoodG",
                  "MatWoodB"]

# (floor) material types
FLOOR_MATERIALS = ["table2d"]

# model_paths
TEMPLATE_PATH = 'shepard_metzler/template_hand_camera.xml'
MODEL_PATH = 'shepard_metzler/box.xml'
ROOM_PATH = 'shepard_metzler/room.xml'


# gen_shepard_metzler_3d_obj generation func
def gen_shepard_metzler_3d_obj(num_parts, scale=1., do_random_rotation=False, do_random_branch=False):
    # check condition
    assert num_parts % 2 == 1

    # define directions
    directions = scale * np.array([
        [ 1.,  0.,  0.],
        [-1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0., -1.,  0.],
        [ 0.,  0.,  1.],
        [ 0.,  0., -1.]], dtype=np.float32)
    num_directions = directions.shape[0]

    # generate shepard metzler 3d objects
    parts_abs = [np.zeros((3,), dtype=np.float32)]
    parts_rel = [np.zeros((3,), dtype=np.float32)]
    for i in range(num_parts-1):
        # get new part such that no collision happen
        is_collide = True
        while is_collide:
            # random sample direction
            index = np.random.randint(num_directions)

            # get prev_part_abs
            if do_random_branch:
                prev_part_abs = parts_abs[np.random.randint(len(parts_abs))]
            else:
                prev_part_abs = parts_abs[-1]

            # get new part
            new_part_abs = parts_abs[-1] + directions[index]
            new_part_rel = directions[index]

            # check new part collide existing parts
            is_collide = False
            for part in parts_abs:
                diff = np.linalg.norm(new_part_abs - part)
                if diff < 0.01:
                    is_collide = True

        parts_abs += [new_part_abs]
        parts_rel += [new_part_rel]

    # concatenate parts to one array
    parts_abs = np.stack(parts_abs, axis=0)
    parts_rel = np.stack(parts_rel, axis=0)

    # estimate center
    mean = np.mean(parts_abs, axis=0)

    if do_random_rotation:
        # sample random rotation
        axis = np.random.normal(0, 1., (3,))
        axis /= np.linalg.norm(axis)
        angle = np.random.uniform(0., 2.*np.pi)
    else:
        axis = np.array([0., 0., 1.], dtype=np.float32)
        angle = 0.

    # from axis + angle to rotation matrix
    r_matrix = r_from_angle_axis(angle, axis)

    # quat
    quat = quat_from_angle_and_axis(angle, axis)

    # rotate abs coord
    parts_abs = r_matrix.dot(parts_abs.T).T

    # apply rotation matrix
    mean_rot = r_matrix.dot(mean)

    # normalize center
    parts_abs -= mean_rot
    parts_rel[0] -= mean_rot

    # return
    return angle, axis, quat, r_matrix, -mean_rot, -mean, parts_rel, parts_abs


# task class
class ManipulateEnv(shepard_metzler_hand_camera_env.ShepardMetzlerEnv, utils.EzPickle):
    def __init__(
        self,
        #model_path,
        #target_position, target_rotation,
        #target_position_range, 
        #reward_type,
        initial_qpos={},
        #randomize_initial_position=True, randomize_initial_rotation=True,
        #distance_threshold=0.01, rotation_threshold=0.1,
        n_substeps=20,
        control_method='absolute',  # 'centered',
        #ignore_z_target_rotation=False,
        control_hand_position=False,
        control_camera_position=False,
        max_num_objs=3,
        rand_pos=True,
        num_parts=5,
        do_random_branch=False,
    ):
        """Initializes a new ShepardMetzler manipulation environment.

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

        ''' init configs '''
        self.configs = {}
        self.configs['n_substeps']=n_substeps
        self.configs['initial_qpos']=initial_qpos
        self.configs['control_method']=control_method
        self.configs['control_hand_position']=control_hand_position
        self.configs['control_camera_position']=control_camera_position
        self.configs['max_num_objs']=max_num_objs
        self.configs['rand_pos']=rand_pos
        self.configs['num_parts']=num_parts
        self.configs['do_random_branch']=do_random_branch

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
            action = np.zeros(self.n_actuators + self.n_positions + self.n_angles, dtype=np.float32) #self.init_mpos
            if self.control_hand_position and self.control_camera_position:
                action[self.n_actuators:self.n_actuators+3] = self.init_mpos[0,:]
                action[self.n_actuators+3+1:self.n_actuators+3+1+3] = self.init_mpos[1,:]
            elif self.control_hand_position:
                action[self.n_actuators:self.n_actuators+self.n_positions] = self.init_mpos[0,:]
            elif self.control_camera_position:
                action[self.n_actuators:self.n_actuators+self.n_positions] = self.init_mpos[1,:]
            self._set_action(action)
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

    def _gen_scene_xml(self, full_path, obj_elts, table_elt=None):
        ''' get model '''
        # load and parse xml
        model_tree = elt.parse(full_path)
        root = model_tree.getroot()
        worldbody = root.findall('worldbody')[0]

        ''' add objects '''
        # add all object xml to worldbody
        for obj_elt in obj_elts:
            worldbody.append(obj_elt)

        ''' add table (wall + floor) '''
        if table_elt is not None:
            for _elt in table_elt:
                worldbody.append(_elt)

        ''' cleanup xml '''
        model_xml = elt.tostring(model_tree.getroot())

        return model_xml

    def _get_obj_info_elt(self, full_path, index, num_parts=5, rand_pos=True, do_random_branch=False):
        ''' init info '''
        info = {}

        ''' save index '''
        info['index'] = index

        ''' set scale '''
        scale = 0.0175
        info['scale'] = scale

        ''' random pos '''
        pos = np.random.uniform(low=-.3, high=0.3, size=(3,), dtype=np.float32) if rand_pos else np.zeros((3,), dtype=np.float32)
        pos[2] = 0. #0.1
        info['pos'] = pos

        ''' generate gen_shepard_metzler_3d_obj (as a set of points) '''
        angle, axis, quat, r_matrix, cm_rot, cm, parts_rel, parts_abs = \
                gen_shepard_metzler_3d_obj(num_parts, 2.*scale, do_random_branch=do_random_branch)
        info['angle'] = angle
        info['axis'] = axis
        info['quat'] = quat
        info['r_matrix'] = r_matrix
        info['cm_rot'] = cm_rot
        info['cm'] = cm

        ''' move body to rand pos '''
        parts_rel[0] += pos
        for i in range(len(parts_abs)):
            parts_abs[i] += pos
        info['parts_rel'] = parts_rel
        info['parts_abs'] = parts_abs

        ''' gnerate colors '''
        #info['colors'] = [np.random.choice(OBJ_MATERIALS, 1)[0] for i in range(num_parts)]
        info['hsv'] = [[np.random.uniform(0., 1.), np.random.uniform(0.75, 1.), 1.] for i in range(num_parts)]
        info['rgb'] = [colorsys.hsv_to_rgb(*info['hsv'][i]) for i in range(num_parts)]

        ''' get model (box) '''
        # load and parse xml
        model_tree = elt.parse(full_path)
        root = model_tree.getroot()
        element = root.findall(".//body[@name='box']")[0]

        ''' generate gen_shepard_metzler_3d_obj (as a set of points) '''
        parts_elt = []
        for i, part in enumerate(parts_rel):
            # copy element
            curr_elt = element if i == 0 else deepcopy(element)

            # update name
            curr_elt.set('name', 'obj{}-part{}'.format(index, i))

            # update position
            curr_elt.set('pos', '{:.5f} {:.5f} {:.5f}'.format(part[0], part[1], part[2]))

            ## update material
            #geom = curr_elt.findall(".//geom[@type='box']")[0]
            #geom.set('material', info['colors'][i])

            # update color
            geom = curr_elt.findall(".//geom[@type='box']")[0]
            geom.set('rgba', '{:.5f} {:.5f} {:.5f} 1'.format(*info['rgb'][i]))

            # update size
            geom = curr_elt.findall(".//geom[@type='box']")[0]
            geom.set('size', '{:.5f} {:.5f} {:.5f}'.format(scale, scale, scale))

            # append to parts_elt
            parts_elt += [curr_elt]

        ''' chain parts '''
        for i in range(1, len(parts_elt)):
            parts_elt[i-1].append(parts_elt[i])

        ''' apply rotation '''
        parts_elt[0].set('axisangle', '{:.5f} {:.5f} {:.5f} {:.5f}'.format(axis[0], axis[1], axis[2], angle))
        #parts_elt[0].set('quat', '{:.5f} {:.5f} {:.5f} {:.5f}'.format(quat[0], quat[1], quat[2], quat[3]))

        return info, parts_elt[0]

    def _gen_table_elt(self, full_path, info):
        ''' get model '''
        # load and parse xml
        model_tree = elt.parse(full_path)
        root = model_tree.getroot()

        ''' update model '''
        # update light
        light = root.findall('.//light')[0]
        light.set('pos', '{:.5f} {:.5f} {:.5f}'.format(info['light_pos'][0], info['light_pos'][1], info['light_pos'][2]))

        # update material of floor
        geom = root.findall(".//geom[@name='floor']")[0]
        geom.set('material', info['floor_material'])

        # update material of walls
        for i in range(1, 4+1):
            geom = root.findall(".//geom[@name='wall{}']".format(i))[0]
            geom.set('material', info['wall_material'])

        # children of worldbody
        worldbody = root.findall('worldbody')[0]
        return worldbody.getchildren()

    def _get_table_info(self):
        ''' init info '''
        info = {}

        # update light
        modified_pos = np.random.uniform(low=-.4, high=0.4, size=(3,), dtype=np.float32)
        modified_pos[2] = 1.5 # fix z position
        info['light_pos'] = modified_pos

        # update material of floor
        material = np.random.choice(FLOOR_MATERIALS, 1)[0]
        info['floor_material'] = material

        # update material of walls
        material = np.random.choice(WALL_MATERIALS, 1)[0]
        info['wall_material'] = material

        return info

    def init(self, info=None):
        ''' check configs '''
        assert self.configs
        n_substeps=self.configs['n_substeps']
        initial_qpos=self.configs['initial_qpos']
        control_method=self.configs['control_method']
        control_hand_position=self.configs['control_hand_position']
        control_camera_position=self.configs['control_camera_position']
        max_num_objs=self.configs['max_num_objs']
        rand_pos=self.configs['rand_pos']
        num_parts=self.configs['num_parts']
        do_random_branch=self.configs['do_random_branch']

        ''' get template_path '''
        template_path = TEMPLATE_PATH
        full_path, prefix_path = set_model_path(template_path)

        ''' set (scene) info '''
        if info is None:
            info = {}
            obj_elts = []
            obj_infos = []

            ''' number of objects '''
            num_objs = np.random.randint(max_num_objs)+1
            info['num_objs'] = num_objs

            ''' generate objs infos '''
            for i in range(num_objs):
                # parse model path
                _full_path, _ = set_model_path(MODEL_PATH)

                # get obj info and xml
                obj_info, obj_elt = self._get_obj_info_elt(_full_path, i, num_parts, rand_pos)
                obj_infos += [obj_info]
                obj_elts += [obj_elt]

            #''' temporary '''
            #obj_infos
            #scale = 0.03
            #geom = elt.Element('geom')
            #geom.set('size', '{:.5f}'.format(scale))
            #body = elt.Element('body')
            #body.set('pos', '{:.5f} {:.5f} {:.5f}'.format(0., 0., 0.2))
            #body.append(geom)
            #obj_elts += [body]

            # add to info
            info['obj_infos'] = obj_infos
            info['obj_elts']  = obj_elts

            #''' generate wall + floor info '''
            #table_info = self._get_table_info()
            #_full_path, _ = set_model_path(TABLE_PATH)
            #table_elt = self._gen_table_elt(_full_path, table_info)
            #info['table_info']  = table_info
            #info['table_elt']   = table_elt
            info['table_elt']   = None

        ''' get model xml using obj_elts '''
        model_xml = self._gen_scene_xml(full_path, info['obj_elts'], info['table_elt'])

        ''' set info '''
        self.info = info

        ''' init environment '''
        with tempfile.NamedTemporaryFile(suffix=".xml", dir=os.path.join(prefix_path, 'shepard_metzler')) as fp:
            # write temporary xml file
            fp.write(model_xml)
            fp.flush()

            # set up modified model path
            modified_model_path=fp.name

            # init environment
            shepard_metzler_hand_camera_env.ShepardMetzlerEnv.__init__(
                self,
                modified_model_path,
                n_substeps=n_substeps,
                initial_qpos=initial_qpos,
                control_method=control_method,
                control_hand_position=control_hand_position,
                control_camera_position=control_camera_position)

        utils.EzPickle.__init__(self)


class ShepardMetzlerHandCameraMultiObjsEnv(ManipulateEnv):
    def __init__(self):
        super().__init__(
            control_hand_position=True,
            control_camera_position=True,
            max_num_objs=3,
            rand_pos=False,
            num_parts=5,
            )

class ShepardMetzlerHandCameraSingleObjEnv(ManipulateEnv):
    def __init__(self):
        super().__init__(
            control_hand_position=True,
            control_camera_position=True,
            max_num_objs=1,
            rand_pos=False,
            num_parts=5,
            )
