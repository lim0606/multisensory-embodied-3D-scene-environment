import os
import copy
import numpy as np
import glfw

import gym
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

class HaptixEnv(gym.Env):
    """Superclass for all MuJoCo HAPTIX environments.
    """

    def __init__(self, model_path, initial_qpos, n_substeps):
        # set model path
        #self.model_path = model_path
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        # load model
        self.model = mujoco_py.load_model_from_path(fullpath)

        # init mujoco-py sim and viewer
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)
        self.data = self.sim.data
        self.viewer = None

        # init metadata
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        # set seed
        self.seed()

        # get initial pos/vel
        if len(initial_qpos) == 0:  # initial_qpos is None:
            self.init_qpos = self.sim.data.qpos.ravel().copy()
        else:
            self.init_qpos = initial_qpos
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        # get initial mocap pos
        self.init_mpos = self.sim.data.mocap_pos.copy()

        # init environment
        self._env_setup(initial_qpos=self.init_qpos,
                        initial_qvel=self.init_qvel,
                        initial_mpos=self.init_mpos)

        ## get initial state
        #self.initial_state = copy.deepcopy(self.sim.get_state())

        # set action space
        if not hasattr(self, 'ctrlrange'):
            self._set_ctrlrange()
        low, high = self._get_ctrlrange()
        self.action_space = spaces.Box(low=low, high=high)

        # set observation space
        #observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        #assert not done
        observation = self._get_obs()  # get observation
        self.obs_dim = observation.size
        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        # set viewer setting
        self.with_fixed_window_size = True

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.forward()
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {}
        info = {**info, **self.info}
        #info = {
        #    'is_success': self._is_success(obs['achieved_goal'], self.goal),
        #}
        #reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        reward = 0. #None
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            #did_reset_sim = self._reset_sim()
            did_reset_sim = self.reset_sim()
        #self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

    def destroy_window(self):
        #if hasattr(self, 'viewer') and self.viewer is not None:
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            del self.viewer

    def close(self):
        self.destroy_window()
        #if self.viewer is not None:
        #    # self.viewer.finish()
        #    self.viewer = None
        #    self._viewers = {}

    def render(self, mode='human'):
        self._render_callback()
        if mode == 'rgb_array':
            # get rendered image
            if self.with_fixed_window_size:
                # set up to hide overlay
                self._get_viewer()._hide_overlay = True

                # render viewer
                self._get_viewer().render()

                # get resolution
                resolution = glfw.get_framebuffer_size(
                        self._get_viewer().sim._render_context_window.window)
                width, height = resolution # 1920, 1200

                # read rendered image (with overlay)
                data = self._get_viewer().read_pixels(width, height, depth=False)

                # original image is upside-down, so flip it
                return data[::-1, :, :]

            else:
                # render viewer
                self._get_viewer().render()

                # read rendered image (without overlay)
                data = self._get_viewer()._read_pixels_as_in_window()

                # original image is already flipped (no need to flip again)
                return data

        elif mode == 'human':
            self._get_viewer().render()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self._viewer_setup()
        return self.viewer

    # Extension methods
    # ----------------------------

    #def _reset_sim(self):
    def reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        #self.sim.set_state(self.initial_state)
        #self.sim.forward()
        #self._env_setup(initial_qpos=self.init_qpos,
        #                initial_qvel=self.init_qvel,
        #                initial_mpos=self.init_mpos)
        self._reset_sim()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_ctrlrange(self):
        """Get control ranges for actions.
        """
        raise NotImplementedError()

    #def _is_success(self, achieved_goal, desired_goal):
    #    """Indicates whether or not the achieved goal successfully achieved the desired goal.
    #    """
    #    raise NotImplementedError()

    #def _sample_goal(self):
    #    """Samples a new goal and returns it.
    #    """
    #    raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
