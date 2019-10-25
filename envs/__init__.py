from gym.envs.registration import registry, register, make, spec

# HAPTIX
# ----------------------------------------
''' toy '''

#register(
#    id='ToyCameraSingleObjSphere-v0',
#    entry_point='envs.haptix:ToyCameraSingleObjSphereEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#register(
#    id='ToyHandSingleObjSphere-v0',
#    entry_point='envs.haptix:ToyHandSingleObjSphereEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#register(
#    id='ToyHandCameraSingleObjSphere-v0',
#    entry_point='envs.haptix:ToyHandCameraSingleObjSphereEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#register(
#    id='ToyCameraSingleObjBox-v0',
#    entry_point='envs.haptix:ToyCameraSingleObjBoxEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#register(
#    id='ToyHandSingleObjBox-v0',
#    entry_point='envs.haptix:ToyHandSingleObjBoxEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#register(
#    id='ToyHandCameraSingleObjBox-v0',
#    entry_point='envs.haptix:ToyHandCameraSingleObjBoxEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#register(
#    id='ToyCameraSingleObjCylinder-v0',
#    entry_point='envs.haptix:ToyCameraSingleObjCylinderEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#register(
#    id='ToyHandSingleObjCylinder-v0',
#    entry_point='envs.haptix:ToyHandSingleObjCylinderEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#register(
#    id='ToyHandCameraSingleObjCylinder-v0',
#    entry_point='envs.haptix:ToyHandCameraSingleObjCylinderEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)

#''' shepard_metzler '''
#
#register(
#    id='ShepardMetzlerCameraSingleObj-v0',
#    entry_point='gym.envs.haptix:ShepardMetzlerCameraSingleObjEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#register(
#    id='ShepardMetzlerCameraMultiObjs-v0',
#    entry_point='gym.envs.haptix:ShepardMetzlerCameraMultiObjsEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)


register(
    id='ShepardMetzlerCameraSingleObjP4-v0',
    entry_point='envs.haptix:ShepardMetzlerCameraSingleObjP4Env',
    kwargs={},
    max_episode_steps=100000,
)

register(
    id='ShepardMetzlerHandSingleObjP4-v0',
    entry_point='envs.haptix:ShepardMetzlerHandSingleObjP4Env',
    kwargs={},
    max_episode_steps=100000,
)

register(
    id='ShepardMetzlerCameraSingleObjP6-v0',
    entry_point='envs.haptix:ShepardMetzlerCameraSingleObjP6Env',
    kwargs={},
    max_episode_steps=100000,
)

register(
    id='ShepardMetzlerHandSingleObjP6-v0',
    entry_point='envs.haptix:ShepardMetzlerHandSingleObjP6Env',
    kwargs={},
    max_episode_steps=100000,
)

register(
    id='ShepardMetzlerCameraSingleObjP5-v0',
    entry_point='envs.haptix:ShepardMetzlerCameraSingleObjEnv',
    kwargs={},
    max_episode_steps=100000,
)

register(
    id='ShepardMetzlerHandSingleObjP5-v0',
    entry_point='envs.haptix:ShepardMetzlerHandSingleObjEnv',
    kwargs={},
    max_episode_steps=100000,
)

#register(
#    id='ShepardMetzlerHandMultiObjs-v0',
#    entry_point='gym.envs.haptix:ShepardMetzlerHandMultiObjsEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#register(
#    id='ShepardMetzlerHandCameraSingleObj-v0',
#    entry_point='gym.envs.haptix:ShepardMetzlerHandCameraSingleObjEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#register(
#    id='ShepardMetzlerHandCameraMultiObjs-v0',
#    entry_point='gym.envs.haptix:ShepardMetzlerHandCameraMultiObjsEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#''' rooms '''
#register(
#    id='HaptixRoomsCameraSingleObj-v0',
#    entry_point='gym.envs.haptix:RoomsCameraSingleObjEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#register(
#    id='HaptixRoomsCameraMultiObjs-v0',
#    entry_point='gym.envs.haptix:RoomsCameraMultiObjsEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#register(
#    id='HaptixRoomsHandSingleObj-v0',
#    entry_point='gym.envs.haptix:RoomsHandSingleObjEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#register(
#    id='HaptixRoomsHandMultiObjs-v0',
#    entry_point='gym.envs.haptix:RoomsHandMultiObjsEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#register(
#    id='HaptixRoomsHandCameraSingleObj-v0',
#    entry_point='gym.envs.haptix:RoomsHandCameraSingleObjEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#register(
#    id='HaptixRoomsHandCameraMultiObjs-v0',
#    entry_point='gym.envs.haptix:RoomsHandCameraMultiObjsEnv',
#    kwargs={},
#    max_episode_steps=100000,
#)
#
#''' basic '''
#for objtype in ['Box', 'Ellipsoid', 'Cylinder', 'Random']:
#    for tasktype in ['Basic', 'CameraCtrl']:
#        for ctrlmode in ['', 'PosCtrl']:
#            register(
#                id='Haptix{}Manipulate{}{}-v0'.format(tasktype, objtype, ctrlmode),
#                entry_point='gym.envs.haptix:{}{}{}Env'.format(tasktype, objtype, ctrlmode),
#                kwargs={},
#                max_episode_steps=100,
#            )
#
#            register(
#                id='Haptix{}Manipulate{}{}-v1'.format(tasktype, objtype, ctrlmode),
#                entry_point='gym.envs.haptix:{}{}{}Env'.format(tasktype, objtype, ctrlmode),
#                kwargs={},
#                max_episode_steps=1000,
#            )
