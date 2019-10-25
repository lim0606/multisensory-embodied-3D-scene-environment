# basic
from envs.haptix.basic.manipulate      import BasicBoxEnv, BasicEllipsoidEnv, BasicCylinderEnv, BasicRandomEnv
from envs.haptix.basic.manipulate      import BasicBoxPosCtrlEnv, BasicEllipsoidPosCtrlEnv, BasicCylinderPosCtrlEnv, BasicRandomPosCtrlEnv
from envs.haptix.cameractrl.manipulate import CameraCtrlBoxEnv, CameraCtrlEllipsoidEnv, CameraCtrlCylinderEnv, CameraCtrlRandomEnv
from envs.haptix.cameractrl.manipulate import CameraCtrlBoxPosCtrlEnv, CameraCtrlEllipsoidPosCtrlEnv, CameraCtrlCylinderPosCtrlEnv, CameraCtrlRandomPosCtrlEnv

# rooms
from envs.haptix.rooms.hand        import RoomsHandMultiObjsEnv, RoomsHandSingleObjEnv
from envs.haptix.rooms.camera      import RoomsCameraMultiObjsEnv, RoomsCameraSingleObjEnv
from envs.haptix.rooms.hand_camera import RoomsHandCameraMultiObjsEnv, RoomsHandCameraSingleObjEnv

# shepard_metzler
from envs.haptix.shepard_metzler.camera      import ShepardMetzlerCameraMultiObjsEnv, ShepardMetzlerCameraSingleObjEnv, ShepardMetzlerCameraSingleObjP4Env, ShepardMetzlerCameraSingleObjP6Env
from envs.haptix.shepard_metzler.hand        import ShepardMetzlerHandMultiObjsEnv, ShepardMetzlerHandSingleObjEnv, ShepardMetzlerHandSingleObjP4Env, ShepardMetzlerHandSingleObjP6Env
from envs.haptix.shepard_metzler.hand_camera import ShepardMetzlerHandCameraMultiObjsEnv, ShepardMetzlerHandCameraSingleObjEnv

# toy
from envs.haptix.toy.camera      import ToyCameraSingleObjSphereEnv,     ToyCameraSingleObjBoxEnv,     ToyCameraSingleObjCylinderEnv
from envs.haptix.toy.hand        import ToyHandSingleObjSphereEnv,       ToyHandSingleObjBoxEnv,       ToyHandSingleObjCylinderEnv
from envs.haptix.toy.hand_camera import ToyHandCameraSingleObjSphereEnv, ToyHandCameraSingleObjBoxEnv, ToyHandCameraSingleObjCylinderEnv
