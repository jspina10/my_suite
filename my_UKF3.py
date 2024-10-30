import myosuite
import mujoco as mj
import mujoco.viewer as viewer
import gym
import time 
from myosuite.simhive.myo_sim.test_sims import TestSims as loader
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
import scipy.sparse as spa
import pandas as pd
import numpy as np
import skvideo.io
import osqp
import os
from scipy.signal import butter, filtfilt
from IPython.display import HTML
from base64 import b64encode
from copy import deepcopy
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
from _myosuite.envs.myo import myobase
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

### INIT
env = gym.make("my_MyoHandEnvForce-v0", frame_skip=1, normalize_act=False)
tausmooth = 5
env.unwrapped.sim.model.actuator_dynprm[:,2] = tausmooth
model = env.sim.model._model
data = mj.MjData(model) 
# TEST
env_test = gym.make("my_MyoHandEnvForce-v0", frame_skip=1, normalize_act=False)
tausmooth = 5
env_test.unwrapped.sim.model.actuator_dynprm[:,2] = tausmooth
model_test = env_test.sim.model._model
data_test = mj.MjData(model_test) 
options_test = mj.MjvOption()
options_test.flags[:] = 0
options_test.flags[4] = 1 # actuator ON
options_test.geomgroup[1:] = 0
renderer_test = mj.Renderer(model_test)
renderer_test.scene.flags[:] = 0
# REFERENCE
model_ref = env.sim.model._model
model_ref.actuator_dynprm[:,2] = tausmooth
data_ref = mj.MjData(model_ref)
options_ref = mj.MjvOption()
options_ref.flags[:] = 0
options_ref.geomgroup[1:] = 0
renderer_ref = mj.Renderer(model_ref)
renderer_ref.scene.flags[:] = 0
# DATA
kinematics = pd.read_csv(os.path.join(os.path.dirname(__file__), "trajectories/traj_standard.csv")).values
kinetics = pd.read_csv(os.path.join(os.path.dirname(__file__), "trajectories/traj_force.csv")).values
kinematics_predicted = np.zeros((kinematics.shape[0], kinematics.shape[1]))
kinetics_predicted = np.zeros((kinetics.shape[0], kinetics.shape[1]))
all_qpos = np.zeros((kinematics.shape[0], kinematics.shape[1], 2))
all_qpos[:,:,-1] = kinematics
all_qfrc = np.zeros((kinematics.shape[0], kinematics.shape[1]))
all_ctrl = np.zeros((kinematics.shape[0], 1+model.nu))
# CAMERA
camera = mj.MjvCamera()
camera.azimuth = 166.553
camera.distance = 1.178
camera.elevation = -36.793
camera.lookat = np.array([-0.93762553, -0.34088276, 0.85067529])
# FUNTIONS
def fx(x, dt):
    data.qpos[:] = x[:nq]
    data.qvel[:] = x[nq:2*nq]
    forces = x[2*nq:]
    env.apply_forces(model, data, forces)
    mj.mj_step(model, data)
    x_new = np.concatenate((data.qpos, data.qvel, forces))    
    return x_new
def hx(x):
    z_pos = x[:nq]
    z_force = x[2*nq:]
    z = np.concatenate((z_pos, z_force))
    return z    
# UKF 
nq = model.nq
dim_x = 2 * nq + 5
dim_z = nq + 5
points = MerweScaledSigmaPoints(dim_x, alpha=0.1, beta=2., kappa=0)
ukf = UKF(dim_x=dim_x, dim_z=dim_z, fx=None, hx=None, dt=0.002, points=points)
ukf.x = np.zeros(dim_x)
ukf.P *= 0.1
ukf.Q = np.eye(dim_x) * 0.01
R_pos = np.eye(nq) * 0.1  
R_force = np.eye(5) * 0.05 
ukf.R = np.block([
            [R_pos, np.zeros((nq, 5))],
            [np.zeros((5, nq)), R_force]
        ]) 
ukf.fx = fx
ukf.hx = hx

# LOOP
obs = env.reset()
obs = env_test.reset()
frames = []
for idx in tqdm(range(kinematics.shape[0])):
    # Reference 
    data_ref.qpos = kinematics[idx, 1:]
    mj.mj_step1(model_ref, data_ref)
    # Prediction UKF
    kinematics_row = kinematics[idx, 1:] 
    kinetics_row = kinetics[idx, 1:]
    z = np.concatenate((kinematics_row, kinetics_row))
    ukf.predict()
    ukf.update(z)
    x = ukf.x
    kinematics_predicted[idx,:] = np.hstack((data.time, x[:nq]))
    kinetics_predicted[idx,:] = np.hstack((data.time, x[2*nq:]))
    # Inverse Dynamics
    target_qpos = kinematics_predicted[idx, 1:]
    qfrc = env.get_qfrc(model_test, data_test, target_qpos)
    all_qpos[idx,:,0] = np.hstack((data_test.time, data_test.qpos))
    all_qfrc[idx,:] = np.hstack((data_test.time, qfrc))
    # Quadratic Problem
    ctrl = env.get_ctrl(model_test, data_test, target_qpos, qfrc, 100, 5)
    data_test.ctrl = ctrl
    mj.mj_step(model_test, data_test)
    env.step(ctrl)
    env_test.renderer.render_to_window() 
    all_ctrl[idx,:] = np.hstack((data_test.time, ctrl))
    # Rendering
    if not idx % round(0.3/(model_test.opt.timestep*25)):
        renderer_ref.update_scene(data_ref, camera=camera, scene_option=options_ref)
        frame_ref = renderer_ref.render()
        renderer_test.update_scene(data_test, camera=camera, scene_option=options_test)
        frame = renderer_test.render()
        frame_merged = np.append(frame_ref, frame, axis=1)
        frames.append(frame_merged)

error = ((all_qpos[:,1:,0] - all_qpos[:,1:,-1])**2).mean(axis=0)
print(f'error max (rad): {error.max()}')
joint_names = [model.joint(i).name for i in range(model.nq)]
env.plot_qxxx(all_qpos, joint_names, ['Achieved qpos', 'Reference qpos'])
env.plot_qxxx_2d(all_qfrc, joint_names, ['Achieved qfrc'])
muscle_names = [model_test.actuator(i).name for i in range(model_test.nu)]
env.plot_uxxx_2d(all_ctrl, muscle_names, ['Achieved ctrl'])

# SAVE
output_name = os.path.join(os.path.dirname(__file__), "videos/ukf3.mp4")
skvideo.io.vwrite(output_name, np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
output_path = os.path.join(os.path.dirname(__file__), "trajectories/simulation/kinematics_predicted_ukf3.csv")
pd.DataFrame(kinematics_predicted).to_csv(output_path, index=False, header=False)
output_path = os.path.join(os.path.dirname(__file__), "trajectories/simulation/kinetics_predicted_ukf3.csv")
pd.DataFrame(kinetics_predicted).to_csv(output_path, index=False, header=False)
