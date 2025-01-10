import myosuite
import mujoco as mj
import gym
import time 
from mujoco.glfw import glfw
from mpl_toolkits.mplot3d import Axes3D
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
from filterpy.kalman import SimplexSigmaPoints
from filterpy.kalman import JulierSigmaPoints
from scipy.optimize import minimize

plt.ion()
fig_2 = plt.figure()
ax_2 = fig_2.add_subplot(111, projection='3d')

### RUNNARE CON PYTHON3 !!!

def show_video(video_path, video_width = 400):
    """
    Display a video within the notebook.
    """
    video_file = open(video_path, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")
def compute_qpos_from_cartesian(desired_cartesian_positions, current_generalized_positions, model, data):
    """
    Compute the generalized positions (qpos) needed to the model to reach the next configuration

    Args: 
        desired_cartesian_positions (np.ndarray): Array of the cartisian position needed,
                                                  shape (N, 3), where N is the number of keypoints detected.
        current_generalized_positions (np.ndarray): Current state of the model (qpos).
        model (mj.MjModel): Model of MuJoCo.
        data (mj.MjData): Data of the MuJoCo model uploaded.

    Returns:
        np.ndarray: The joints positions (qpos) computed.
    """
    nq = model.nq  # number of the joints variables (qpos)

    def cost_function(qpos):
        """
        Cost function that measure the distance between the actual joints positions and the wished ones.
        """
        # Update the model with the actual qpos
        data.qpos[:] = qpos
        data.qvel[:] = 0  # Initial velocities equal to zero 
        mj.mj_forward(model, data)
        
        # Compute the actual cartesian position of the keypoints
        current_cartesian_positions = []
        joint_ids = [2, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22]
        body_ids = [21, 28, 33, 38, 43]
        lista = [4, 8, 12, 16, 20]
        
        for i in joint_ids:
            current_cartesian_positions.append(data.xanchor[i])
        for i, j in enumerate(body_ids):
            current_cartesian_positions.insert(lista[i], data.xpos[j])
        
        current_cartesian_positions = np.array(current_cartesian_positions).flatten()
        
        # Compute the distance between the actual cartesian positions and the wished ones
        error = np.linalg.norm(current_cartesian_positions - desired_cartesian_positions.flatten())
        return error

    # Configuration of the minimizator function
    bounds = [(model.jnt_range[i][0], model.jnt_range[i][1]) for i in range(nq)]
    result = minimize(
        cost_function, 
        current_generalized_positions, 
        bounds=bounds, 
        method='SLSQP', 
        options={'disp': True, 'maxiter': 500}
    )

    if result.success:
        return result.x  # Returns the optimal qpos
    else:
        raise ValueError("It has not been found an adequate solution: " + result.message)
def solve_qp(P, q, lb, ub, x0):
    """
    Solve a quadratic program.
    """
    P = spa.csc_matrix(P)
    A = spa.csc_matrix(spa.eye(q.shape[0]))
    m = osqp.OSQP()
    m.setup(P=P, q=q, A=A, l=lb, u=ub, verbose=False)
    m.warm_start(x=x0)
    res = m.solve()
    return res.x
def get_qfrc(model, data, target_qpos):
    """
    Compute the generalized force needed to reach the target position in the next mujoco step.
    """
    data_copy = deepcopy(data)
    data_copy.qacc = (((target_qpos - data.qpos) / model.opt.timestep) - data.qvel) / model.opt.timestep
    model.opt.disableflags += mj.mjtDisableBit.mjDSBL_CONSTRAINT
    mj.mj_inverse(model, data_copy)
    model.opt.disableflags -= mj.mjtDisableBit.mjDSBL_CONSTRAINT
    return data_copy.qfrc_inverse
def get_ctrl(model, data, target_qpos, qfrc, qfrc_scaler, qvel_scaler):
    """
    Compute the control needed to reach the target position in the next mujoco step.
    qfrc: generalized force resulting from inverse dynamics.
    """
    act = data.act
    ctrl0 = data.ctrl
    ts = model.opt.timestep
    tA = model.actuator_dynprm[:,0] * (0.5 + 1.5 * act)
    tD = model.actuator_dynprm[:,1] / (0.5 + 1.5 * act)
    tausmooth = model.actuator_dynprm[:,2]
    t1 = (tA - tD) * 1.875 / tausmooth
    t2 = (tA + tD) * 0.5
    # ---- gain, bias, and moment computation
    data_copy = deepcopy(data)
    data_copy.qpos = target_qpos
    data_copy.qvel = ((target_qpos - data.qpos) / model.opt.timestep) / qvel_scaler
    mj.mj_step1(model, data_copy) # gain, bias, and moment depend on qpos and qvel
    gain = np.zeros(model.nu)
    bias = np.zeros(model.nu)
    for idx_actuator in range(model.nu):
        length = data_copy.actuator_length[idx_actuator]
        lengthrange = model.actuator_lengthrange[idx_actuator]
        velocity = data_copy.actuator_velocity[idx_actuator]
        acc0 = model.actuator_acc0[idx_actuator]
        prmb = model.actuator_biasprm[idx_actuator,:9]
        prmg = model.actuator_gainprm[idx_actuator,:9]
        bias[idx_actuator] = mj.mju_muscleBias(length, lengthrange, acc0, prmb)
        gain[idx_actuator] = min(-1, mj.mju_muscleGain(length, velocity, lengthrange, acc0, prmg))
    AM = data_copy.actuator_moment.T
    # ---- ctrl computation
    P = 2 * AM.T @ AM
    k = AM @ (gain * act) + AM @ bias - (qfrc / qfrc_scaler)
    q = 2 * k @ AM
    lb = gain * (1 - act) * ts / (t2 + t1 * (1 - act))
    ub = - gain * act * ts / (t2 - t1 * act)
    x0 = (gain * (ctrl0 - act) * ts) / ((ctrl0 - act) * t1 + t2)
    x = solve_qp(P, q, lb, ub, x0)
    ctrl = act + x * t2 / (gain * ts - x * t1)
    return np.clip(ctrl,0,1)

### INIT
env = gym.make("my_MyoHandEnvForce-v0", frame_skip=1, normalize_act=False)
model = env.sim.model._model
data = mj.MjData(model)
tausmooth = 5
### REFERENCE
model_ref = env.sim.model._model
model_ref.actuator_dynprm[:,2] = tausmooth
data_ref = mj.MjData(model_ref)
options_ref = mj.MjvOption()
options_ref.flags[:] = 0
options_ref.geomgroup[1:] = 0
renderer_ref = mj.Renderer(model_ref)
renderer_ref.scene.flags[:] = 0
### TEST
model_test = env.sim.model._model
model_test.actuator_dynprm[:,2] = tausmooth
data_test = mj.MjData(model_test) 
options_test = mj.MjvOption()
options_test.flags[:] = 0
options_test.flags[4] = 1 # actuator ON
options_test.geomgroup[1:] = 0
renderer_test = mj.Renderer(model_test)
renderer_test.scene.flags[:] = 0
### DATA
nq = model_test.nq
nu = model_test.nu
nk = 21
kinematics_tot = pd.read_csv(os.path.join(os.path.dirname(__file__), "trajectories/traj_keypoints_ref.csv")).values
kinematics = kinematics_tot[:, :]
time_steps = kinematics[:, 0]
# CAMERA
camera = mj.MjvCamera()
camera.azimuth = 166.553
camera.distance = 1.178
camera.elevation = -36.793
camera.lookat = np.array([-0.93762553, -0.34088276, 0.85067529])

def hx(x):
    """
    Observation function:
    Maps the augmented state into the observating measurements.
    Includes the cartesian positions of the keypoints detected and the measured forces at the fingerprints.
    Args:
    x: Augmented state vector [includes the state vector (positions, velocities)].
    Returns:
    z: Measurements vector [includes the keypoints cartesian position].
    """
    # Extract the joints position & velocity
    data.qpos[:] = x[:nq]
    data.qvel[:] = x[nq:]
    # Update model with the current state
    mj.mj_forward(model, data)
    # Compute keypoints positions
    joint_ids = [2, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22]
    body_ids = [21, 28, 33, 38, 43]
    lista = [4, 8, 12, 16, 20]
    keypoints = []
    for i in joint_ids:
        keypoints.append(data.xanchor[i])
    for i, j in enumerate(body_ids):
        keypoints.insert(lista[i], data.xpos[j])
    keypoints_flat = np.array(keypoints).flatten()
    z = keypoints_flat
    return z

# LOOP
obs = env.reset()
frames = []
last_result = None
calculated_trajectories = []
for idx in tqdm(range(kinematics.shape[0])):
    ## Inverse Kinematics
    if idx == 0:
        current_generalized_positions = np.zeros(nq)
    else:
        current_generalized_positions = last_result
    desired_cartesian_positions = kinematics[idx,1:]
    result = compute_qpos_from_cartesian(
        desired_cartesian_positions, 
        current_generalized_positions, 
        model_ref, 
        data_ref)
    last_result = result
    # Update the model with the computed state
    data_ref.qpos[:] = result
    mj.mj_forward(model_ref, data_ref)

    ## Inverse Dynamics
    target_qpos = result
    qfrc = get_qfrc(model_test, data_test, target_qpos)

    ## Quadratic Problem
    ctrl = get_ctrl(model_test, data_test, target_qpos, qfrc, 100, 5)
    data_test.ctrl = ctrl
    mj.mj_step(model_test, data_test)
    
    ## Rendering
    # Use the function h(x) to compute the position of the keypoints
    x_state = np.concatenate([data_test.qpos, data_test.qvel])
    calculated_keypoints = hx(x_state)
    # Save the computed cartesian position of the keypoints
    calculated_trajectories.append(calculated_keypoints)
    if not idx % round(0.3/(model_test.opt.timestep*25)):
        renderer_test.update_scene(data_test, camera=camera, scene_option=options_test)
        frame = renderer_test.render()
        frames.append(frame)

# Error
errors = []
# Converts the computed cartesian keypoints positions into a NumPy array
calculated_trajectories = np.array(calculated_trajectories)
for idx in range(kinematics.shape[0]):
    original_positions = kinematics[idx, 1:]  # Posizioni originali dei punti chiave
    calculated_positions = calculated_trajectories[idx]  # Posizioni calcolate dei punti chiave
    # Calcola la distanza euclidea tra la traiettoria originale e quella calcolata
    error = np.linalg.norm(original_positions - calculated_positions)
    errors.append(error)
# Compute the mean
mean_error = np.mean(errors)
# Plot the error in time
plt.figure()
plt.plot(time_steps, errors, label='Errore Traiettoria')
plt.xlabel('Step temporale')
plt.ylabel('Errore (Distanza euclidea)')
plt.title('Errore tra la traiettoria originale e quella calcolata')
plt.legend()
plt.show()

print(f"Errore medio tra la traiettoria originale e quella calcolata: {mean_error}")

output_name = os.path.join(os.path.dirname(__file__), "videos/my_sim.mp4")
skvideo.io.vwrite(output_name, np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
