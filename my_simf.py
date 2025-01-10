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

# plt.ion()
# fig_2 = plt.figure()
# ax_2 = fig_2.add_subplot(111, projection='3d')

### RUNNARE CON PYTHON3 !!!

def show_video(video_path, video_width = 400):
    """
    Display a video within the notebook.
    """
    video_file = open(video_path, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")
def plot_qxxx(qxxx, joint_names, labels):
    """
    Plot generalized variables to be compared.
    qxxx[:,0,-1] = time axis
    qxxx[:,1:,n] = n-th sequence
    qxxx[:,1:,-1] = reference sequence
    """
    fig, axs = plt.subplots(4, 6, figsize=(12, 8))
    axs = axs.flatten()
    line_objects = []
    linestyle = ['-'] * qxxx.shape[2]
    linestyle[-1] = '--'
    for j in range(1, len(joint_names)+1):
        ax = axs[j-1]
        for i in range(qxxx.shape[2]):
            line, = ax.plot(qxxx[:, 0, -1], qxxx[:, j, i], linestyle[i])
            if j == 1: # add only one set of lines to the legend
                line_objects.append(line)
        ax.set_xlim([qxxx[:, 0].min(), qxxx[:, 0].max()])
        ax.set_ylim([qxxx[:, 1:, :].min(), qxxx[:, 1:, :].max()])
        ax.set_title(joint_names[j-1])
    legend_ax = axs[len(joint_names)] # create legend in the 24th subplot area
    legend_ax.axis('off')
    legend_ax.legend(line_objects, labels, loc='center')
    plt.tight_layout()
    # plt.show()
    plt.savefig('graphs/UKF2_qpos_simf.png')  # Save the plot to a file
    plt.close()  # Close the figure to free memory
def plot_qxxx222(qxxx, joint_names, labels):
    """
    Plot generalized variables to be compared.
    qxxx[:,0,-1] = time axis
    qxxx[:,1:,n] = n-th sequence
    qxxx[:,1:,-1] = reference sequence
    """
    fig, axs = plt.subplots(4, 6, figsize=(12, 8))
    axs = axs.flatten()
    line_objects = []
    linestyle = ['-'] * qxxx.shape[2]
    linestyle[-1] = '--'
    for j in range(1, len(joint_names)+1):
        ax = axs[j-1]
        for i in range(qxxx.shape[2]):
            line, = ax.plot(qxxx[:, 0, -1], qxxx[:, j, i], linestyle[i])
            if j == 1: # add only one set of lines to the legend
                line_objects.append(line)
        ax.set_xlim([qxxx[:, 0].min(), qxxx[:, 0].max()])
        ax.set_ylim([qxxx[:, 1:, :].min(), qxxx[:, 1:, :].max()])
        ax.set_title(joint_names[j-1])
    legend_ax = axs[len(joint_names)] # create legend in the 24th subplot area
    legend_ax.axis('off')
    legend_ax.legend(line_objects, labels, loc='center')
    plt.tight_layout()
    # plt.show()
    plt.savefig('graphs/UKF2_qpos_simf2.png')  # Save the plot to a file
    plt.close()  # Close the figure to free memory
def plot_qxxx_2(qxxx, joint_names, labels):
    """
    Plot generalized variables to be compared.
    qxxx[:,0,-1] = time axis
    qxxx[:,1:,n] = n-th sequence
    qxxx[:,1:,-1] = reference sequence
    """
    fig, axs = plt.subplots(4, 6, figsize=(12, 8))
    axs = axs.flatten()
    line_objects = []
    linestyle = ['-'] * qxxx.shape[2]
    linestyle[-1] = '--'
    for j in range(1, len(joint_names)+1):
        ax = axs[j-1]
        for i in range(qxxx.shape[2]):
            line, = ax.plot(qxxx[:, 0, -1], qxxx[:, j, i], linestyle[i])
            if j == 1: # add only one set of lines to the legend
                line_objects.append(line)
        ax.set_xlim([qxxx[:, 0].min(), qxxx[:, 0].max()])
        ax.set_ylim([qxxx[:, 1:, :].min(), qxxx[:, 1:, :].max()])
        ax.set_title(joint_names[j-1])
    legend_ax = axs[len(joint_names)] # create legend in the 24th subplot area
    legend_ax.axis('off')
    legend_ax.legend(line_objects, labels, loc='center')
    plt.tight_layout()
    # plt.show()
    plt.savefig('graphs/UKF2_qfrc_simf.png')  # Save the plot to a file
    plt.close()  # Close the figure to free memory
def plot_fxxx(fxxx, fingertips_names, labels):
    """
    Plot generalized variables to be compared.
    fxxx[:,0,-1] = time axis
    fxxx[:,1:,n] = n-th sequence
    fxxx[:,1:,-1] = reference sequence
    """
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.flatten()
    line_objects = []
    linestyle = ['-'] * fxxx.shape[2]
    linestyle[-1] = '--'
    for j in range(1, len(fingertips_names)+1):
        ax = axs[j-1]
        for i in range(fxxx.shape[2]):
            line, = ax.plot(fxxx[:, 0, -1], fxxx[:, j, i], linestyle[i])
            if j == 1: # add only one set of lines to the legend
                line_objects.append(line)
        ax.set_xlim([fxxx[:, 0].min(), fxxx[:, 0].max()])
        ax.set_ylim([fxxx[:, 1:, :].min(), fxxx[:, 1:, :].max()])
        ax.set_title(fingertips_names[j-1])
    legend_ax = axs[len(fingertips_names)] # create legend in the 24th subplot area
    legend_ax.axis('off')
    legend_ax.legend(line_objects, labels, loc='center')
    plt.tight_layout()
    # plt.show()
    plt.savefig('graphs/UKF2_frcs_simf.png')  # Save the plot to a file
    plt.close()  # Close the figure to free memory
def plot_qxxx_2d(qxxx, joint_names, labels):
    """
    Plot generalized variables to be compared.
    qxxx[:,0] = time axis
    qxxx[:,1:] = n-th sequence
    """
    fig, axs = plt.subplots(4, 6, figsize=(12, 8))
    axs = axs.flatten()
    line_objects = []
    for j in range(1, len(joint_names)+1):
        ax = axs[j-1]
        line, = ax.plot(qxxx[:, 0], qxxx[:, j])
        if j == 1: # add only one set of lines to the legend
            line_objects.append(line)
        ax.set_xlim([qxxx[:, 0].min(), qxxx[:, 0].max()])
        ax.set_ylim([qxxx[:, 1:].min(), qxxx[:, 1:].max()])
        ax.set_title(joint_names[j-1])
    legend_ax = axs[len(joint_names)] # create legend in the 24th subplot area
    legend_ax.axis('off')
    legend_ax.legend(line_objects, labels, loc='center')
    plt.tight_layout()
    # plt.show()
    plt.savefig('graphs/UKF2_qfrc_simf_noref.png')  # Save the plot to a file
    plt.close()  # Close the figure to free memory
def plot_uxxx_2(uxxx, muscle_names, labels):
    """
    Plot actuator variables to be compared.
    uxxx[:,0] = time axis
    uxxx[:,1:] = n-th sequence
    """
    fig, axs = plt.subplots(5, 8, figsize=(12, 8))
    axs = axs.flatten()
    line_objects = []
    linestyle = ['-'] * uxxx.shape[2]
    linestyle[-1] = '--'
    for j in range(1, len(muscle_names)+1):
        ax = axs[j-1]
        for i in range(uxxx.shape[2]):
            line, = ax.plot(uxxx[:, 0, -1], uxxx[:, j, i], linestyle[i])
            if j == 1: # add only one set of lines to the legend
                line_objects.append(line)
        ax.set_xlim([uxxx[:, 0].min(), uxxx[:, 0].max()])
        ax.set_ylim([uxxx[:, 1:].min(), uxxx[:, 1:].max()])
        ax.set_title(muscle_names[j-1])
    legend_ax = axs[len(muscle_names)] # create legend in the 40th subplot area
    legend_ax.axis('off')
    legend_ax.legend(line_objects, labels, loc='center')
    plt.tight_layout()
    # plt.show()
    plt.savefig('graphs/UKF2_ctrl_simf.png')  # Save the plot to a file
    plt.close()  # Close the figure to free memory
def plot_uxxx_2d(uxxx, muscle_names, labels):
    """
    Plot actuator variables to be compared.
    uxxx[:,0] = time axis
    uxxx[:,1:] = n-th sequence
    """
    fig, axs = plt.subplots(5, 8, figsize=(12, 8))
    axs = axs.flatten()
    line_objects = []
    for j in range(1, len(muscle_names)+1):
        ax = axs[j-1]
        line, = ax.plot(uxxx[:, 0], uxxx[:, j])
        if j == 1: # add only one set of lines to the legend
            line_objects.append(line)
        ax.set_xlim([uxxx[:, 0].min(), uxxx[:, 0].max()])
        ax.set_ylim([uxxx[:, 1:].min(), uxxx[:, 1:].max()])
        ax.set_title(muscle_names[j-1])
    legend_ax = axs[len(muscle_names)] # create legend in the 40th subplot area
    legend_ax.axis('off')
    legend_ax.legend(line_objects, labels, loc='center')
    plt.tight_layout()
    # plt.show()
    plt.savefig('graphs/UKF2_ctrl_simf_noref.png')  # Save the plot to a file
    plt.close()  # Close the figure to free memory
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
def apply_forces(model, data, forces):
    """
    Apply external forces to the distal phalanges.
    Args:
        model: Your model (not used in this function).
        data: Your data (not used in this function).
        forces_matrix: A matrix of shape (4752, 6) where each row corresponds to a timestep.
                            Columns 1-5 represent the scalar force applied to each finger along the z-axis.
        timestep: The current timestep.
    """
    # Body IDs for distal phalanges
    body_ids = [21, 28, 33, 38, 43]  # Thumb and other finger IDs

    for i, body_id in enumerate(body_ids):
        # Extract the scalar force for the current finger
        scalar_force = forces[i]*10
        # print(scalar_force)
        # Get the rotation matrix from the global frame to the local frame
        body_xmat = data.xmat[body_id].reshape(3, 3)
        # Construct the force vector (assuming x and y components are zero)
        external_force_local = np.array([0, 0, scalar_force])
        global_force =  body_xmat @ external_force_local
        # Apply the local force to the body
        data.xfrc_applied[body_id, :3] = global_force 

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
### TEST0
model_test0 = env.sim.model._model
model_test0.actuator_dynprm[:,2] = tausmooth
data_test0 = mj.MjData(model_test0)
options_test0 = mj.MjvOption()
options_test0.flags[:] = 0
options_test0.flags[4] = 1 # actuator ON
options_test0.geomgroup[1:] = 0
renderer_test0 = mj.Renderer(model_test0)
renderer_test0.scene.flags[:] = 0
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
observator = ['1nf','1wf','2nf','2wf']
type = observator[0]
nq = model_test.nq
nu = model_test.nu
if type in ['1wf', '2nf', '2wf']:
    nf = 5
    kinetics = pd.read_csv(os.path.join(os.path.dirname(__file__), "trajectories/traj_force.csv")).values
    all_frcs = np.zeros((kinetics.shape[0], 1+nf, 2))
    all_frcs[:,:,-1] = kinetics
if type == '1nf':
    nf = 0
nk = 21
kinematics_tot = pd.read_csv(os.path.join(os.path.dirname(__file__), "trajectories/traj_keypoints_ref.csv")).values
kinematics = kinematics_tot[:, :]
time_steps = kinematics[:, 0]
all_qpos = np.zeros((kinematics.shape[0], 1+nq, 2))
all_qpos_2 = np.zeros((kinematics.shape[0], 1+nq, 2))
all_qfrc = np.zeros((kinematics.shape[0], 1+nq, 2))
all_ctrl = np.zeros((kinematics.shape[0], 1+nu, 2))
# CAMERA
camera = mj.MjvCamera()
camera.azimuth = 166.553
camera.distance = 1.178
camera.elevation = -36.793
camera.lookat = np.array([-0.93762553, -0.34088276, 0.85067529])
# FUNTIONS
def fx(x, dt):
    """
    Transition Function:
    It applies the forces to the fingerprints and computes system dynamics.

    Args:
    x: Augmented state vector [includes the state vector (positions, velocities) and the forces predicted].
    dt: Time step.

    Returns:
    x_new: New augmented state vector after prediction.
    """
    data.qpos[:] = x[:nq]
    data.qvel[:] = x[nq:2*nq]
    forces = x[2*nq:]
    if type == '2wf':
        apply_forces(model, data, forces)
    mj.mj_step(model, data)
    if type in ['2nf', '2wf']:
        x_new = np.concatenate((data.qpos, data.qvel, forces))
    else:
        x_new = np.concatenate((data.qpos, data.qvel))
    return x_new
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
    data.qvel[:] = x[nq:2*nq]
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
    if type in ['2nf', '2wf']:
        z_force = x[2*nq:]
        z = np.concatenate((keypoints_flat, z_force))
    else:
        z = keypoints_flat
    # actualizar_grafica(np.array(keypoints))
    return z

# UKF 
if type in ['1nf', '1wf']:
    dim_x = 2 * nq
    dim_z = 3 * nk
if type in ['2nf', '2wf']:
    dim_x = 2 * nq + nf
    dim_z = 3 * nk + nf
alpha = 1
beta = 2
kappa = 3-dim_x
points = MerweScaledSigmaPoints(dim_x, alpha=alpha, beta=beta, kappa=kappa)
# points = JulierSigmaPoints(dim_x, kappa=0)
# points = SimplexSigmaPoints(dim_x, alpha=1)
ukf = UKF(dim_x=dim_x, dim_z=dim_z, fx=fx, hx=hx, dt=0.002, points=points)
ukf.x = np.zeros(dim_x)
ukf.P *= 0.1
Q_pos = np.eye(dim_x-nf) * 0.01  
Q_force = np.eye(nf) * 0.005 
ukf.Q = np.block([
            [Q_pos, np.zeros((dim_x-nf, nf))],
            [np.zeros((nf, dim_x-nf)), Q_force]
        ])
R_pos = np.eye(dim_z-nf) * 0.001  
R_force = np.eye(nf) * 0.05 
ukf.R = np.block([
            [R_pos, np.zeros((dim_z-nf, nf))],
            [np.zeros((nf, dim_z-nf)), R_force]
        ])

# LOOP
obs = env.reset()
frames_1 = []
frames_2 = []
last_result = None
estimated_traj = []
IK_traj = []
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
    data_ref.qpos[:] = result
    mj.mj_forward(model_ref, data_ref)
    all_qpos[idx,:,-1] = np.hstack((kinematics[idx, 0], result))
    all_qpos_2[idx,:,-1] = np.hstack((kinematics[idx, 0], result))
    ## Inverse Dynamics
    target_qpos_test0 = result
    if type in ['1wf', '2wf']:
        apply_forces(model_test0, data_test0, kinetics[idx,1:])
    qfrc_test0 = get_qfrc(model_test0, data_test0, target_qpos_test0)
    all_qfrc[idx,:,-1] = np.hstack((kinematics[idx, 0], qfrc_test0))
    ## Quadratic Problem
    ctrl_test0 = get_ctrl(model_test0, data_test0, target_qpos_test0, qfrc_test0, 100, 5)
    all_ctrl[idx,:,-1] = np.hstack((kinematics[idx, 0], ctrl_test0))
    data_test0.ctrl = ctrl_test0
    mj.mj_step(model_test0, data_test0)
    all_qpos_2[idx,:,0] = np.hstack((kinematics[idx, 0], data_test0.qpos))

    # Estimation of UKF
    if idx == 0:
        data_test.qpos[:] = result
        # data_test.qpos[:] = [-1.57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    if type == '1nf':
        z = kinematics[idx, 1:]
    elif type == '1wf':
        z = kinematics[idx, 1:]
        apply_forces(model_test, data_test, kinetics[idx,1:])
    elif type in ['2nf', '2wf']:
        kinematics_row = kinematics[idx, 1:] 
        kinetics_row = kinetics[idx, 1:]
        z = np.concatenate((kinematics_row, kinetics_row))
    ukf.predict()
    ukf.update(z)
    x = ukf.x
    if type == '2wf':
        forces = x[2*nq:]
        max_value = 100
        for ii in range(len(forces)):
            if forces[ii] < 0:
                forces[ii] = 0
            elif forces[ii] > max_value:
                forces[ii] = max_value
        x[2*nq:] = forces
        all_frcs[idx,:,0] = np.hstack((kinetics[idx,0], x[2*nq:]))
    # Inverse Dynamics
    target_qpos_test = x[:nq]
    qfrc_test = get_qfrc(model_test, data_test, target_qpos_test)
    # Quadratic Problem
    ctrl_test = get_ctrl(model_test, data_test, target_qpos_test, qfrc_test, 100, 5)
    data_test.ctrl = ctrl_test
    mj.mj_step(model_test, data_test)
    all_qpos[idx,:,0] = np.hstack((kinematics[idx, 0], data_test.qpos))
    all_qfrc[idx,:,0] = np.hstack((kinematics[idx, 0], qfrc_test))
    all_ctrl[idx,:,0] = np.hstack((kinematics[idx, 0], ctrl_test))

    # Use the function h(x) to compute the position of the keypoints
    x_state = np.concatenate([data_test.qpos, data_test.qvel])
    x_state0 = np.concatenate([data_test0.qpos, data_test0.qvel])
    estimated_keypoints = hx(x_state)
    IK_keypoints = hx(x_state0)
    # Save the computed cartesian position of the keypoints
    estimated_traj.append(estimated_keypoints)
    IK_traj.append(IK_keypoints)

    ## Rendering
    if not idx % round(0.3/(model_test.opt.timestep*25)):
        renderer_ref.update_scene(data_ref, camera=camera, scene_option=options_ref)
        frame_ref = renderer_ref.render()
        renderer_test0.update_scene(data_test0, camera=camera, scene_option=options_test0)
        frame_test0 = renderer_test0.render()
        renderer_test.update_scene(data_test, camera=camera, scene_option=options_test)
        frame_test = renderer_test.render()
        frame_merged_1 = np.append(frame_ref, frame_test0, axis=1)
        frame_merged_2 = np.append(frame_ref, frame_test, axis=1)
        frames_1.append(frame_merged_1)
        frames_2.append(frame_merged_2)

# Error
print('\n')
print(f"Observator {type}")
print(f"UKF parameters for σ-points: α={alpha}, β={beta}, κ={kappa}")
print('\n')
errors = []
errors0 = []
# Converts the computed cartesian keypoints positions into a NumPy array
estimated_traj = np.array(estimated_traj)
IK_traj = np.array(IK_traj)
for idx in range(kinematics.shape[0]):
    original_positions = kinematics[idx, 1:]  # Posizioni originali dei punti chiave
    estimated_positions = estimated_traj[idx]  # Posizioni calcolate dei punti chiave
    IK_positions = IK_traj[idx]
    # Calcola la distanza euclidea tra la traiettoria originale e quella calcolata
    error = np.linalg.norm(original_positions - estimated_positions)
    errors.append(error)
    error0 = np.linalg.norm(original_positions - IK_positions)
    errors0.append(error0)
mean_error = np.mean(errors)
max_error = np.max(errors)
mean_error0 = np.mean(errors0)
max_error0 = np.max(errors0)
print(f"cartesian error max [Original - Estimated (UKF+ID+QP)]: {max_error}")
print(f"cartesian error med [Original - Estimated (UKF+ID+QP)]: {mean_error}")
print('\n')
print(f"cartesian error max [Original - Computed (IK+ID+QP)]: {max_error0}")
print(f"cartesian error med [Original - Computed (IK+ID+QP)]: {mean_error0}")
print('\n')
error_rad = np.sqrt(((all_qpos[:,1:,0] - all_qpos[:,1:,-1])**2)).mean(axis=0)
error_deg = (180*error_rad)/np.pi
print(f'error max [Reference IK qpos - Estimated qpos] (rad): {error_rad.max()}')
print(f'error max [Reference IK qpos - Estimated qpos] (deg): {error_deg.max()}')
print('\n')
error0_rad = np.sqrt(((all_qpos_2[:,1:,0] - all_qpos_2[:,1:,-1])**2)).mean(axis=0)
error0_deg = (180*error0_rad)/np.pi
print(f'error max [Reference IK qpos - IK+ID+QP qpos] (rad): {error0_rad.max()}')
print(f'error max [Reference IK qpos - IK+ID+QP qpos] (deg): {error0_deg.max()}')
print('\n')
joint_names = [model_test.joint(i).name for i in range(model_test.nq)]
plot_qxxx(all_qpos, joint_names, ['qpos: UKF', 'qpos: IK'])
plot_qxxx222(all_qpos_2, joint_names, ['qpos: IK+ID+QP', 'qpos: IK'])
plot_qxxx_2(all_qfrc, joint_names, ['qfrc: UKF+ID', 'qfrc: IK+ID'])
plot_qxxx_2d(all_qfrc[:,:,0], joint_names, ['qfrc: UKF+ID'])
muscle_names = [model_test.actuator(i).name for i in range(model_test.nu)]
plot_uxxx_2(all_ctrl, muscle_names, ['ctrl: UKF+ID+QP', 'ctrl: IK+ID+QP'])
plot_uxxx_2d(all_ctrl[:,:,0], muscle_names, ['ctrl: UKF+ID+QP'])
if type == '2wf':
    fingertips_names = ['Thumb Fingertip', 'Index Fingertip', 'Middle Fingertip', 'Ring Fingertip', 'Little Fingertip']
    plot_fxxx(all_frcs, fingertips_names, ['Predicted force', 'Reference force'])

# Save
output_name = os.path.join(os.path.dirname(__file__), "videos/my_simf_notpredicted.mp4")
skvideo.io.vwrite(output_name, np.asarray(frames_1),outputdict={"-pix_fmt": "yuv420p"})
output_name = os.path.join(os.path.dirname(__file__), "videos/my_simf_predicted.mp4")
skvideo.io.vwrite(output_name, np.asarray(frames_2),outputdict={"-pix_fmt": "yuv420p"})
