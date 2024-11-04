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

# Inicialización de la figura global
plt.ion()  # Activar el modo interactivo de Matplotlib para actualizar la gráfica.
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
    plt.savefig('graphs/UKF2_qpos_draft_2.png')  # Save the plot to a file
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
    plt.savefig('graphs/UKF2_frcs_draft_2.png')  # Save the plot to a file
    plt.close()  # Close the figure to free memory
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
    plt.savefig('graphs/UKF2_qfrc_draft_2.png')  # Save the plot to a file
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
    plt.savefig('graphs/UKF2_ctrl_draft_2.png')  # Save the plot to a file
    plt.close()  # Close the figure to free memory
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
def actualizar_grafica(arr):
    # Clear the actual graph to avoid superpositions.
    ax_2.clear()
    # Be sure the array is numpy.ndarray.
    if not isinstance(arr, np.ndarray):
        print("The parameter must be numpy.ndarray")
        return
    # Check that the array has 3 coloumns [x, y, z].
    if arr.shape[1] != 3:
        print("The array must have exactly 3 coloumns.")
        return
    # Extract coordinates.
    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]
    # Save the first point coordinates to compute the relative position.
    x_rel = x - x[0]
    y_rel = y - y[0]
    z_rel = z - z[0]
    # Sketch the points in the 3D space.
    ax_2.scatter(x_rel, y_rel, z_rel, color='blue', marker='o', s=20)  # 's' define the size of the p
    # Add the indeces to the respective points.
    for i in range(len(x_rel)):
        ax_2.text(x_rel[i], y_rel[i], z_rel[i], f'{i}', color='red', fontsize=10)
    # Configurate the plot.
    ax_2.set_title('3D plot of the keypoints')
    ax_2.set_xlabel('Axis X')
    ax_2.set_ylabel('Axis Y')
    ax_2.set_zlabel('Axis Z')
    # Configurate the limits of the axes to better visualize.
    ax_2.set_xlim([x_rel.min() - 0.1, x_rel.max() + 0.1])
    ax_2.set_ylim([y_rel.min() - 0.1, y_rel.max() + 0.1])
    ax_2.set_zlim([z_rel.min() - 0.1, z_rel.max() + 0.1])
    # Update the visualization.
    plt.draw()
    plt.pause(0.0001)  # Break to permit the plot update.

### INIT
env = gym.make("my_MyoHandEnvForce-v0", frame_skip=1, normalize_act=False)
# model = env.sim.model._model
# data = mj.MjData(model) 
tausmooth = 5
# TEST
model_test = env.sim.model._model
model_test.actuator_dynprm[:,2] = tausmooth
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
kinematics_qpos = pd.read_csv(os.path.join(os.path.dirname(__file__), "trajectories/traj_standard.csv")).values
# kinematics_keypoints = np.zeros((kinematics.shape[0], 1+3*21))
# kinematics_keypoints[:,0] = kinematics[:,0]
kinematics = pd.read_csv(os.path.join(os.path.dirname(__file__), "trajectories/traj_keypoints.csv")).values
kinetics = pd.read_csv(os.path.join(os.path.dirname(__file__), "trajectories/traj_force.csv")).values
kinematics_predicted = np.zeros((kinematics.shape[0], kinematics.shape[1]))
kinetics_predicted = np.zeros((kinetics.shape[0], kinetics.shape[1]))
real_time_simulation = np.zeros((kinematics.shape[0],1))
all_qpos = np.zeros((kinematics.shape[0], 1+model_test.nq, 2))
all_qpos[:,:,-1] = kinematics_qpos[1:,:]
all_qfrc = np.zeros((kinematics.shape[0], 1+model_test.nq))
all_ctrl = np.zeros((kinematics.shape[0], 1+model_test.nu))
all_frcs =  np.zeros((kinetics.shape[0], kinetics.shape[1], 2))
all_frcs[:,:,-1] = kinetics
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
    data_test.qpos[:] = x[:nq]
    data_test.qvel[:] = x[nq:2*nq]
    forces = x[2*nq:]
    apply_forces(model_test, data_test, forces)
    mj.mj_step(model_test, data_test)
    x_new = np.concatenate((data_test.qpos, data_test.qvel, forces))    
    return x_new
# def hx(x):
#     z_pos = x[:nq]
#     z_force = x[2*nq:]
#     z = np.concatenate((z_pos, z_force))
#     return z    
def hx(x):
    """
    Observation function:
    Maps the augmented state into the observating measurements.
    Includes the cartesian positions of the keypoints detected and the measured forces at the fingerprints.
    
    Args:
    x: Augmented state vector [includes the state vector (positions, velocities) and the forces predicted].

    Returns:
    z: Measurements vector [includes the keypoints cartesian position and the measured forces].
    """
    # Extract the joints position
    qpos = x[:nq]
    # Update model with the current state
    data_test.qpos[:] = qpos
    mj.mj_forward(model_test, data_test)
    # Compute keypoints positions
    joint_ids = [2, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22]
    body_ids = [21, 28, 33, 38, 43]
    lista = [4, 8, 12, 16, 20]
    pos_joints = data_test.xanchor
    vector_total = []
    for i in joint_ids:
        vector_total.append(pos_joints[i])
    for i, j in enumerate(body_ids):
        vector_total.insert(lista[i], data_test.xpos[j])
    keypoints_flat = np.array(vector_total).flatten()
    # Extract forces at fingerprints
    z_force = x[2 * nq:]  # Forze predette ai polpastrelli
    # Combine keypoints positions and forces
    z = np.concatenate((keypoints_flat, z_force))
    # print(len(z))
    # print(z)
    return z
# UKF 
nq = model_test.nq
nf = 5
nk = 21
dim_x = 2 * nq + nf
# dim_z = nq + nf
dim_z = 3 * nk + nf
points = MerweScaledSigmaPoints(dim_x, alpha=0.1, beta=2., kappa=0)
ukf = UKF(dim_x=dim_x, dim_z=dim_z, fx=None, hx=None, dt=0.002, points=points)
ukf.x = np.zeros(dim_x)
ukf.P *= 0.1
ukf.Q = np.eye(dim_x) * 0.01
# R_pos = np.eye(nq) * 0.1  
R_pos = np.eye(3*nk) * 0.1  
R_force = np.eye(5) * 0.05 
# ukf.R = np.block([
#             [R_pos, np.zeros((nq, 5))],
#             [np.zeros((5, nq)), R_force]
#         ]) 
ukf.R = np.block([
            [R_pos, np.zeros((3*nk, 5))],
            [np.zeros((5, 3*nk)), R_force]
        ]) 
ukf.fx = fx
ukf.hx = hx

# LOOP
obs = env.reset()
frames = []
for idx in tqdm(range(kinematics.shape[0])):
    # # Reference 
    # joint_ids = [2, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22]
    # body_ids = [21, 28, 33, 38, 43]
    # lista = [4, 8, 12, 16, 20]
    # for i in joint_ids:
    #     data_ref.xanchor[i] = kinematics[idx,1+i:4+i]
    # for i, j in enumerate(body_ids):

    # data_ref.qpos = kinematics[idx, 1:]
    # mj.mj_step1(model_ref, data_ref)

    # Prediction UKF
    ukf.predict()
    kinematics_row = kinematics[idx, 1:] 
    kinetics_row = kinetics[idx, 1:]
    z = np.concatenate((kinematics_row, kinetics_row))
    ukf.update(z)
    x = ukf.x

    real_time_simulation[idx,:] = data_test.time

    # pos_joints = data_test.xanchor
    # vector_total = []
    # joint_ids = [2, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22]
    # body_ids = [21, 28, 33, 38, 43]
    # lista = [4, 8, 12, 16, 20]
    # for i in joint_ids:
    #     joint_id = i
    #     joint_name_offset = model_test.name_jntadr[joint_id]
    #     joint_name = mj.mj_id2name(model_test, mj.mjtObj.mjOBJ_JOINT, joint_id)
    #     vector_total.append(pos_joints[i])    
    # for i, j in enumerate(body_ids):
    #     vector_total.insert(lista[i], data_test.xpos[j])
    # vector_total = np.array(vector_total)
    # actualizar_grafica(vector_total)
    # keypoints_flat = np.array(vector_total).flatten()
    # kinematics_predicted[idx,:] = np.hstack((kinematics[idx,0], keypoints_flat))

    kinetics_predicted[idx,:] = np.hstack((kinetics[idx,0], x[2*nq:]))
    all_frcs[idx,:,0] = np.hstack((kinetics[idx,0], x[2*nq:]))
    # Inverse Dynamics
    target_qpos = x[:nq]
    qfrc = get_qfrc(model_test, data_test, target_qpos)
    all_qpos[idx,:,0] = np.hstack((kinematics_predicted[idx, 0], data_test.qpos))
    all_qfrc[idx,:] = np.hstack((kinematics_predicted[idx, 0], qfrc))
    # Quadratic Problem
    ctrl = get_ctrl(model_test, data_test, target_qpos, qfrc, 100, 5)
    data_test.ctrl = ctrl
    mj.mj_step(model_test, data_test)
    all_ctrl[idx,:] = np.hstack((kinematics_predicted[idx, 0], ctrl))
    # Rendering
    if not idx % round(0.3/(model_test.opt.timestep*25)):
        renderer_ref.update_scene(data_ref, camera=camera, scene_option=options_ref)
        frame_ref = renderer_ref.render()
        renderer_test.update_scene(data_test, camera=camera, scene_option=options_test)
        frame = renderer_test.render()
        frame_merged = np.append(frame_ref, frame, axis=1)
        frames.append(frame_merged)

error_rad = np.sqrt(((all_qpos[:,1:,0] - all_qpos[:,1:,-1])**2)).mean(axis=0)
error_deg = (180*error_rad)/np.pi
print(f'error max (rad): {error_rad.max()}')
print(f'error max (deg): {error_deg.max()}')
joint_names = [model_test.joint(i).name for i in range(model_test.nq)]
plot_qxxx(all_qpos, joint_names, ['Predicted qpos', 'Reference qpos'])
plot_qxxx_2d(all_qfrc, joint_names, ['Predicted qfrc'])
muscle_names = [model_test.actuator(i).name for i in range(model_test.nu)]
plot_uxxx_2d(all_ctrl, muscle_names, ['Predicted ctrl'])
fingertips_names = ['Thumb Fingertip', 'Index Fingertip', 'Middle Fingertip', 'Ring Fingertip', 'Little Fingertip']
plot_fxxx(all_frcs, fingertips_names, ['Predicted force', 'Reference force'])

# SAVE
output_name = os.path.join(os.path.dirname(__file__), "videos/ukf2_draft.mp4")
skvideo.io.vwrite(output_name, np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
output_path = os.path.join(os.path.dirname(__file__), "trajectories/simulation/kinematics_predicted_ukf2_draft_2.csv")
pd.DataFrame(kinematics_predicted).to_csv(output_path, index=False, header=False)
output_path = os.path.join(os.path.dirname(__file__), "trajectories/simulation/kinetics_predicted_ukf2_draft_2.csv")
pd.DataFrame(kinetics_predicted).to_csv(output_path, index=False, header=False)
output_path = os.path.join(os.path.dirname(__file__), "trajectories/simulation/time_simulation_ukf2_draft_2.csv")
pd.DataFrame(real_time_simulation).to_csv(output_path, index=False, header=False)
# output_path = os.path.join(os.path.dirname(__file__), "trajectories/simulation/kinematics_keypoints_draft_2.csv")
# pd.DataFrame(kinematics_keypoints).to_csv(output_path, index=False, header=False)