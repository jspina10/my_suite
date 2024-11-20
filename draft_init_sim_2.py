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

# Inicialización de la figura global
plt.ion()  # Activar el modo interactivo de Matplotlib para actualizar la gráfica.
fig_2 = plt.figure()
ax_2 = fig_2.add_subplot(111, projection='3d')

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
    plt.savefig('graphs/UKF2_qpos_draft_init_sim.png')  # Save the plot to a file
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
    plt.savefig('graphs/UKF2_qfrc_draft_init_sim.png')  # Save the plot to a file
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
    plt.savefig('graphs/UKF2_ctrl_draft_init_sim.png')  # Save the plot to a file
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
    ax_2.set_xlim([x_rel.min() - 0.01, x_rel.max() + 0.01])
    ax_2.set_ylim([y_rel.min() - 0.01, y_rel.max() + 0.01])
    ax_2.set_zlim([z_rel.min() - 0.001, z_rel.max() + 0.001])
    # Update the visualization.
    plt.draw()
    plt.pause(0.0001)  # Break to permit the plot update.


### INIT
env = gym.make("my_MyoHandEnvForce-v0", frame_skip=1, normalize_act=False)
model = env.sim.model._model
data = mj.MjData(model)
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
nq = model_test.nq
nu = model_test.nu
nf = 0
nk = 21
# kinematics_qpos = pd.read_csv(os.path.join(os.path.dirname(__file__), "trajectories/traj_standard.csv")).values
kinematics = pd.read_csv(os.path.join(os.path.dirname(__file__), "trajectories/hand_data_interpolated.csv")).values
kinematics_predicted = np.zeros((kinematics.shape[0], kinematics.shape[1]))
real_time_simulation = np.zeros((kinematics.shape[0],1))
all_qpos = np.zeros((kinematics.shape[0], 1+nq, 2))
# all_qpos[:,:,-1] = kinematics_qpos[1:,:]
all_qfrc = np.zeros((kinematics.shape[0], 1+nq))
all_ctrl = np.zeros((kinematics.shape[0], 1+nu))
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
    # apply_forces(model, data, forces)
    mj.mj_step(model, data)
    x_new = np.concatenate((data.qpos, data.qvel, forces))    
    return x_new
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
    # Extract forces at fingerprints
    z_force = x[2*nq:]  # Forze predette ai polpastrelli
    # Combine keypoints positions and forces
    z = np.concatenate((keypoints_flat, z_force))
    # actualizar_grafica(np.array(keypoints))
    return z

from scipy.optimize import minimize

def compute_qpos_from_cartesian(desired_cartesian_positions, initial_qpos, model, data):
    """
    Calcula las posiciones articulares (qpos) necesarias para que el modelo alcance
    las posiciones cartesianas deseadas.

    Args:
        desired_cartesian_positions (np.ndarray): Array de posiciones cartesianas deseadas, 
                                                  de tamaño (N, 3), donde N es el número de keypoints.
        initial_qpos (np.ndarray): Estado inicial de las posiciones articulares (qpos).
        model (mj.MjModel): Modelo de MuJoCo cargado.
        data (mj.MjData): Datos asociados al modelo de MuJoCo.

    Returns:
        np.ndarray: Las posiciones articulares (qpos) calculadas.
    """
    nq = model.nq  # Número de variables articulares (qpos)

    def cost_function(qpos):
        """
        Función de costo que mide la distancia entre las posiciones actuales y deseadas.
        """
        # Actualizar el estado del modelo con qpos actual
        data.qpos[:] = qpos
        data.qvel[:] = 0  # Velocidades iniciales en cero
        mj.mj_forward(model, data)
        
        # Calcular las posiciones cartesianas actuales de los keypoints
        current_cartesian_positions = []
        joint_ids = [2, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22]
        body_ids = [21, 28, 33, 38, 43]
        lista = [4, 8, 12, 16, 20]
        
        for i in joint_ids:
            current_cartesian_positions.append(data.xanchor[i])
        for i, j in enumerate(body_ids):
            current_cartesian_positions.insert(lista[i], data.xpos[j])
        
        current_cartesian_positions = np.array(current_cartesian_positions).flatten()
        
        # Calcular la distancia entre las posiciones actuales y deseadas
        error = np.linalg.norm(current_cartesian_positions - desired_cartesian_positions.flatten())
        return error

    # Configuración del optimizador
    bounds = [(model.jnt_range[i][0], model.jnt_range[i][1]) for i in range(nq)]
    result = minimize(
        cost_function, 
        initial_qpos, 
        bounds=bounds, 
        method='SLSQP', 
        options={'disp': False, 'maxiter': 500}
    )

    if result.success:
        return result.x  # Retorna el qpos óptimo
    else:
        raise ValueError("No se encontró una solución adecuada: " + result.message)

# UKF 
dim_x = 2 * nq + nf
# dim_z = nq + nf
dim_z = 3 * nk + nf
alpha = 1
beta = 2
kappa = 3-dim_x
print(f"UKF parameters for σ-points: α={alpha}, β={beta}, κ={kappa}")
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



desired_cartesian_positions = kinematics[0,1:]
initial_qpos = np.zeros(nq)
result = compute_qpos_from_cartesian(
    desired_cartesian_positions,
    initial_qpos,
    model_ref,
    data_ref
)

obs = env.reset(initial_joint_positions=result)
frames = []

for idx in tqdm(range(kinematics.shape[0])):
    # Reference
    desired_cartesian_positions = kinematics[idx,1:]
    result = compute_qpos_from_cartesian(
        desired_cartesian_positions,
        initial_qpos,
        model_ref,
        data_ref
    )
    initial_qpos = result
    data_ref.qpos[:] = result
    mj.mj_step1(model_ref, data_ref)
    all_qpos[idx,:,-1] = np.hstack((kinematics[idx, 0], result))

    # Prediction UKF
    if idx == 0:
        data_test.qpos[:] = result
    z = kinematics[idx, 1:]
    ukf.predict()
    ukf.update(z)
    x = ukf.x

    # Model Test
    real_time_simulation[idx,:] = data_test.time
    # kinetics_predicted[idx,:] = np.hstack((kinetics[idx,0], x[2*nq:]))
    # all_frcs[idx,:,0] = np.hstack((kinetics[idx,0], x[2*nq:]))

    # Inverse Dynamics
    target_qpos = x[:nq]
    qfrc = get_qfrc(model_test, data_test, target_qpos)
    # Quadratic Problem
    ctrl = get_ctrl(model_test, data_test, target_qpos, qfrc, 100, 5)
    data_test.ctrl = ctrl
    mj.mj_step(model_test, data_test)
    all_qpos[idx,:,0] = np.hstack((kinematics[idx, 0], data_test.qpos))
    all_qfrc[idx,:] = np.hstack((kinematics[idx, 0], qfrc))
    all_ctrl[idx,:] = np.hstack((kinematics[idx, 0], ctrl))

    # Rendering
    if not idx % round(0.3/(model_test.opt.timestep*25)):
        renderer_ref.update_scene(data_ref, camera=camera, scene_option=options_ref)
        frame_ref = renderer_ref.render()
        renderer_test.update_scene(data_test, camera=camera, scene_option=options_test)
        frame = renderer_test.render()
        frame_merged = np.append(frame_ref, frame, axis=1)
        frames.append(frame_merged)

### PRINTS & PLOTS
error_rad = np.sqrt(((all_qpos[:,1:,0] - all_qpos[:,1:,-1])**2)).mean(axis=0)
error_deg = (180*error_rad)/np.pi
print(f'error max (rad): {error_rad.max()}')
print(f'error max (deg): {error_deg.max()}')
joint_names = [model.joint(i).name for i in range(nq)]
plot_qxxx(all_qpos, joint_names, ['Predicted qpos', 'Reference qpos'])
plot_qxxx_2d(all_qfrc, joint_names, ['Predicted qfrc'])
muscle_names = [model_test.actuator(i).name for i in range(nu)]
plot_uxxx_2d(all_ctrl, muscle_names, ['Predicted ctrl'])
# fingertips_names = ['Thumb Fingertip', 'Index Fingertip', 'Middle Fingertip', 'Ring Fingertip', 'Little Fingertip']
# plot_fxxx(all_frcs, fingertips_names, ['Predicted force', 'Reference force'])

# SAVE
output_name = os.path.join(os.path.dirname(__file__), "videos/draft_init_sim.mp4")
skvideo.io.vwrite(output_name, np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
output_path = os.path.join(os.path.dirname(__file__), "trajectories/simulation/kinematics_predicted_draft_init_sim.csv")
pd.DataFrame(kinematics_predicted).to_csv(output_path, index=False, header=False)
output_path = os.path.join(os.path.dirname(__file__), "trajectories/simulation/time_simulation_draft_init_sim.csv")
pd.DataFrame(real_time_simulation).to_csv(output_path, index=False, header=False)
