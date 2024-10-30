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

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Inicialización de la figura global
plt.ion()  # Activar el modo interactivo de Matplotlib para actualizar la gráfica.
fig_2 = plt.figure()
ax_2 = fig_2.add_subplot(111, projection='3d')

def actualizar_grafica(arr):
    # Limpiar el gráfico actual para evitar superposiciones.
    ax_2.clear()
    # Asegurarse de que el arreglo sea de tipo numpy.ndarray.
    if not isinstance(arr, np.ndarray):
        print("El parámetro debe ser de tipo numpy.ndarray")
        return
    # Comprobar que el arreglo tenga 3 columnas (coordenadas x, y, z).
    if arr.shape[1] != 3:
        print("El arreglo debe tener exactamente 3 columnas para representar coordenadas 3D.")
        return
    # Extraer las coordenadas x, y, z.
    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]
    # # Dibujar los puntos en el espacio 3D.
    # ax_2.scatter(x, y, z, color='blue', marker='o', s=20)  # 's' define el tamaño de los puntos.
    # for i in range(len(x)):
    #     ax_2.text(x[i], y[i], z[i], f'{i}', color='red', fontsize=10)
    # Restar las coordenadas del primer punto para obtener posiciones relativas.
    x_rel = x - x[0]
    y_rel = y - y[0]
    z_rel = z - z[0]
    # Dibujar los puntos en el espacio 3D.
    ax_2.scatter(x_rel, y_rel, z_rel, color='blue', marker='o', s=20)  # 's' define el tamaño de los puntos.
    # Añadir los índices de los puntos.
    for i in range(len(x_rel)):
        ax_2.text(x_rel[i], y_rel[i], z_rel[i], f'{i}', color='red', fontsize=10)
    # Configurar etiquetas y título.
    ax_2.set_title('Gráfica de puntos en 3D')
    ax_2.set_xlabel('Eje X')
    ax_2.set_ylabel('Eje Y')
    ax_2.set_zlabel('Eje Z')
    # Configurar los límites de los ejes para una mejor visualización.
    ax_2.set_xlim([x_rel.min() - 0.1, x_rel.max() + 0.1])
    ax_2.set_ylim([y_rel.min() - 0.1, y_rel.max() + 0.1])
    ax_2.set_zlim([z_rel.min() - 0.1, z_rel.max() + 0.1])
    # ax_2.set_xlim([-10, 10])
    # ax_2.set_ylim([-10, 10])
    # ax_2.set_zlim([-10, 10])
    # Actualizar la visualización.
    plt.draw()
    plt.pause(0.0001)  # Pausar brevemente para permitir la actualización de la gráfica.

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
    plt.savefig('graphs/UKF2_qpos.png')  # Save the plot to a file
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
    plt.savefig('graphs/UKF2_qfrc.png')  # Save the plot to a file
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
    plt.savefig('graphs/UKF2_ctrl.png')  # Save the plot to a file
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
        scalar_force = forces[i]
        # print(scalar_force)
        # Get the rotation matrix from the global frame to the local frame
        body_xmat = data.xmat[body_id].reshape(3, 3)
        # Construct the force vector (assuming x and y components are zero)
        external_force_local = np.array([0, 0, scalar_force])
        global_force =  body_xmat @ external_force_local
        # Apply the local force to the body
        data.xfrc_applied[body_id, :3] = global_force 

def get_joint_anchor_global_position(data, joint_id):
    """
    Calcola la posizione globale dell'ancora del giunto.

    Args:
    data: Oggetto MuJoCo `mjData` contenente lo stato corrente della simulazione.
    joint_id: ID del giunto per cui si vuole calcolare la posizione dell'ancora.

    Returns:
    anchor_global: Posizione globale dell'ancora del giunto (array 3D).
    """
    # Ottieni l'ID del corpo a cui il giunto è connesso
    body_id = model.jnt_bodyid[joint_id]

    # Ottieni la posizione del corpo nel sistema globale
    body_pos_global = data.xpos[body_id]

    # Ottieni la matrice di rotazione del corpo (3x3)
    body_rot_matrix = data.xmat[body_id].reshape(3, 3)

    # Ottieni la posizione dell'ancora del giunto nelle coordinate locali
    anchor_local = data.xanchor[joint_id]

    # Trasforma la posizione dell'ancora dal sistema locale al sistema globale
    anchor_global = body_pos_global + np.dot(body_rot_matrix, anchor_local)

    return anchor_global

### INIT
env = gym.make("my_MyoHandEnvForce-v0", frame_skip=1, normalize_act=False)
tausmooth = 5
env.unwrapped.sim.model.actuator_dynprm[:,2] = tausmooth
model = env.sim.model._model
data = mj.MjData(model) 
# TEST
model_test = env.sim.model._model
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
    apply_forces(model, data, forces)
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
ukf.R = np.eye(dim_z) * 0.1
ukf.fx = fx
ukf.hx = hx



# # Carica la matrice delle traiettorie dal file CSV
# trajectory_matrix = pd.read_csv('traj_standard.csv').values
# # Indici dei joint di cui vogliamo salvare le posizioni cartesiane
# joint_indices = [2, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22]
# # Numero di istanti di tempo (righe della matrice delle traiettorie)
# num_timesteps = trajectory_matrix.shape[0]
# # Matrice per salvare le posizioni cartesiane dei joint
# cartesian_positions_matrix = np.zeros((num_timesteps, len(joint_indices) * 3))
# # Itera su ogni istante di tempo per calcolare le posizioni cartesiane dei joint
# for t in range(num_timesteps):
#     joint_rotations = trajectory_matrix[t]
#     # Aggiorna le rotazioni dei joint nel modello
#     data_ref.qpos[:len(joint_rotations)] = joint_rotations
#     # Esegui una forward dynamics per aggiornare lo stato del modello
#     mj.mj_forward(model_ref, data_ref)
#     # Estrai le posizioni cartesiane dei joint usando la funzione data_ref.site_xpos
#     cartesian_positions = []
#     for idx in joint_indices:
#         position = data_ref.site_xpos[idx]
#         cartesian_positions.extend(position)  # Aggiungi la posizione cartesiana (x, y, z) alla lista
#     # Salva le posizioni cartesiane nella matrice
#     cartesian_positions_matrix[t, :len(cartesian_positions)] = cartesian_positions
# # La matrice cartesian_positions_matrix ora contiene le posizioni cartesiane dei joint selezionati per ogni istante di tempo
# # Salva la matrice delle posizioni cartesiane in un file CSV
# pd.DataFrame(cartesian_positions_matrix).to_csv('cartesian_positions.csv', index=False)



# LOOP
obs = env.reset()
frames = []
for idx in tqdm(range(kinematics.shape[0])):
    # Reference 
    data_ref.qpos = kinematics[idx, 1:]
    mj.mj_step1(model_ref, data_ref)



    pos_joints = data.xanchor
    print(f"{pos_joints}")
    # time.sleep(10)
    vector_total = []
    lista = [2, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22]
    for i in lista:
        # ID del corpo corrisponde all'indice nell'array
        joint_id = i
        # usa l'array 'body_names' per ottenere il nome del corpo dall'ID
        joint_name_offset = model.name_jntadr[joint_id]
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        # joint_point = data.xanchor
        # print(f"Joint: {joint_name}, ID: {joint_id}")
        # anchor_global_position = get_joint_anchor_global_position(data, joint_id)
        # print("Posizione globale dell'ancora del giunto:", anchor_global_position)
        # print(f"Point:{joint_point}")
        vector_total.append(pos_joints[i])    
    vector_total = np.array(vector_total)
    print(f"{vector_total}")
    actualizar_grafica(vector_total)



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
    qfrc = get_qfrc(model_test, data_test, target_qpos)
    all_qpos[idx,:,0] = np.hstack((data_test.time, data_test.qpos))
    all_qfrc[idx,:] = np.hstack((data_test.time, qfrc))
    # Quadratic Problem
    ctrl = get_ctrl(model_test, data_test, target_qpos, qfrc, 100, 5)
    data_test.ctrl = ctrl
    mj.mj_step(model_test, data_test)
    all_ctrl[idx,:] = np.hstack((data_test.time, ctrl))
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
joint_names = [model.joint(i).name for i in range(model.nq)]
plot_qxxx(all_qpos, joint_names, ['Achieved qpos', 'Reference qpos'])
plot_qxxx_2d(all_qfrc, joint_names, ['Achieved qfrc'])
muscle_names = [model_test.actuator(i).name for i in range(model_test.nu)]
plot_uxxx_2d(all_ctrl, muscle_names, ['Achieved ctrl'])

# SAVE
output_name = os.path.join(os.path.dirname(__file__), "videos/ukf2.mp4")
skvideo.io.vwrite(output_name, np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
output_path = os.path.join(os.path.dirname(__file__), "trajectories/simulation/kinematics_predicted_ukf2.csv")
pd.DataFrame(kinematics_predicted).to_csv(output_path, index=False, header=False)
output_path = os.path.join(os.path.dirname(__file__), "trajectories/simulation/kinetics_predicted_ukf2.csv")
pd.DataFrame(kinetics_predicted).to_csv(output_path, index=False, header=False)
