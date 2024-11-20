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

nq = model_test.nq
nu = model_test.nu
nf = 5
nk = 21
# kinematics_qpos = pd.read_csv(os.path.join(os.path.dirname(__file__), "trajectories/traj_standard.csv")).values
kinematics = pd.read_csv(os.path.join(os.path.dirname(__file__), "trajectories/hand_data_interpolated.csv")).values
# CAMERA
camera = mj.MjvCamera()
camera.azimuth = 166.553
camera.distance = 1.178
camera.elevation = -36.793
camera.lookat = np.array([-0.93762553, -0.34088276, 0.85067529])
# FUNTIONS
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
        options={'disp': True, 'maxiter': 500}
    )

    if result.success:
        return result.x  # Retorna el qpos óptimo
    else:
        raise ValueError("No se encontró una solución adecuada: " + result.message)


obs = env.reset()
frames = []
last_result = None
for idx in tqdm(range(kinematics.shape[0])):
    if idx == 0:
        initial_qpos = np.zeros(nq)
    else:
        initial_qpos = last_result
    desired_cartesian_positions = kinematics[idx,1:]
    # initial_qpos = np.zeros(nq)
    result = compute_qpos_from_cartesian(desired_cartesian_positions, initial_qpos, model_test, data_test)
    last_result = result

    data_test.qpos[:] = result
    mj.mj_step1(model_test, data_test)
    # Rendering
    if not idx % round(0.3/(model_test.opt.timestep*25)):
        renderer_test.update_scene(data_test, camera=camera, scene_option=options_test)
        frame = renderer_test.render()
        frames.append(frame)

output_name = os.path.join(os.path.dirname(__file__), "videos/draft_init.mp4")
skvideo.io.vwrite(output_name, np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
