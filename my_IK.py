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

### INIT
env = gym.make("my_MyoHandEnvForce-v0", frame_skip=1, normalize_act=False)
model = env.sim.model._model
data = mj.MjData(model)
tausmooth = 5
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
kinematics = pd.read_csv(os.path.join(os.path.dirname(__file__), "trajectories/traj_keypoints_ref.csv")).values
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
    if idx == 0:
        current_generalized_positions = np.zeros(nq)
    else:
        current_generalized_positions = last_result
    desired_cartesian_positions = kinematics[idx,1:]
    result = compute_qpos_from_cartesian(
        desired_cartesian_positions, 
        current_generalized_positions, 
        model_test, 
        data_test)
    last_result = result




    # Aggiorna lo stato del modello con le posizioni calcolate
    data_test.qpos[:] = result
    mj.mj_forward(model_test, data_test)

    # Usa la funzione hx per calcolare la posizione dei punti chiave
    x_state = np.concatenate([data_test.qpos, data_test.qvel])
    calculated_keypoints = hx(x_state)

    # Salva la posizione dei punti chiave calcolata
    calculated_trajectories.append(calculated_keypoints)

    data_test.qpos[:] = result
    mj.mj_step1(model_test, data_test)
    # Rendering
    if not idx % round(0.3/(model_test.opt.timestep*25)):
        renderer_test.update_scene(data_test, camera=camera, scene_option=options_test)
        frame = renderer_test.render()
        frames.append(frame)

# Converti le posizioni calcolate in un array NumPy per un confronto più facile
calculated_trajectories = np.array(calculated_trajectories)

# Calcolo dell'errore tra la traiettoria originale e quella calcolata
errors = []
for idx in range(kinematics.shape[0]):
    original_positions = kinematics[idx, 1:]  # Posizioni originali dei punti chiave
    calculated_positions = calculated_trajectories[idx]  # Posizioni calcolate dei punti chiave

    # Calcola la distanza euclidea tra la traiettoria originale e quella calcolata
    error = np.linalg.norm(original_positions - calculated_positions)
    errors.append(error)

# Calcola l'errore medio
mean_error = np.mean(errors)

# Plot dell'errore nel tempo
plt.figure()
plt.plot(time_steps, errors, label='Errore Traiettoria')
plt.xlabel('Step temporale')
plt.ylabel('Errore (Distanza euclidea)')
plt.title('Errore tra la traiettoria originale e quella calcolata')
plt.legend()
plt.show()

print(f"Errore medio tra la traiettoria originale e quella calcolata: {mean_error}")

output_name = os.path.join(os.path.dirname(__file__), "videos/my_IK.mp4")
skvideo.io.vwrite(output_name, np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
