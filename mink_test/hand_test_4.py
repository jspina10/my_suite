from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

import time

import sys
import os
import gym
sys.path.append(os.getcwd())
from _myosuite.envs.myo import myobase
from loop_rate_limiters import RateLimiter

# Crea l'ambiente di MyoHand
env = gym.make("my_MyoHandEnvForce-v0", frame_skip=1, normalize_act=False)

if __name__ == "__main__":
    # Ottieni il modello e i dati dall'ambiente
    model = env.sim.model._model
    # data = env.sim.data
    data = mujoco.MjData(model) 

    # Configurazione del modello per mink
    configuration = mink.Configuration(model)
    posture_task = mink.PostureTask(model, cost=1e-2)

    # Definisci i nomi dei corpi associati alle dita
    fingers = ["distal_thumb_f", "2distph_f", "3distph_f", "4distph_f", "5distph_f"]
    finger_tasks = []
    for finger in fingers:
        task = mink.FrameTask(
            frame_name=finger,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        finger_tasks.append(task)

    # Elenco dei task per il solver
    tasks = [
        posture_task,
        *finger_tasks,
    ]

    # Definisci le posizioni target in coordinate cartesiane per ciascun corpo mocap
    target_positions = {
        "distal_thumb_f_target": [0, 0, 0],
        "2distph_f_target": [0, 0, 0],
        "3distph_f_target": [0, 0, 0],
        "4distph_f_target": [0, 0, 0],
        "5distph_f_target": [0, 0, 0]
    }

    # Imposta il solver per l'inverse kinematics
    solver = "quadprog"

    # Avvia il visualizzatore MuJoCo
    with mujoco.viewer.launch_passive(
        model=model, data=data
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        posture_task.set_target_from_configuration(configuration)

        # Ciclo principale della simulazione
        rate = RateLimiter(frequency=500.0, warn=False)
        while viewer.is_running():
            # Assegna le posizioni ai corpi mocap direttamente nelle coordinate specificate
            for (i,(finger, pos)) in enumerate(target_positions.items()):
                mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, finger)
                data.mocap_pos[i] = pos
                print(mocap_id)
                print(data.mocap_pos)
                print(data.mocap_pos.shape)
                print(finger)
                finger_tasks[i].set_target(
                    mink.SE3.from_mocap_name(model, data, f"{finger}")
                )

            tasks = [
                posture_task,
                *finger_tasks,
            ]

            # Risolvi i compiti IK e aggiorna la configurazione
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-5)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Visualizza e sincronizza
            viewer.sync()
            rate.sleep()

time.sleep(100)
