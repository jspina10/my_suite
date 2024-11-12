import sys
import os
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink
import gym
import pandas as pd

sys.path.append(os.getcwd())
from _myosuite.envs.myo import myobase

# Cargar la trayectoria desde un archivo CSV
kinematics = pd.read_csv("trajectories/simulation/kinematics_fingerprints_ref.csv").values

env = gym.make("my_MyoHandEnvForce-v0", frame_skip=1, normalize_act=False)

if __name__ == "__main__":
    model = env.sim.model._model
    configuration = mink.Configuration(model)
    posture_task = mink.PostureTask(model, cost=1e-2)

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

    tasks = [posture_task, *finger_tasks]

    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        configuration.update_from_keyframe("zero")

        # Inicializar los cuerpos de captura de movimiento (mocap)
        posture_task.set_target_from_configuration(configuration)
        for finger in fingers:
            mink.move_mocap_to_frame(model, data, f"{finger}_target", finger, "site")

        rate = RateLimiter(frequency=500.0, warn=False)
        dt = rate.dt
        t = 0
        step_index = 0

        while viewer.is_running() and step_index < len(kinematics):
            # Extraer las posiciones de la trayectoria para cada dedo
            trajectory_point = kinematics[step_index, 1:].reshape(-1, 3)

            for i, (finger, task) in enumerate(zip(fingers, finger_tasks)):
                position_target = trajectory_point[i]
                task.set_target(mink.SE3(pos=position_target))

            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-5)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            viewer.sync()
            rate.sleep()
            t += dt
            step_index += 1
