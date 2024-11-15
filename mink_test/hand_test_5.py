from pathlib import Path

import mujoco
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

# Ruta al archivo XML de la mano Shadow Hand
_HERE = Path(__file__).parent
_XML = _HERE / "shadow_hand" / "scene_left.xml"

def setup_tasks(model, desired_positions):
    """
    Configura tareas de marco para controlar las posiciones de los dedos.

    Args:
        model: Modelo MuJoCo cargado.
        desired_positions: Diccionario con posiciones deseadas (x, y, z) para cada dedo.

    Returns:
        Lista de tareas configuradas.
    """
    tasks = []
    for frame_name, position in desired_positions.items():
        task = mink.FrameTask(
            frame_name=frame_name,
            frame_type="site",  # Cambia si el frame no es un 'site' en tu modelo
            position_cost=1.0,
            orientation_cost=0.0,  # Ignorar orientación
            lm_damping=1e-3,  # Amortiguamiento para estabilidad
        )
        task.set_target(mink.SE3.from_translation(np.array(position)))  # Fijar objetivo en x, y, z
        tasks.append(task)
    return tasks

def main():
    # Cargar el modelo MuJoCo
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    configuration = mink.Configuration(model)
    data = configuration.data

    # Configurar posiciones deseadas (en metros)
    desired_positions = {
        "thumb": [0.1, 0.2, 0.3],   # Puntero del pulgar
        "first": [0.15, 0.25, 0.35],  # Puntero del índice
        "middle": [0.2, 0.3, 0.4],   # Puntero del medio
    }

    # Configurar tareas para cada dedo
    tasks = setup_tasks(model, desired_positions)

    # Configurar visualización
    with mujoco.viewer.launch_passive(
        model=model, data=data
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Inicializar configuración
        configuration.update_from_keyframe("grasp hard")  # Keyframe inicial
        rate = RateLimiter(frequency=500.0, warn=False)

        while viewer.is_running():
            # Actualizar objetivos dinámicos (si cambian)
            for task, (frame_name, position) in zip(tasks, desired_positions.items()):
                task.set_target(mink.SE3.from_translation(np.array(position)))

            # Resolver cinemática inversa
            velocities = mink.solve_ik(
                configuration=configuration,
                tasks=tasks,
                dt=rate.dt,
                solver="quadprog",
                damping=1e-5,
            )

            # Integrar la nueva configuración
            configuration.integrate_inplace(velocities, rate.dt)

            # Actualizar la visualización
            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()

if __name__ == "__main__":
    main()
