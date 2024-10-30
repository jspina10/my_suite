from gym.envs.registration import register
from myosuite.envs.env_variants import register_env_variant
from myosuite.envs.myo.myobase import register_env_with_variants

import os
import numpy as np

# print("MyoSuite:> Registering Myo Envs")

# MY_ENV
register_env_with_variants(id='my_MyoHandEnv-v0',
        entry_point='_myosuite.envs.myo.myobase.my_env:MyPoseEnv',
        max_episode_steps=100,
        kwargs={
            'model_path': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "simhive/myo_sim/hand/my_myohand.xml"),
            'viz_site_targets': None,
            'target_jnt_value': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'normalize_act': False,
            'pose_thd': .1,
            'reset_type': "init",        # none, init, random, test, IC
            'target_type': 'fixed',      # generate, fixed
        }
    )

# MY_ENV_FORCE
register_env_with_variants(id='my_MyoHandEnvForce-v0',
        entry_point='_myosuite.envs.myo.myobase.my_env:MyPoseEnv',
        max_episode_steps=100,
        kwargs={
            'model_path': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "simhive/myo_sim/hand/my_myohand_force.xml"),
            'viz_site_targets': None,
            'target_jnt_value': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'normalize_act': False,
            'pose_thd': .1,
            'reset_type': "init",        # none, init, random, test, IC
            'target_type': 'fixed',      # generate, fixed
        }
    )