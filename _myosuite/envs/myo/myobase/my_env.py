""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
import gym
import numpy as np

import mujoco as mj
from copy import deepcopy
import matplotlib.pyplot as plt
import scipy.sparse as spa
import pandas as pd
import numpy as np
import skvideo.io
import osqp
import os

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.simhive.myo_sim.test_sims import TestSims as loader
from scipy.signal import butter, filtfilt
from IPython.display import HTML
from base64 import b64encode
from tqdm import tqdm
from mujoco.glfw import glfw
import myosuite
import time 


class MyPoseEnv(BaseV0):

    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'pose_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 1.0,
        "bonus": 4.0,
        "act_reg": 1.0,
        "penalty": 50,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)

        self._setup(**kwargs)
    def _setup(self,
            viz_site_targets:tuple = None,  # site to use for targets visualization []
            target_jnt_range:dict = None,   # joint ranges as tuples {name:(min, max)}_nq
            target_jnt_value:list = None,   # desired joint vector [des_qpos]_nq
            reset_type = "init",           # none; init; random
            target_type = "generate",       # generate; switch; fixed
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            pose_thd = 0.35,
            weight_bodyname = None,
            weight_range = None,
            **kwargs,
        ):
        self.reset_type = reset_type
        self.target_type = target_type
        self.pose_thd = pose_thd
        self.weight_bodyname = weight_bodyname
        self.weight_range = weight_range

        # resolve joint demands
        if target_jnt_range:
            self.target_jnt_ids = []
            self.target_jnt_range = []
            for jnt_name, jnt_range in target_jnt_range.items():
                self.target_jnt_ids.append(self.sim.model.joint_name2id(jnt_name))
                self.target_jnt_range.append(jnt_range)
            self.target_jnt_range = np.array(self.target_jnt_range)
            self.target_jnt_value = np.mean(self.target_jnt_range, axis=1)  # pseudo targets for init
        else:
            self.target_jnt_value = target_jnt_value

        super()._setup(obs_keys=obs_keys,
                weighted_reward_keys=weighted_reward_keys,
                sites=viz_site_targets,
                **kwargs,
                )

    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['qpos'] = self.sim.data.qpos[:].copy()
        self.obs_dict['qvel'] = self.sim.data.qvel[:].copy()*self.dt
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()
        self.obs_dict['pose_err'] = self.target_jnt_value - self.obs_dict['qpos']
        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs
    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['qpos'] = sim.data.qpos[:].copy()
        obs_dict['qvel'] = sim.data.qvel[:].copy()*self.dt
        obs_dict['act'] = sim.data.act[:].copy() if sim.model.na>0 else np.zeros_like(obs_dict['qpos'])
        obs_dict['pose_err'] = self.target_jnt_value - obs_dict['qpos']
        return obs_dict
    def get_reward_dict(self, obs_dict):
        pose_dist = np.linalg.norm(obs_dict['pose_err'], axis=-1)
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)
        if self.sim.model.na !=0: act_mag= act_mag/self.sim.model.na
        far_th = 4*np.pi/2
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('pose',    -1.*pose_dist),
            ('bonus',   1.*(pose_dist<self.pose_thd) + 1.*(pose_dist<1.5*self.pose_thd)),
            ('penalty', -1.*(pose_dist>far_th)),
            ('act_reg', -1.*act_mag),
            # Must keys
            ('sparse',  -1.0*pose_dist),
            ('solved',  pose_dist<self.pose_thd),
            ('done',    pose_dist>far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    # generate a valid target pose
    def get_target_pose(self):
        if self.target_type == "fixed":
            return self.target_jnt_value
        elif self.target_type == "generate":
            return self.np_random.uniform(low=self.target_jnt_range[:,0], high=self.target_jnt_range[:,1])
        else:
            raise TypeError("Unknown Target type: {}".format(self.target_type))
    # update sim with a new target pose
    def update_target(self, restore_sim=False):
        if restore_sim:
            qpos = self.sim.data.qpos[:].copy()
            qvel = self.sim.data.qvel[:].copy()
        # generate targets
        self.target_jnt_value = self.get_target_pose()
        # update finger-tip target viz
        self.sim.data.qpos[:] = self.target_jnt_value.copy()
        self.sim.forward()
        for isite in range(len(self.tip_sids)):
            self.sim.model.site_pos[self.target_sids[isite]] = self.sim.data.site_xpos[self.tip_sids[isite]].copy()
        if restore_sim:
            self.sim.data.qpos[:] = qpos[:]
            self.sim.data.qvel[:] = qvel[:]
        self.sim.forward()
    # reset_type = none; init; random; test; IC
    # target_type = generate; switch
    def reset(self, initial_joint_positions=None):

        # udpate wegith
        if self.weight_bodyname is not None:
            bid = self.sim.model.body_name2id(self.weight_bodyname)
            gid = self.sim.model.body_geomadr[bid]
            weight = self.np_random.uniform(low=self.weight_range[0], high=self.weight_range[1])
            self.sim.model.body_mass[bid] = weight
            self.sim_obsd.model.body_mass[bid] = weight
            # self.sim_obsd.model.geom_size[gid] = self.sim.model.geom_size[gid] * weight/10
            self.sim.model.geom_size[gid][0] = 0.01 + 2.5*weight/100
            # self.sim_obsd.model.geom_size[gid][0] = weight/10

        # update target
        if self.target_type == "generate":
            # use target_jnt_range to generate targets
            self.update_target(restore_sim=True)
        elif self.target_type == "switch":
            # switch between given target choices
            # TODO: Remove hard-coded numbers
            if self.target_jnt_value[0] != -0.145125:
                self.target_jnt_value = np.array([-0.145125, 0.92524251, 1.08978337, 1.39425813, -0.78286243, -0.77179383, -0.15042819, 0.64445902])
                self.sim.model.site_pos[self.target_sids[0]] = np.array([-0.11000209, -0.01753063, 0.20817679])
                self.sim.model.site_pos[self.target_sids[1]] = np.array([-0.1825131, 0.07417956, 0.11407256])
                self.sim.forward()
            else:
                self.target_jnt_value = np.array([-0.12756566, 0.06741454, 1.51352705, 0.91777418, -0.63884237, 0.22452487, 0.42103326, 0.4139465])
                self.sim.model.site_pos[self.target_sids[0]] = np.array([-0.11647777, -0.05180014, 0.19044284])
                self.sim.model.site_pos[self.target_sids[1]] = np.array([-0.17728016, 0.01489491, 0.17953786])
        elif self.target_type == "fixed":
            self.update_target(restore_sim=True)
        else:
            print("{} Target Type not found ".format(self.target_type))

        # update init state
        if self.reset_type is None or self.reset_type == "none":
            # no reset; use last state
            obs = self.get_obs()
        elif self.reset_type == "init":
            # reset to init state
            obs = super().reset()
        elif self.reset_type == "random":
            # reset to random state
            jnt_init = self.np_random.uniform(high=self.sim.model.jnt_range[:,1], low=self.sim.model.jnt_range[:,0])
            obs = super().reset(reset_qpos=jnt_init)
        elif self.reset_type == "test":
            jnt_init = np.array([-1.57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            obs = super().reset(reset_qpos=jnt_init)
        elif self.reset_type == "IC":
            # reste to init or initial position
            obs = super().reset(reset_qpos=initial_joint_positions)
        else:
            print("Reset Type not found")

        # if initial_joint_positions is not None:
        #     # self.sim.data.qpos[:] = initial_joint_positions
        #     # self.sim.forward()  # Aggiorna la simulazione con la nuova posizione
        #     obs = super().reset(reset_qpos=initial_joint_positions)
        
        return obs
    
# ----------------------------------------------------------------------------------------------------------------------

    def show_video(self, video_path, video_width = 400):
        """
        Display a video within the notebook.
        """
        video_file = open(video_path, "r+b").read()
        video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
        return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")
    def plot_qxxx(self, qxxx, joint_names, labels):
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
        plt.show()
    def plot_uxxx(self, uxxx, muscle_names, labels):
        """
        Plot actuator variables to be compared.
        uxxx[:,0,-1] = time axis
        uxxx[:,1:,n] = n-th sequence
        """
        fig, axs = plt.subplots(5, 8, figsize=(12, 8))
        axs = axs.flatten()
        line_objects = []
        for j in range(1, len(muscle_names)+1):
            ax = axs[j-1]
            for i in range(uxxx.shape[2]):
                line, = ax.plot(uxxx[:, 0, -1], uxxx[:, j, i])
                if j == 1: # add only one set of lines to the legend
                    line_objects.append(line)
            ax.set_xlim([uxxx[:, 0].min(), uxxx[:, 0].max()])
            ax.set_ylim([uxxx[:, 1:, :].min(), uxxx[:, 1:, :].max()])
            ax.set_title(muscle_names[j-1])
        legend_ax = axs[len(muscle_names)] # create legend in the 40th subplot area
        legend_ax.axis('off')
        legend_ax.legend(line_objects, labels, loc='center')
        plt.tight_layout()
        plt.show()
    def plot_qxxx_2d(self, qxxx, joint_names, labels):
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
        plt.show()
    def plot_uxxx_2d(self, uxxx, muscle_names, labels):
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
        plt.show()

    def solve_qp(self, P, q, lb, ub, x0):
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
    def get_qfrc(self, model, data, target_qpos):
        """
        Compute the generalized force needed to reach the target position in the next mujoco step.
        """
        data_copy = deepcopy(data)
        data_copy.qacc = (((target_qpos - data.qpos) / model.opt.timestep) - data.qvel) / model.opt.timestep
        model.opt.disableflags += mj.mjtDisableBit.mjDSBL_CONSTRAINT
        mj.mj_inverse(model, data_copy)
        model.opt.disableflags -= mj.mjtDisableBit.mjDSBL_CONSTRAINT
        return data_copy.qfrc_inverse
    def get_ctrl(self, model, data, target_qpos, qfrc, qfrc_scaler, qvel_scaler):
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
        x = self.solve_qp(P, q, lb, ub, x0)
        ctrl = act + x * t2 / (gain * ts - x * t1)
        return np.clip(ctrl,0,1)

    def apply_forces(self, model, data, forces):
        # Body IDs for distal phalanges
        body_ids = [21, 28, 33, 38, 43]  # Thumb and other finger IDs

        for i, body_id in enumerate(body_ids):
            # Extract the scalar force for the current finger
            scalar_force = forces[i]
            # Get the rotation matrix from the global frame to the local frame
            body_xmat = data.xmat[body_id].reshape(3, 3)
            # Construct the force vector (considering x and y components are zero)
            external_force_local = np.array([0, 0, scalar_force])
            global_force =  body_xmat @ external_force_local
            # Apply the local force to the body
            data.xfrc_applied[body_id, :3] = global_force 
