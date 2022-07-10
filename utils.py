'''
func:
- scale_to_mg
- normalize_state
- cal_cost
- extra_reward
- plot_return
- plot_pf_results
- view_profile
'''

import os
import logging
import pickle
from pathlib import Path
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from setting import *

# --- Action Scaling ---
def scale_to_mg(nn_action, min_action, max_action):
    nn_action = np.clip(nn_action, -1., 1.)
    return (nn_action + 1) * (max_action - min_action) / 2 + min_action

# --- Normalization ---
def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class NormalizeAction:
    def __init__(self, epsilon=1e-8):
        self.a_rms = RunningMeanStd(shape=(N_ACTION,))
        self.epsilon = epsilon

    def normalize(self, a):
        self.a_rms.update(a)
        a = (a -self.a_rms.mean) / np.sqrt(self.a_rms.var + self.epsilon)
        a = np.clip(a, -5, 5)
        return a

    def tf_normalize(self, a):
        mean = tf.convert_to_tensor(self.a_rms.mean, dtype=tf.float32)
        var = tf.convert_to_tensor(self.a_rms.var, dtype=tf.float32)
        a = (a - mean) / tf.math.sqrt(var + self.epsilon)
        a = tf.clip_by_value(a, -5, 5)
        return a

class NormalizeObservation:
    def __init__(self, epsilon=1e-8):
        self.obs_seq_rms = RunningMeanStd(shape=STATE_SEQ_SHAPE)
        self.obs_fnn_rms = RunningMeanStd(shape=STATE_FNN_SHAPE)
        self.epsilon = epsilon

    def normalize(self, obs, update=True):
        obs_seq, obs_fnn = obs
        if update:
            self.obs_seq_rms.update(obs_seq)
            self.obs_fnn_rms.update(obs_fnn)
        obs_seq = (obs_seq - self.obs_seq_rms.mean) / np.sqrt(self.obs_seq_rms.var + self.epsilon)
        obs_seq = np.clip(obs_seq, -5, 5)
        obs_fnn = (obs_fnn - self.obs_fnn_rms.mean) / np.sqrt(self.obs_fnn_rms.var + self.epsilon)
        obs_fnn = np.clip(obs_fnn, -5, 5)
        return obs_seq, obs_fnn
    
    def save(self, dir):
        fpath = Path(os.path.join(dir, 'obs.pkl'))
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, 'wb') as f:
            pickle.dump({
                'obs_seq_mean': self.obs_seq_rms.mean,
                'obs_seq_var': self.obs_seq_rms.var,
                'obs_fnn_mean': self.obs_fnn_rms.mean,
                'obs_fnn_var': self.obs_fnn_rms.var,
            }, f)

    def load(self, dir):
        with open(os.path.join(dir, 'obs.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.obs_seq_rms.mean = data['obs_seq_mean']
            self.obs_seq_rms.var = data['obs_seq_var']
            self.obs_fnn_rms.mean = data['obs_fnn_mean']
            self.obs_fnn_rms.var = data['obs_fnn_var']

class NormalizeReward:
    def __init__(self, gamma=GAMMA, epsilon=1e-8):
        self.return_rms = RunningMeanStd()
        self.return_ = np.zeros(1)
        self.gamma = gamma
        self.epsilon = epsilon

    def normalize(self, r):
        self.return_ = r + self.gamma * self.return_
        self.return_rms.update(self.return_)
        r /= np.sqrt(self.return_rms.var + self.epsilon)
        r = np.clip(r, -5, 5)
        return r

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def normalize_state(state) -> Dict:
    normalized_state_rnn = state[0] / np.array([P_EXCESS_MAX, C_PRICE_MAX])
    # normalized_state_rnn = state[0] / np.array([*P_PV_MAX_LIST, *P_WT_MAX_LIST, *P_LOAD_MAX_LIST, C_PRICE_MAX])
    # normalized_state_rnn = state[0] / np.array([*P_PV_MAX_LIST, *P_WT_MAX_LIST, *P_LOAD_MAX_LIST, P_EXCESS_MAX, C_PRICE_MAX])
    normalized_state_fnn = state[1] / SOC_MAX

    return normalized_state_rnn, normalized_state_fnn

# --- Reward ---
def cal_cost(price, pcc_p_mw, bat5_soc_now, bat5_soc_prev, bat10_soc_now, bat10_soc_prev, **kwargs):
    transaction_cost = price * pcc_p_mw
    # mgt_cost = C_MGT5[0] * pow(mgt5_p_mw, 2) + C_MGT5[1] * mgt5_p_mw + \
    #     C_MGT9[0] * pow(mgt9_p_mw, 2) + C_MGT9[1] * mgt9_p_mw + \
    #     C_MGT10[0] * pow(mgt10_p_mw, 2) + C_MGT10[1] * mgt10_p_mw
    battery_cost = C_BAT5_DoD * pow((bat5_soc_now - bat5_soc_prev), 2) + \
        C_BAT10_DoD * pow((bat10_soc_now - bat10_soc_prev), 2)
    soc_penalty = C_SOC_LIMIT if ((bat5_soc_now > (1+SOC_TOLERANCE)*SOC_MAX or bat5_soc_now < (1-SOC_TOLERANCE)*SOC_MIN) or 
        (bat10_soc_now > (1+SOC_TOLERANCE)*SOC_MAX or bat10_soc_now < (1-SOC_TOLERANCE)*SOC_MIN)) else 0

    if len(kwargs):
        ids = kwargs['ids']
        t = kwargs['t']
        net = kwargs['net']
        log_cost_info(transaction_cost, battery_cost, soc_penalty, t, net=net, ids=ids, pcc_p_mw=pcc_p_mw)

    cost = (transaction_cost + battery_cost) * HOUR_PER_TIME_STEP + soc_penalty 
    normalized_cost = cost / MAX_COST

    return cost, normalized_cost

def extra_reward(nn_bat_p_mw, valid_bat_p_mw):
    # penalty for invalid action
    dif = np.sum(np.abs(nn_bat_p_mw - valid_bat_p_mw))
    dif /= (P_B10_MAX + P_B5_MAX)
    reward = 0. if (dif < 1e-3) else (REWARD_INVALID_ACTION + dif * REWARD_INVALID_ACTION)
    return reward

# --- Plot ---
def plot_ep_values(ep_values, train_length, epochs, ylabel):
    runs = ep_values.shape[0]
    fig_path = os.path.join('plot', f'{int(train_length/24)}days_{runs}runs_{epochs}eps_{str.lower(ylabel)}.png')
    arr_path = os.path.join('plot', f'{int(train_length/24)}days_{runs}runs_{epochs}eps_{str.lower(ylabel)}.npy')
    np.save(arr_path, ep_values)

    ep_return = np.median(ep_values, axis=0)
    epochs = range(1, len(ep_return) + 1)
    plt.plot(epochs, ep_return)
    plt.title(f'Training')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.savefig(fig_path)
    plt.show()

def plot_pf_results(dir, start, length):
    # pv, wt, mgt, load, bat, util, excess
    res_sgen_file = os.path.join(dir, 'res_sgen', 'p_mw.csv')
    res_load_file = os.path.join(dir, 'res_load', 'p_mw.csv')
    res_storage_file = os.path.join(dir, 'res_storage', 'p_mw.csv')
    res_trafo_file = os.path.join(dir, 'res_trafo', 'p_lv_mw.csv')

    # pv, wt, mgt
    sgen_p_mw = pd.read_csv(res_sgen_file)
    pv_p_mw = sgen_p_mw.iloc[:, 1:9]
    pv_p_mw.columns = ['pv3', 'pv4', 'pv5', 'pv6', 'pv8', 'pv9', 'pv10', 'pv11']
    wt_p_mw = sgen_p_mw.iloc[:, [9]]
    wt_p_mw.columns = ['wt7']
    # mgt_p_mw = sgen_p_mw.iloc[:, 10:]
    # mgt_p_mw.columns = ['mgt5', 'mgt9', 'mgt10']

    # load
    load_p_mw = pd.read_csv(res_load_file)
    load_p_mw = load_p_mw.iloc[:, 1:]
    load_p_mw.columns = ['load_r1', 'load_r3', 'load_r4', 'load_r5', 'load_r6', 'load_r8', 'load_r10', 'load_r11']

    # bat
    bat_p_mw = pd.read_csv(res_storage_file)
    bat_p_mw = bat_p_mw.iloc[:, 1:]
    bat_p_mw.columns = ['bat5', 'bat10']

    # utility
    trafo_p_mw = pd.read_csv(res_trafo_file)
    util_p_mw = -trafo_p_mw.iloc[:, [1]]
    util_p_mw.columns = ['utility']

    # price
    price = pd.read_csv(os.path.join('.', 'data', 'profile', 'price_profile.csv'))

    excess_p_mw = pv_p_mw.sum(axis=1) + wt_p_mw.sum(axis=1) - load_p_mw.sum(axis=1)
    excess_p_mw = pd.DataFrame({'excess': excess_p_mw})

    ax = excess_p_mw.iloc[start: start+length].plot(drawstyle='steps-post')
    bat_p_mw.iloc[start: start+length].plot(ax=ax, drawstyle='steps-post')
    price.iloc[start: start+length].plot(ax=ax, drawstyle='steps-post')
    plt.title('Power Flow')
    plt.xlabel('hour')
    plt.ylabel('MW')
    plt.show()

def view_profile(pv_profile, wt_profile, load_profile, price_profile, start=None, length=None):
    start = 0 if start is None else start
    length = (len(pv_profile.index)-start) if length is None else length
    pv_p_mw = pv_profile.iloc[start: start+length, :]
    wt_p_mw = wt_profile.iloc[start: start+length, :]
    load_p_mw = load_profile.iloc[start: start+length, :]
    price_profile = price_profile.iloc[start: start+length, :]

    # MW and excess profile
    profile_p_mw = pd.concat([pv_p_mw, wt_p_mw, load_p_mw]).iloc[start: start+length, :]
    excess_profile = pv_p_mw.sum(axis=1) + wt_p_mw.sum(axis=1) - load_p_mw.sum(axis=1)
    excess_profile = pd.DataFrame({'Excess': excess_profile})

    # info
    print('--- Profile ---')
    print(f'PV:\n max = {pv_profile.max(numeric_only=True)}, \nmin = {pv_profile.min(numeric_only=True)}')
    print(f'WT:\n max = {wt_profile.max(numeric_only=True)}, \nmin = {wt_profile.min(numeric_only=True)}')
    print(f'Load:\n max = {load_profile.max(numeric_only=True)}, \nmin = {load_profile.min(numeric_only=True)}')
    print(f'Excess:\n max = {excess_profile.max(numeric_only=True)}, \nmin = {excess_profile.min(numeric_only=True)}')
    print(f'Price:\n max = {price_profile.max(numeric_only=True)}, \nmin = {price_profile.min(numeric_only=True)}')

    # plot
    pv_p_mw.plot(xlabel='hour', ylabel='p_mw', title='PV')
    wt_p_mw.plot(xlabel='hour', ylabel='p_mw', title='WT')
    load_p_mw.plot(xlabel='hour', ylabel='p_mw', title='Load')
    price_profile.plot(xlabel='hour', ylabel='price', title='Price')
    profile_p_mw.plot(xlabel='hour', ylabel='p_mw', title='Microgrid')
    ax = excess_profile.plot(xlabel='hour', ylabel='p_mw', title='excess')
    ax.plot(range(start, start+length), np.zeros((length),))
    plt.show()

# --- Logging ---
def log_actor_critic_info(actor_loss, critic_loss, t=None, freq=20, **kwargs):
    if t is None:
        logging.info('--- Learn ---')
        logging.info(f'actor loss = {actor_loss}')
        logging.info(f'critic loss = {critic_loss}')
        return

    if t % freq == 0:
        logging.info('--- Learn ---')
        logging.info(f'actor loss = {actor_loss}')
        logging.info(f'critic loss = {critic_loss}')

def log_cost_info(transaction_cost, battery_cost, soc_penalty, t, freq=100, **kwargs):
    if t % freq == 0:
        net = kwargs['net']
        ids = kwargs['ids']
        pcc_p_mw = kwargs['pcc_p_mw']
        p_wt = net.res_sgen['p_mw'].iloc[ids['wt7']].sum()
        p_pv = net.res_sgen['p_mw'].sum() - p_wt
        p_bat = net.res_storage['p_mw'].sum()
        p_load = net.res_load['p_mw'].sum()
        excess = p_pv + p_wt - p_bat - p_load

        logging.info('--- Cost ---')
        logging.info(f'trans: {transaction_cost:.3f}, bat: {battery_cost:.3f}, soc: {soc_penalty:.3f}')
        logging.info('--- Power flow ---')
        logging.info(f'pcc = {pcc_p_mw:.3f}, excess = {excess:.3f},  pv = {p_pv:.3f}, wt = {p_wt:.3f}, bat = {p_bat:.3f}, load = {p_load:.3f}')

def log_trans_info(s, a, t, freq=100, **kwargs):
    if t % freq == 0:
        s_seq = s[0]    
        s_fnn = s[1]

        logging.info('--- State ---')
        logging.info(f'shape: ({s_seq.shape}, {s_fnn.shape})')
        logging.info(f'content: {s_seq[0]}, {s_fnn}')
        logging.info('--- Action ---')
        logging.info(f'shape: {a.shape}')
        logging.info(f'content: {a}')

# --- Others ---
def get_excess(pv_profile, wt_profile, load_profile, t):
    excess = pv_profile['pv3'][t] +\
        pv_profile['pv4'][t] +\
        pv_profile['pv5'][t] +\
        pv_profile['pv6'][t] +\
        pv_profile['pv8'][t] +\
        pv_profile['pv9'][t] +\
        pv_profile['pv10'][t] +\
        pv_profile['pv11'][t] +\
        wt_profile['wt7'][t] -\
        load_profile['load_r1'][t] -\
        load_profile['load_r3'][t] -\
        load_profile['load_r4'][t] -\
        load_profile['load_r5'][t] -\
        load_profile['load_r6'][t] -\
        load_profile['load_r8'][t] -\
        load_profile['load_r10'][t] -\
        load_profile['load_r11'][t]

    return excess

def policy_simple(net, ids, bat5_soc, bat10_soc, bat5_max_e_mwh, bat10_max_e_mwh):
    p_pv = net.sgen.at[ids.get('pv3'), 'p_mw'] +\
        net.sgen.at[ids.get('pv4'), 'p_mw'] +\
        net.sgen.at[ids.get('pv5'), 'p_mw'] +\
        net.sgen.at[ids.get('pv6'), 'p_mw'] +\
        net.sgen.at[ids.get('pv8'), 'p_mw'] +\
        net.sgen.at[ids.get('pv9'), 'p_mw'] +\
        net.sgen.at[ids.get('pv10'), 'p_mw'] +\
        net.sgen.at[ids.get('pv11'), 'p_mw']
    p_wt = net.sgen.at[ids.get('wt7'), 'p_mw']
    p_load = net.load.at[ids.get('load_r1'), 'p_mw'] +\
        net.load.at[ids.get('load_r3'), 'p_mw'] +\
        net.load.at[ids.get('load_r4'), 'p_mw'] +\
        net.load.at[ids.get('load_r5'), 'p_mw'] +\
        net.load.at[ids.get('load_r6'), 'p_mw'] +\
        net.load.at[ids.get('load_r8'), 'p_mw'] +\
        net.load.at[ids.get('load_r10'), 'p_mw'] +\
        net.load.at[ids.get('load_r11'), 'p_mw']
                        
    p_b5_max = min((SOC_MAX - bat5_soc) * bat5_max_e_mwh / HOUR_PER_TIME_STEP, P_B5_MAX)
    p_b5_min = max((SOC_MIN - bat5_soc) * bat5_max_e_mwh / HOUR_PER_TIME_STEP, P_B5_MIN)
    p_b10_max = min((SOC_MAX - bat10_soc) * bat10_max_e_mwh / HOUR_PER_TIME_STEP, P_B10_MAX)
    p_b10_min = max((SOC_MIN - bat10_soc) * bat10_max_e_mwh / HOUR_PER_TIME_STEP, P_B10_MIN)

    excess = p_pv + p_wt - p_load
    # print(f'Excess = {excess}, pv: {p_pv}, wt: {p_wt}, load: {p_load}')
    if excess > 0:
        # charge
        b5_ratio = p_b5_max / (p_b5_max + p_b10_max) if (p_b5_max + p_b10_max) != 0. else 0.
        b10_ratio = p_b10_max / (p_b5_max + p_b10_max) if (p_b5_max + p_b10_max) != 0. else 0.
        p_b5 = min(excess * b5_ratio, p_b5_max)
        p_b10 = min(excess * b10_ratio, p_b10_max)
        # p_mgt5 = 0.
        # p_mgt9 = 0.
        # p_mgt10 = 0.
    else:
        # discharge
        b5_ratio = p_b5_min / (p_b5_min + p_b10_min) if (p_b5_min + p_b10_min) != 0. else 0.
        b10_ratio = p_b10_min / (p_b5_min + p_b10_min) if (p_b5_min + p_b10_min) != 0. else 0.
        p_b5 = max(excess * b5_ratio, p_b5_min)
        p_b10 = max(excess * b10_ratio, p_b10_min)
        p_b = p_b5 + p_b10

        # mgt5_ratio = P_MGT5_MAX / (P_MGT5_MAX + P_MGT9_MAX + P_MGT10_MAX)
        # mgt9_ratio = P_MGT9_MAX / (P_MGT5_MAX + P_MGT9_MAX + P_MGT10_MAX)
        # mgt10_ratio = P_MGT10_MAX / (P_MGT5_MAX + P_MGT9_MAX + P_MGT10_MAX)
        # mgt5_op_point = (C_BUY - C_MGT5[1]) / C_MGT5[0]
        # mgt9_op_point = (C_BUY - C_MGT9[1]) / C_MGT9[0]
        # mgt10_op_point = (C_BUY - C_MGT10[1]) / C_MGT10[0]
        # p_mgt5 = 0. if excess > p_b  else min((p_b - excess) * mgt5_ratio, mgt5_op_point)
        # p_mgt9 = 0. if excess > p_b  else min((p_b - excess) * mgt9_ratio, mgt9_op_point)
        # p_mgt10 = 0. if excess > p_b  else min((p_b - excess) * mgt10_ratio, mgt10_op_point)
    
    return np.array([p_b5, p_b10])