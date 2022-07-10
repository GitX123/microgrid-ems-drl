import numpy as np
from pandapower.control.basic_controller import Controller

import utils
from setting import *

class SimpleControl(Controller):
    def __init__(self, net, ids, **kwargs):
        super().__init__(net, **kwargs)
        self.rewards = []
        self.costs = []
        self.bat5_soc_prev = 0.
        self.bat10_soc_prev = 0.
        self.bat5_soc = 0.
        self.bat10_soc = 0.
        self.soc_history = {'bat5_soc': [self.bat5_soc], 'bat10_soc': [self.bat10_soc]}
        self.last_time_step = None
        self.applied = False

        self.price_profile = kwargs['price_profile']

        self.ids = ids
        self.trafo0_id = ids.get('trafo0')
        # self.mgt5_id = ids.get('mgt5')
        # self.mgt5_p_mw = net.sgen.at[self.mgt5_id, 'p_mw']
        # self.mgt9_id = ids.get('mgt9')
        # self.mgt9_p_mw = net.sgen.at[self.mgt9_id, 'p_mw']
        # self.mgt10_id = ids.get('mgt10')
        # self.mgt10_p_mw = net.sgen.at[self.mgt10_id, 'p_mw']

        self.bat5_id = ids.get('bat5')
        self.bat5_p_mw = net.storage.at[self.bat5_id, 'p_mw']
        self.bat5_max_e_mwh = net.storage.at[self.bat5_id, 'max_e_mwh']
        self.bat10_id = ids.get('bat10')
        self.bat10_p_mw = net.storage.at[self.bat10_id, 'p_mw']
        self.bat10_max_e_mwh = net.storage.at[self.bat10_id, 'max_e_mwh']

    def is_converged(self, net) -> bool:
        return self.applied

    def calculate_reward(self, net, t):
        price = self.price_profile['price'][t - 1]
        cost, normalized_cost = utils.cal_cost(
            price=price,
            pcc_p_mw=-net.res_trafo.at[self.trafo0_id, 'p_lv_mw'],
            # mgt5_p_mw=self.mgt5_p_mw,
            # mgt9_p_mw=self.mgt9_p_mw,
            # mgt10_p_mw=self.mgt10_p_mw,
            bat5_soc_now=self.bat5_soc,
            bat5_soc_prev=self.bat5_soc_prev,
            bat10_soc_now=self.bat10_soc,
            bat10_soc_prev=self.bat10_soc_prev,
        )
        reward = -normalized_cost

        return cost, reward

    def control_step(self, net):
        # net.sgen.at[self.mgt5_id, 'p_mw'] = self.mgt5_p_mw
        # net.sgen.at[self.mgt9_id, 'p_mw'] = self.mgt9_p_mw
        # net.sgen.at[self.mgt10_id, 'p_mw'] = self.mgt10_p_mw
        net.storage.at[self.bat5_id, 'p_mw'] = self.bat5_p_mw
        net.storage.at[self.bat10_id, 'p_mw'] = self.bat10_p_mw
        self.applied = True

    def finalize_step(self, net, t):
        super().finalize_step(net, t)

        # update soc
        self.bat5_soc_prev = self.bat5_soc
        self.bat10_soc_prev = self.bat10_soc
        self.bat5_soc += (self.bat5_p_mw * HOUR_PER_TIME_STEP) / self.bat5_max_e_mwh
        self.bat10_soc += (self.bat10_p_mw * HOUR_PER_TIME_STEP) / self.bat10_max_e_mwh

        # calculate reward
        t += 1
        cost, reward = self.calculate_reward(net, t)
        self.costs.append(cost)
        self.rewards.append(reward)

    def time_step(self, net, t):
        # select action
        self.bat5_p_mw, self.bat10_p_mw = self.policy(net)

        self.soc_history['bat5_soc'].append(self.bat5_soc)
        self.soc_history['bat10_soc'].append(self.bat10_soc)

        self.applied = False
        self.last_time_step = t

    def policy(self, net):
        p_pv = net.sgen.at[self.ids.get('pv3'), 'p_mw'] +\
            net.sgen.at[self.ids.get('pv4'), 'p_mw'] +\
            net.sgen.at[self.ids.get('pv5'), 'p_mw'] +\
            net.sgen.at[self.ids.get('pv6'), 'p_mw'] +\
            net.sgen.at[self.ids.get('pv8'), 'p_mw'] +\
            net.sgen.at[self.ids.get('pv9'), 'p_mw'] +\
            net.sgen.at[self.ids.get('pv10'), 'p_mw'] +\
            net.sgen.at[self.ids.get('pv11'), 'p_mw']
        p_wt = net.sgen.at[self.ids.get('wt7'), 'p_mw']
        p_load = net.load.at[self.ids.get('load_r1'), 'p_mw'] +\
            net.load.at[self.ids.get('load_r3'), 'p_mw'] +\
            net.load.at[self.ids.get('load_r4'), 'p_mw'] +\
            net.load.at[self.ids.get('load_r5'), 'p_mw'] +\
            net.load.at[self.ids.get('load_r6'), 'p_mw'] +\
            net.load.at[self.ids.get('load_r8'), 'p_mw'] +\
            net.load.at[self.ids.get('load_r10'), 'p_mw'] +\
            net.load.at[self.ids.get('load_r11'), 'p_mw']
                            
        p_b5_max = min((SOC_MAX - self.bat5_soc) * self.bat5_max_e_mwh / HOUR_PER_TIME_STEP, P_B5_MAX)
        p_b5_min = max((SOC_MIN - self.bat5_soc) * self.bat5_max_e_mwh / HOUR_PER_TIME_STEP, P_B5_MIN)
        p_b10_max = min((SOC_MAX - self.bat10_soc) * self.bat10_max_e_mwh / HOUR_PER_TIME_STEP, P_B10_MAX)
        p_b10_min = max((SOC_MIN - self.bat10_soc) * self.bat10_max_e_mwh / HOUR_PER_TIME_STEP, P_B10_MIN)

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
        
        return p_b5, p_b10

    def reset(self):
        # self.mgt5_p_mw = 0.
        # self.mgt9_p_mw = 0.
        # self.mgt10_p_mw = 0.
        self.bat5_p_mw = 0.
        self.bat10_p_mw = 0.
        self.rewards = []
        self.costs = []
        self.bat5_soc = 0.
        self.bat10_soc = 0.
        self.soc_history = {'bat5_soc': [self.bat5_soc], 'bat10_soc': [self.bat10_soc]}
        self.last_time_step = None
        self.applied = False

class RandomControl(Controller):
    def __init__(self, net, ids, **kwargs):
        super().__init__(net, **kwargs)
        self.rewards = []
        self.costs = []
        self.bat5_soc = np.random.uniform(low=SOC_MIN, high=SOC_MAX)
        self.bat10_soc = np.random.uniform(low=SOC_MIN, high=SOC_MAX)
        self.last_time_step = None
        self.applied = False

        self.price_profile = kwargs['price_profile']

        self.trafo0_id = ids.get('trafo0')
        # self.mgt5_id = ids.get('mgt5')
        # self.mgt5_p_mw = net.sgen.at[self.mgt5_id, 'p_mw']
        # self.mgt9_id = ids.get('mgt9')
        # self.mgt9_p_mw = net.sgen.at[self.mgt9_id, 'p_mw']
        # self.mgt10_id = ids.get('mgt10')
        # self.mgt10_p_mw = net.sgen.at[self.mgt10_id, 'p_mw']

        self.bat5_id = ids.get('bat5')
        self.bat5_p_mw = net.storage.at[self.bat5_id, 'p_mw']
        self.bat5_max_e_mwh = net.storage.at[self.bat5_id, 'max_e_mwh']
        self.bat10_id = ids.get('bat10')
        self.bat10_p_mw = net.storage.at[self.bat10_id, 'p_mw']
        self.bat10_max_e_mwh = net.storage.at[self.bat10_id, 'max_e_mwh']

    def is_converged(self, net):
        return self.applied
    
    def control_step(self, net):
        # net.sgen.at[self.mgt5_id, 'p_mw'] = self.mgt5_p_mw
        # net.sgen.at[self.mgt9_id, 'p_mw'] = self.mgt9_p_mw
        # net.sgen.at[self.mgt10_id, 'p_mw'] = self.mgt10_p_mw
        net.storage.at[self.bat5_id, 'p_mw'] = self.bat5_p_mw
        net.storage.at[self.bat10_id, 'p_mw'] = self.bat10_p_mw
        self.applied = True
    
    def time_step(self, net, t):
        if self.last_time_step is not None:
            # update soc
            bat5_soc_prev = self.bat5_soc
            bat10_soc_prev = self.bat10_soc
            self.bat5_soc += (self.bat5_p_mw * HOUR_PER_TIME_STEP) / self.bat5_max_e_mwh
            self.bat10_soc += (self.bat10_p_mw * HOUR_PER_TIME_STEP) / self.bat10_max_e_mwh

            # calculate reward
            price = self.price_profile['price'][t]
            cost, normalized_cost = utils.cal_cost(
                price=price,
                pcc_p_mw=-net.res_trafo.at[self.trafo0_id, 'p_lv_mw'],
                # mgt5_p_mw=self.mgt5_p_mw,
                # mgt9_p_mw=self.mgt9_p_mw,
                # mgt10_p_mw=self.mgt10_p_mw,
                bat5_soc_now=self.bat5_soc,
                bat5_soc_prev=bat5_soc_prev,
                bat10_soc_now=self.bat10_soc,
                bat10_soc_prev=bat10_soc_prev
            )
            reward = -normalized_cost
            self.rewards.append(reward)
            self.costs.append(cost)
        
        # select action
        self.bat5_p_mw, self.bat10_p_mw = np.random.uniform(low=MIN_ACTION, high=MAX_ACTION, size=(N_ACTIONS,))
        p_b5_max = min((SOC_MAX - self.bat5_soc) * self.bat5_max_e_mwh / HOUR_PER_TIME_STEP, P_B5_MAX)
        p_b5_min = max((SOC_MIN - self.bat5_soc) * self.bat5_max_e_mwh / HOUR_PER_TIME_STEP, P_B5_MIN)
        p_b10_max = min((SOC_MAX - self.bat10_soc) * self.bat10_max_e_mwh / HOUR_PER_TIME_STEP, P_B10_MAX)
        p_b10_min = max((SOC_MIN - self.bat10_soc) * self.bat10_max_e_mwh / HOUR_PER_TIME_STEP, P_B10_MIN)
        self.bat5_p_mw = np.clip(self.bat5_p_mw, p_b5_min, p_b5_max)
        self.bat10_p_mw = np.clip(self.bat10_p_mw, p_b10_min, p_b10_max)
        self.applied = False
        self.last_time_step = t
    
    def reset(self):
        # self.mgt5_p_mw = 0.
        # self.mgt9_p_mw = 0.
        # self.mgt10_p_mw = 0.
        self.bat5_p_mw = 0.
        self.bat10_p_mw = 0.
        self.rewards = []
        self.costs = []
        self.bat5_soc = np.random.uniform(low=SOC_MIN, high=SOC_MAX)
        self.bat10_soc = np.random.uniform(low=SOC_MIN, high=SOC_MAX)
        self.last_time_step = None
        self.applied = False