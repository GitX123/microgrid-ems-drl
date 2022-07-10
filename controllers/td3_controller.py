'''
class:
- TD3Agent
    - adapt_param_noise()
    - adjust_action_noise()
    - calculate_distance()
    - calculate_reward()
    - control_step()
    - get_state()
    - is_converged()
    - learn()
    - load_models()
    - model_info()
    - perturb_policy()
    - policy()
    - reset()
    - save_models()
    - update_actor()
    - update_critics()
    - update_sequence_model()
    - update_target_networks()
    - time_step()
'''

import logging
import os
from typing import Dict
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

from pandapower.control.basic_controller import Controller
from controllers.models import ActorMuModel, CriticQModel, SequenceModel, get_mu_actor, get_q_critic
from controllers.buffer import ReplayBuffer, PrioritizedReplayBuffer
from setting import *
import utils

class TD3Agent(Controller):
    def __init__(self, net, ids, pv_profile, wt_profile, load_profile, price_profile,
        noise_type = 'action', sequence_model_type='none', use_pretrained_sequence_model=False,
        n_epochs=None, training=False,
        delay=2, gamma=GAMMA, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, 
        buffer_size=50000, batch_size=128, epsilon_p=0.001, **kwargs):
        super().__init__(net, **kwargs)
        self.ids = ids
        self.pv_profile = pv_profile
        self.wt_profile = wt_profile
        self.load_profile = load_profile
        self.price_profile = price_profile
        self.state_seq_shape = STATE_SEQ_SHAPE
        self.state_fnn_shape = STATE_FNN_SHAPE
        self.n_action = N_ACTION
        self.use_pretrained_sequence_model = use_pretrained_sequence_model
        self.training = training
        self.noise_type = noise_type
        self.action_noise_scale = ACTION_NOISE_SCALE
        self.action_noise_scale_ = ACTION_NOISE_SCALE
        self.param_noise_adapt_rate = PARAM_NOISE_ADAPT_RATE
        self.param_noise_bound = PARAM_NOISE_BOUND
        self.param_noise_scale = PARAM_NOISE_SCALE
        self.n_epochs = n_epochs
        self.update_freq = UPDATE_FREQ
        self.update_times = UPDATE_TIMES
        self.warmup = WARMUP
        self.delay = delay
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_p = epsilon_p

        # counter
        self.time_step_counter = 0
        self.learn_step_counter = 0

        # normalization
        self.obs_norm = utils.NormalizeObservation()
        self.a_norm = utils.NormalizeAction()
        self.r_norm = utils.NormalizeReward()

        # action bounds
        self.max_action = MAX_ACTION
        self.min_action = MIN_ACTION

        # generator
        # self.mgt5_id = ids.get('mgt5')
        # self.mgt5_p_mw = net.sgen.at[self.mgt5_id, 'p_mw']
        # self.mgt9_id = ids.get('mgt9')
        # self.mgt9_p_mw = net.sgen.at[self.mgt9_id, 'p_mw']
        # self.mgt10_id = ids.get('mgt10')
        # self.mgt10_p_mw = net.sgen.at[self.mgt10_id, 'p_mw']

        # battery
        self.bat5_id = ids.get('bat5')
        self.bat5_p_mw = net.storage.at[self.bat5_id, 'p_mw']
        self.bat5_max_e_mwh = net.storage.at[self.bat5_id, 'max_e_mwh']
        self.bat10_id = ids.get('bat10')
        self.bat10_p_mw = net.storage.at[self.bat10_id, 'p_mw']
        self.bat10_max_e_mwh = net.storage.at[self.bat10_id, 'max_e_mwh']

        # internal states
        self.prev_state = None
        self.bat5_soc = 0.
        self.bat10_soc = 0.
        self.action = None
        self.rewards = []
        self.costs = []
        self.history = {
            'price': [],
            # 'mgt5_p_mw': [round(self.mgt5_p_mw, 3)],
            # 'mgt9_p_mw': [round(self.mgt9_p_mw, 3)],
            # 'mgt10_p_mw': [round(self.mgt10_p_mw, 3)],
            'excess': [],
            'nn_bat5_p_mw': [],
            'nn_bat10_p_mw': [],
            'bat5_p_mw': [],
            'bat10_p_mw': [],
            'bat5_soc': [], 
            'bat10_soc': []}
        self.last_time_step = None
        self.applied = False

        # other elements
        self.pv3_id = ids.get('pv3')
        self.pv4_id = ids.get('pv4')
        self.pv5_id = ids.get('pv5')
        self.pv6_id = ids.get('pv6')
        self.pv8_id = ids.get('pv8')
        self.pv9_id = ids.get('pv9')
        self.loadr1_id = ids.get('load_r1')
        self.loadr3_id = ids.get('Load_r3')
        self.loadr4_id = ids.get('Load_r4')
        self.loadr5_id = ids.get('Load_r5')
        self.loadr6_id = ids.get('Load_r6')
        self.loadr8_id = ids.get('Load_r8')
        self.loadr10_id = ids.get('Load_r10')
        self.loadr11_id = ids.get('Load_r11')
        self.trafo0_id = ids.get('trafo0') # PCC trafo

        # buffer
        # self.buffer = ReplayBuffer(buffer_size, self.state_seq_shape, self.state_fnn_shape, self.n_action)
        self.buffer = PrioritizedReplayBuffer(buffer_size, self.state_seq_shape, self.state_fnn_shape, self.n_action)

        # models
        self.sequence_model_type = sequence_model_type
        if self.sequence_model_type == 'none':
            # actor critic
            self.actor = get_mu_actor(SequenceModel(sequence_model_type, name='sequence_model'), ActorMuModel(self.n_action))
            self.perturbed_actor = get_mu_actor(SequenceModel(sequence_model_type, name='sequence_model'), ActorMuModel(self.n_action))
            self.target_actor = get_mu_actor(SequenceModel(sequence_model_type, name='sequence_model'), ActorMuModel(self.n_action))
            self.critic1 = get_q_critic(SequenceModel(sequence_model_type, name='sequence_model'), CriticQModel())
            self.critic2 = get_q_critic(SequenceModel(sequence_model_type, name='sequence_model'), CriticQModel())
            self.target_critic1 = get_q_critic(SequenceModel(sequence_model_type, name='sequence_model'), CriticQModel())
            self.target_critic2 = get_q_critic(SequenceModel(sequence_model_type, name='sequence_model'), CriticQModel())
        else:
            # sequence model
            self.sequence_model = SequenceModel(sequence_model_type, name='sequence_model')
            self.sequence_model.compile(optimizer=Adam(learning_rate=lr_critic, epsilon=1e-5))

            # actor critic
            self.actor = get_mu_actor(self.sequence_model, ActorMuModel(self.n_action))
            self.perturbed_actor = get_mu_actor(self.sequence_model, ActorMuModel(self.n_action))
            self.target_actor = get_mu_actor(self.sequence_model, ActorMuModel(self.n_action))
            self.critic1 = get_q_critic(self.sequence_model, CriticQModel())
            self.critic2 = get_q_critic(self.sequence_model, CriticQModel())
            self.target_critic1 = get_q_critic(self.sequence_model, CriticQModel())
            self.target_critic2 = get_q_critic(self.sequence_model, CriticQModel())

        self.actor.compile(optimizer=Adam(learning_rate=lr_actor, epsilon=1e-5))
        self.critic1.compile(optimizer=Adam(learning_rate=lr_critic, epsilon=1e-5))
        self.critic2.compile(optimizer=Adam(learning_rate=lr_critic, epsilon=1e-5))

        # initialization
        if self.training:
            if self.use_pretrained_sequence_model:
                file_path = os.path.join('.', 'pretrained_sequence_model', self.sequence_model_type, 'pretrained_sequence_model_weights.hdf5')
                self.sequence_model.load_weights(file_path, by_name=True)
            self.perturbed_actor.set_weights(self.actor.get_weights())
            self.update_target_networks(tau=1)

    def adapt_param_noise(self, d):
        if d <= self.param_noise_bound:
            self.param_noise_scale *= self.param_noise_adapt_rate
        else:
            self.param_noise_scale /= self.param_noise_adapt_rate
    
    def adjust_action_noise(self, adjust=True):
        if not adjust:
            return
        else:
            self.action_noise_scale -= self.action_noise_scale_ / self.n_epochs
            self.action_noise_scale = max(self.action_noise_scale, 0.1)

    # distance for parameter noise
    @tf.function
    def calculate_distance(self, state_seq_batch, state_fnn_batch):
        actions = self.actor([state_seq_batch, state_fnn_batch])
        perturbed_actions = self.perturbed_actor([state_seq_batch, state_fnn_batch])
        d = tf.math.square(actions - perturbed_actions)
        d = tf.math.reduce_mean(d)
        d = tf.math.sqrt(d)
        return d
    
    def calculate_reward(self, net, t):
        price = self.price_profile['price'][t - 1]
        cost, normalized_cost = utils.cal_cost(
            price=price,
            pcc_p_mw=-net.res_trafo.at[self.trafo0_id, 'p_lv_mw'],
            # mgt5_p_mw=self.mgt5_p_mw,
            # mgt9_p_mw=self.mgt9_p_mw,
            # mgt10_p_mw=self.mgt10_p_mw,
            bat5_soc_now=self.bat5_soc,
            bat5_soc_prev=self.prev_state[1][0],
            bat10_soc_now=self.bat10_soc,
            bat10_soc_prev=self.prev_state[1][1],
            # ids=self.ids,
            # t=t,
            # net=net
            )
        reward = -normalized_cost

        # invalid action penalty
        # nn_bat_p_mw = self.action * np.array([P_B5_MAX, P_B10_MAX])
        # valid_bat_p_mw = np.array([self.bat5_p_mw, self.bat10_p_mw])
        # extra_reward = utils.extra_reward(nn_bat_p_mw, valid_bat_p_mw)
        # reward += extra_reward

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

        # next time step
        t += 1

        # update soc
        self.bat5_soc += self.bat5_p_mw * HOUR_PER_TIME_STEP / self.bat5_max_e_mwh
        self.bat10_soc += self.bat10_p_mw * HOUR_PER_TIME_STEP / self.bat10_max_e_mwh

        # observe transition
        state = self.get_state(net, t)
        cost, reward = self.calculate_reward(net, t)
        self.rewards.append(reward)
        self.costs.append(cost)

        if self.training:
            # store transition
            normalized_prev_state = self.obs_norm.normalize(self.prev_state)
            normalized_state = self.obs_norm.normalize(state)
            normalized_action = self.a_norm.normalize(self.action)
            normalized_reward = self.r_norm.normalize(reward)
            self.buffer.store_transition(normalized_prev_state[0], normalized_prev_state[1], normalized_action, normalized_reward, normalized_state[0], normalized_state[1])

            # update networks
            self.learn()

    def get_state(self, net, t):
        state_seq = np.zeros(self.state_seq_shape)
        state_fnn = np.zeros(self.state_fnn_shape)

        for i in range(SEQ_LENGTH):
            state_seq[i, 0] = self.pv_profile['pv3'][t + i]
            state_seq[i, 1] = self.pv_profile['pv4'][t + i]
            state_seq[i, 2] = self.pv_profile['pv5'][t + i]
            state_seq[i, 3] = self.pv_profile['pv6'][t + i]
            state_seq[i, 4] = self.pv_profile['pv8'][t + i]
            state_seq[i, 5] = self.pv_profile['pv9'][t + i]
            state_seq[i, 6] = self.pv_profile['pv10'][t + i]
            state_seq[i, 7] = self.pv_profile['pv11'][t + i]
            state_seq[i, 8] = self.wt_profile['wt7'][t + i]
            state_seq[i, 9] = self.load_profile['load_r1'][t + i]
            state_seq[i, 10] = self.load_profile['load_r3'][t + i]
            state_seq[i, 11] = self.load_profile['load_r4'][t + i]
            state_seq[i, 12] = self.load_profile['load_r5'][t + i]
            state_seq[i, 13] = self.load_profile['load_r6'][t + i]
            state_seq[i, 14] = self.load_profile['load_r8'][t + i]
            state_seq[i, 15] = self.load_profile['load_r10'][t + i]
            state_seq[i, 16] = self.load_profile['load_r11'][t + i]
            # state_seq[i, 17] = utils.get_excess(self.pv_profile, self.wt_profile, self.load_profile, t+i)
            state_seq[i, 17] = self.price_profile['price'][t + i]

            # state_seq[i, 0] = utils.get_excess(self.pv_profile, self.wt_profile, self.load_profile, t+i)
            # state_seq[i, 1] = self.price_profile['price'][t + i]
        
        state_fnn[0] = self.bat5_soc
        state_fnn[1] = self.bat10_soc

        return state_seq, state_fnn

    def is_converged(self, net) -> bool:
        return self.applied

    def learn(self):
        if self.buffer.buffer_counter < self.batch_size:
            return

        if self.buffer.buffer_counter < self.warmup:
            return

        if self.time_step_counter % self.update_freq != 0:
            return

        for _ in range(self.update_times):
            self.update()

    def load_models(self, dir='model_weights', run=1):
        print('... Loading Models ...')
        self.actor.load_weights(os.path.join(dir, self.sequence_model_type, str(run), 'actor_weights'))
        self.critic1.load_weights(os.path.join(dir, self.sequence_model_type, str(run), 'critic1_weights'))
        self.critic2.load_weights(os.path.join(dir, self.sequence_model_type, str(run), 'critic2_weights'))
        self.target_actor.load_weights(os.path.join(dir, self.sequence_model_type, str(run), 'target_actor_weights'))
        self.target_critic1.load_weights(os.path.join(dir, self.sequence_model_type, str(run), 'target_critic1_weights'))
        self.target_critic2.load_weights(os.path.join(dir, self.sequence_model_type, str(run), 'target_critic2_weights'))

    def model_info(self):
        self.actor.summary()
        self.critic1.summary()
    
    @tf.function
    def perturb_policy(self):
        # if self.sequence_model_type == 'none':
        #     perturbed_actor_weights = self.perturbed_actor.trainable_weights
        #     actor_weights = self.actor.trainable_weights   
        # else:
        #     perturbed_actor_weights = self.perturbed_actor.get_layer('actor_mu_model_1').trainable_weights
        #     actor_weights = self.actor.get_layer('actor_mu_model').trainable_weights
        perturbed_actor_weights = self.perturbed_actor.get_layer('actor_mu_model_1').trainable_weights
        actor_weights = self.actor.get_layer('actor_mu_model').trainable_weights

        for (perturbed_weights, weights) in zip(perturbed_actor_weights, actor_weights):
            perturbed_weights.assign(weights + tf.random.normal(shape=tf.shape(weights), mean=0., stddev=self.param_noise_scale))
    
    def policy(self, net, t, state):
        # network outputs
        if self.time_step_counter < self.warmup and self.training:
            # warmup
            nn_action = np.random.uniform(low=-NN_BOUND, high=NN_BOUND, size=(self.n_action,))
        else:
            state_seq, state_fnn = self.obs_norm.normalize(state, update=False)
            # add batch index
            tf_state_seq = tf.expand_dims(tf.convert_to_tensor(state_seq, dtype=tf.float32), axis=0) 
            tf_state_fnn = tf.expand_dims(tf.convert_to_tensor(state_fnn, dtype=tf.float32), axis=0)

            if self.training:
                # param noise
                if self.noise_type == 'param':
                    tf_action = self.perturbed_actor([tf_state_seq, tf_state_fnn], training=self.training)
                    tf_action = tf.squeeze(tf_action, axis=0) # remove batch index
                    nn_action = tf_action.numpy()
                # action noise
                else:
                    tf_action = self.actor([tf_state_seq, tf_state_fnn], training=self.training)
                    tf_action = tf.squeeze(tf_action, axis=0) # remove batch index
                    nn_action = tf_action.numpy()
                    if t % 100 == 0:
                        print(f'nn outputs = {nn_action}')
                    nn_action += np.random.normal(loc=0., scale=self.action_noise_scale, size=(self.n_action,))
                # testing
            else:
                tf_action = self.actor([tf_state_seq, tf_state_fnn], training=self.training)
                tf_action = tf.squeeze(tf_action, axis=0) # remove batch index
                nn_action = tf_action.numpy()
            nn_action = np.clip(nn_action, -NN_BOUND, NN_BOUND)

        # mg action
        p_b5_min = max(P_B5_MIN, (SOC_MIN - self.bat5_soc) * self.bat5_max_e_mwh / HOUR_PER_TIME_STEP)
        p_b5_max = min(P_B5_MAX, (SOC_MAX - self.bat5_soc) * self.bat5_max_e_mwh / HOUR_PER_TIME_STEP)
        p_b10_min = max(P_B10_MIN, (SOC_MIN - self.bat10_soc) * self.bat10_max_e_mwh / HOUR_PER_TIME_STEP)
        p_b10_max = min(P_B10_MAX, (SOC_MAX - self.bat10_soc) * self.bat10_max_e_mwh / HOUR_PER_TIME_STEP)

        # invalid action clipping
        # mgt5_p_mw, mgt9_p_mw, mgt10_p_mw, bat5_p_mw, bat10_p_mw = utils.scale_to_mg(nn_action, self.max_action, self.min_action)
        # bat5_p_mw = np.clip(bat5_p_mw, p_b5_min, p_b5_max)
        # bat10_p_mw = np.clip(bat10_p_mw, p_b10_min, p_b10_max)

        # invalid action masking
        self.min_action[ACTION_IDX.get('p_b5')] = p_b5_min
        self.min_action[ACTION_IDX.get('p_b10')] = p_b10_min
        self.max_action[ACTION_IDX.get('p_b5')] = p_b5_max
        self.max_action[ACTION_IDX.get('p_b10')] = p_b10_max
        bat5_p_mw, bat10_p_mw = utils.scale_to_mg(nn_action, self.min_action, self.max_action)

        mg_action = np.array([bat5_p_mw, bat10_p_mw])
        self.time_step_counter += 1
        return mg_action, nn_action
    
    def reset(self):
        # init states
        # self.mgt5_p_mw = 0.
        # self.mgt9_p_mw = 0.
        # self.mgt10_p_mw = 0.
        self.bat5_p_mw = 0.
        self.bat10_p_mw = 0.
        self.prev_state = None
        self.bat5_soc = 0.
        self.bat10_soc = 0.
        self.action = None
        self.rewards = []
        self.costs = []
        self.history = {
            'price': [],
            # 'mgt5_p_mw': [round(self.mgt5_p_mw, 3)],
            # 'mgt9_p_mw': [round(self.mgt9_p_mw, 3)],
            # 'mgt10_p_mw': [round(self.mgt10_p_mw, 3)],
            'excess': [],
            'nn_bat5_p_mw': [],
            'nn_bat10_p_mw': [],
            'bat5_p_mw': [],
            'bat10_p_mw': [],
            'bat5_soc': [], 
            'bat10_soc': []}
        self.last_time_step = None
        self.applied = False

        if not self.training:
            return

        # parameter scheduling
        if self.noise_type == 'action':
            self.adjust_action_noise()  
        self.buffer.schedule_beta(beta_inc=0.01)

    def save_models(self, dir='model_weights', run=1):
        print('... Saving Models ...')
        self.actor.save_weights(os.path.join(dir, self.sequence_model_type, str(run), 'actor_weights'))
        self.critic1.save_weights(os.path.join(dir, self.sequence_model_type, str(run), 'critic1_weights'))
        self.critic2.save_weights(os.path.join(dir, self.sequence_model_type, str(run), 'critic2_weights'))
        self.target_actor.save_weights(os.path.join(dir, self.sequence_model_type, str(run), 'target_actor_weights'))
        self.target_critic1.save_weights(os.path.join(dir, self.sequence_model_type, str(run), 'target_critic1_weights'))
        self.target_critic2.save_weights(os.path.join(dir, self.sequence_model_type, str(run), 'target_critic2_weights'))

    def update(self):
        # sample
        state_seq_batch, state_fnn_batch, action_batch, reward_batch, next_state_seq_batch, next_state_fnn_batch, idxs, weights = self.buffer.sample(self.batch_size)
        state_seq_batch = tf.convert_to_tensor(state_seq_batch, dtype=tf.float32)
        state_fnn_batch = tf.convert_to_tensor(state_fnn_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_seq_batch = tf.convert_to_tensor(next_state_seq_batch, dtype=tf.float32)
        next_state_fnn_batch = tf.convert_to_tensor(next_state_fnn_batch, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)

        # update critics
        critic_loss, target_values, td_errs = self.update_critics(state_seq_batch, state_fnn_batch, action_batch, reward_batch, next_state_seq_batch, next_state_fnn_batch, weights)
        self.update_priority(td_errs, idxs)
        self.learn_step_counter += 1

        # update sequence model
        if (self.sequence_model_type != 'none') and (not self.use_pretrained_sequence_model):
            self.update_sequence_model(state_seq_batch, state_fnn_batch, action_batch, target_values)

        if self.learn_step_counter % self.delay != 0:
            return

        # update actor
        actor_loss = self.update_actor(state_seq_batch, state_fnn_batch)

        # parameter noise
        if self.noise_type == 'param':
            self.perturb_policy()
            d = self.calculate_distance(state_seq_batch, state_fnn_batch)
            self.adapt_param_noise(d)

        # update targets
        self.update_target_networks()

    @tf.function
    def update_actor(self, state_seq_batch, state_fnn_batch):
        # trainable variables
        if self.sequence_model_type == 'none':
            actor_vars = self.actor.trainable_variables
        else:
            actor_vars = self.actor.get_layer('actor_mu_model').trainable_variables

        # gradient descent
        with tf.GradientTape() as tape:
            actions = self.actor([state_seq_batch, state_fnn_batch], training=True)
            actions = self.a_norm.tf_normalize(actions)
            q_values = self.critic1([state_seq_batch, state_fnn_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(q_values)
        actor_grads = tape.gradient(actor_loss, actor_vars)
        self.actor.optimizer.apply_gradients(zip(actor_grads, actor_vars))

        return actor_loss
    
    @tf.function
    def update_critics(self, state_seq_batch, state_fnn_batch, action_batch, reward_batch, next_state_seq_batch, next_state_fnn_batch, weights):
        # Issue: https://github.com/tensorflow/tensorflow/issues/35928
        # with tf.GradientTape(persistent=True) as tape:

        # target actions
        target_actions = self.target_actor([next_state_seq_batch, next_state_fnn_batch], training=True)
        target_actions += tf.clip_by_value(tf.random.normal(shape=(self.batch_size, self.n_action), stddev=0.2), -0.5, 0.5)
        target_actions = tf.clip_by_value(target_actions, -NN_BOUND, NN_BOUND)
        target_actions = self.a_norm.tf_normalize(target_actions)

        # target values
        target_q_value1 = self.target_critic1([next_state_seq_batch, next_state_fnn_batch, target_actions], training=True)
        target_q_value2 = self.target_critic2([next_state_seq_batch, next_state_fnn_batch, target_actions], training=True)
        target_values = reward_batch + self.gamma * tf.math.minimum(target_q_value1, target_q_value2)

        # td errors
        td_errs = target_values - self.critic1([state_seq_batch, state_fnn_batch, action_batch])

        # trainable variables
        if self.sequence_model_type == 'none':
            critic1_vars = self.critic1.trainable_variables
            critic2_vars = self.critic2.trainable_variables
        else:
            critic1_vars = self.critic1.get_layer('critic_q_model').trainable_variables
            critic2_vars = self.critic2.get_layer('critic_q_model_1').trainable_variables

        huber_loss = keras.losses.Huber()
        # update critic model 1
        with tf.GradientTape() as tape1:
            critic_loss1 = huber_loss(weights*target_values, weights*self.critic1([state_seq_batch, state_fnn_batch, action_batch], training=True))
        critic_grads1 = tape1.gradient(critic_loss1, critic1_vars)
        self.critic1.optimizer.apply_gradients(zip(critic_grads1, critic1_vars))

        # update critic model 2
        with tf.GradientTape() as tape2:
            critic_loss2 = huber_loss(weights*target_values, weights*self.critic2([state_seq_batch, state_fnn_batch, action_batch], training=True))
        critic_grads2 = tape2.gradient(critic_loss2, critic2_vars)
        self.critic2.optimizer.apply_gradients(zip(critic_grads2, critic2_vars))

        return critic_loss1, target_values, td_errs
    
    def update_priority(self, td_errs, idxs):
        priorities = np.abs(td_errs.numpy().flatten()) + self.epsilon_p
        for idx, p in zip(idxs, priorities):
            self.buffer.update_tree(idx, p)

    @tf.function
    def update_sequence_model(self, state_seq_batch, state_fnn_batch, action_batch, target_values):
        huber_loss = keras.losses.Huber()
        with tf.GradientTape() as tape:
            critic_loss = huber_loss(target_values, self.critic1([state_seq_batch, state_fnn_batch, action_batch], training=True)) 
            critic_loss += huber_loss(target_values, self.critic2([state_seq_batch, state_fnn_batch, action_batch], training=True))
            critic_loss /= (2 * SEQ_LENGTH)
        seq_grads = tape.gradient(critic_loss, self.sequence_model.trainable_variables)
        seq_grads = [tf.clip_by_norm(g, 1.0) for g in seq_grads]
        self.sequence_model.optimizer.apply_gradients(zip(seq_grads, self.sequence_model.trainable_variables))
    
    @tf.function
    def update_target_networks(self, tau=0.005):
        if self.sequence_model_type == 'none':
            target_actor_weights = self.target_actor.trainable_weights
            actor_weights = self.actor.trainable_weights
            target_critic1_weights = self.target_critic1.trainable_weights
            critic1_weights = self.critic1.trainable_weights
            target_critic2_weights = self.target_critic2.trainable_weights
            critic2_weights = self.critic2.trainable_weights
        else:
            target_actor_weights = self.target_actor.get_layer('actor_mu_model_2').trainable_weights
            actor_weights = self.actor.get_layer('actor_mu_model').trainable_weights
            target_critic1_weights = self.target_critic1.get_layer('critic_q_model_2').trainable_weights
            critic1_weights = self.critic1.get_layer('critic_q_model').trainable_weights
            target_critic2_weights = self.target_critic2.get_layer('critic_q_model_3').trainable_weights
            critic2_weights = self.critic2.get_layer('critic_q_model_1').trainable_weights

        # update target actor
        for target_weight, weight in zip(target_actor_weights, actor_weights):
            target_weight.assign(tau * weight + (1 - tau) * target_weight)

        # update target critic1
        for target_weight, weight in zip(target_critic1_weights, critic1_weights):
            target_weight.assign(tau * weight + (1 - tau) * target_weight)

        # update target critic2
        for target_weight, weight in zip(target_critic2_weights, critic2_weights):
            target_weight.assign(tau * weight + (1 - tau) * target_weight)

    def time_step(self, net, t):
        # action
        state = self.get_state(net, t)
        mg_action, nn_action = self.policy(net, t, state)
        self.bat5_p_mw, self.bat10_p_mw = mg_action
        self.action = nn_action
        self.prev_state = state
        if self.training:
            utils.log_trans_info(state, mg_action, t+5, freq=50)
        
        # history
        self.history['price'].append(round(self.price_profile['price'][t], 3))
        excess = net.res_sgen['p_mw'].sum() - net.res_load['p_mw'].sum()
        self.history['excess'].append(round(excess, 3))
        # self.history['mgt5_p_mw'].append(round(self.mgt5_p_mw, 5))
        # self.history['mgt9_p_mw'].append(round(self.mgt9_p_mw, 5))
        # self.history['mgt10_p_mw'].append(round(self.mgt10_p_mw, 5))
        self.history['nn_bat5_p_mw'].append(round(self.action[ACTION_IDX['p_b5']], 3))
        self.history['nn_bat10_p_mw'].append(round(self.action[ACTION_IDX['p_b10']], 3))
        self.history['bat5_p_mw'].append(round(self.bat5_p_mw, 3))
        self.history['bat10_p_mw'].append(round(self.bat10_p_mw, 3))
        self.history['bat5_soc'].append(round(self.bat5_soc, 3))
        self.history['bat10_soc'].append(round(self.bat10_soc, 3))

        self.applied = False
        self.last_time_step = t