import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
from pandapower.control.basic_controller import Controller

from controllers.buffer import Buffer
from controllers.models import ActorPiModel, CriticVModel, SequenceModel, get_pi_actor, get_v_critic
from setting import *
import utils

class PPOAgent(Controller):
    def __init__(self, net, ids, pv_profile, wt_profile, load_profile, price_profile, 
        sequence_model_type, training=False,
        lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, 
        **kwargs):
        super().__init__(net, **kwargs)

        self.ids = ids
        self.pv_profile = pv_profile
        self.wt_profile = wt_profile
        self.load_profile = load_profile
        self.price_profile = price_profile
        self.state_seq_shape = STATE_SEQ_SHAPE
        self.state_fnn_shape = STATE_FNN_SHAPE
        self.n_action = N_ACTION
        self.training = training
        self.time_step_counter = 0

        # normalization
        self.obs_norm = utils.NormalizeObservation()
        self.r_norm = utils.NormalizeReward()

        # action bounds
        self.max_action = MAX_ACTION
        self.min_action = MIN_ACTION

        # hyper parameters
        self.batch_size = PPO_BATCH_SIZE
        self.policy_clip = POLICY_CLIP
        self.target_kl = TARGET_KL
        self.train_freq = PPO_TRAIN_FREQ
        self.train_iters = PPO_TRAIN_ITERS

        # battery
        self.bat5_id = ids.get('bat5')
        self.bat5_p_mw = net.storage.at[self.bat5_id, 'p_mw']
        self.bat5_max_e_mwh = net.storage.at[self.bat5_id, 'max_e_mwh']
        self.bat10_id = ids.get('bat10')
        self.bat10_p_mw = net.storage.at[self.bat10_id, 'p_mw']
        self.bat10_max_e_mwh = net.storage.at[self.bat10_id, 'max_e_mwh']

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

        # internal states
        self.state = None
        self.state_value = None
        self.action_logprob = None
        self.bat5_soc = 0.
        self.bat10_soc = 0.
        self.rewards = []
        self.costs = []
        self.last_time_step = None
        self.applied = False

        self.history = {
            'price': [],
            'excess': [],
            'nn_bat5_p_mw': [],
            'nn_bat10_p_mw': [],
            'bat5_p_mw': [],
            'bat10_p_mw': [],
            'bat5_soc': [], 
            'bat10_soc': []}

        # buffer
        self.buffer = Buffer(self.train_freq, STATE_SEQ_SHAPE, STATE_FNN_SHAPE, N_ACTION)
        
        # networks
        self.sequence_model_type = sequence_model_type
        self.actor = get_pi_actor(SequenceModel(sequence_model_type, activation='tanh'), ActorPiModel(self.n_action))
        self.actor_dist = tfp.distributions.Normal
        self.actor.compile(optimizer=Adam(lr_actor))
        self.critic = get_v_critic(SequenceModel(sequence_model_type, activation='tanh'), CriticVModel())
        self.critic.compile(optimizer=Adam(lr_critic))

    @tf.function
    def cal_kl(self, state_seq_buffer, state_fnn_buffer, action_buffer, action_logprob_buffer):
        action_means, action_stds = self.actor([state_seq_buffer, state_fnn_buffer])
        action_dists = self.actor_dist(action_means, action_stds)
        action_logprobs = self.cal_logprob(action_dists, action_buffer)
        kl = tf.math.reduce_mean(action_logprob_buffer - action_logprobs)
        return kl

    @tf.function
    def cal_logprob(self, dist, a):
        a_logprob = dist.log_prob(a)
        a_logprob = tf.math.reduce_sum(a_logprob, axis=-1, keepdims=True)
        return a_logprob

    def cal_reward(self, net, t, bat5_soc_prev, bat10_soc_prev) -> float:
        price = self.price_profile['price'][t]
        cost, normalized_cost = utils.cal_cost(
            price=price,
            pcc_p_mw=-net.res_trafo.at[self.trafo0_id, 'p_lv_mw'],
            bat5_soc_now=self.bat5_soc,
            bat5_soc_prev=bat5_soc_prev,
            bat10_soc_now=self.bat10_soc,
            bat10_soc_prev=bat10_soc_prev
        )
        reward = -normalized_cost

        return cost, reward

    def control_step(self, net):
        net.storage.at[self.bat5_id, 'p_mw'] = self.bat5_p_mw
        net.storage.at[self.bat10_id, 'p_mw'] = self.bat10_p_mw
        self.applied = True

    def finalize_step(self, net, t):
        super().finalize_step(net, t)
        # update soc
        bat5_soc_prev = self.bat5_soc
        bat10_soc_prev = self.bat10_soc
        self.bat5_soc += self.bat5_p_mw * HOUR_PER_TIME_STEP / self.bat5_max_e_mwh
        self.bat10_soc += self.bat10_p_mw * HOUR_PER_TIME_STEP / self.bat10_max_e_mwh

        # reward
        cost, reward = self.cal_reward(net, t, bat5_soc_prev, bat10_soc_prev)
        self.rewards.append(reward)
        self.costs.append(cost)

        if not self.training:
            return

        # store transition
        state_seq, state_fnn = self.obs_norm.normalize(self.state)
        reward = self.r_norm.normalize(reward)
        self.buffer.store_transition(state_seq, state_fnn, self.action, reward, self.state_value, self.action_logprob)
        self.time_step_counter += 1

        if self.time_step_counter % self.train_freq != 0:
            return

        # finish trajectory
        state = self.get_state(net, t+1)
        state_seq, state_fnn = self.obs_norm.normalize(state)
        state_seq = tf.expand_dims(tf.convert_to_tensor(state_seq, dtype=tf.float32), axis=0) 
        state_fnn = tf.expand_dims(tf.convert_to_tensor(state_fnn, dtype=tf.float32), axis=0)
        last_value = self.critic([state_seq, state_fnn])
        last_value = tf.squeeze(last_value).numpy()
        self.buffer.finish_trajectory(last_value)

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
            state_seq[i, 17] = self.price_profile['price'][t + i]
            # state_seq[i, 0] = utils.get_excess(self.pv_profile, self.wt_profile, self.load_profile, t+i)
            # state_seq[i, 1] = self.price_profile['price'][t + i]

        state_fnn[0] = self.bat5_soc
        state_fnn[1] = self.bat10_soc

        return state_seq, state_fnn

    def is_converged(self, net) -> bool:
        return self.applied

    def learn(self):
        kl = 0.
        for _ in range(self.train_iters):
            state_seq_buffer, \
            state_fnn_buffer, \
            action_buffer, \
            action_logprob_buffer, \
            return_buffer, \
            advantage_buffer, \
            batches = self.buffer.sample(self.batch_size)

            # batch update
            for batch in batches:
                state_seq_batch = tf.convert_to_tensor(state_seq_buffer[batch], dtype=tf.float32)
                state_fnn_batch = tf.convert_to_tensor(state_fnn_buffer[batch], dtype=tf.float32)
                action_batch = tf.convert_to_tensor(action_buffer[batch], dtype=tf.float32)
                action_logprob_batch = tf.convert_to_tensor(action_logprob_buffer[batch], dtype=tf.float32)
                return_batch = tf.convert_to_tensor(return_buffer[batch], dtype=tf.float32)
                advantage_batch = tf.convert_to_tensor(advantage_buffer[batch], dtype=tf.float32)

                if kl < 1.5 * self.target_kl:
                    actor_loss = self.update_actor(state_seq_batch, state_fnn_batch, action_batch, action_logprob_batch, advantage_batch)
                critic_loss = self.update_critic(state_seq_batch, state_fnn_batch, return_batch)

            # kl divergence
            if kl < 1.5 * self.target_kl:
                state_seq_buffer = tf.convert_to_tensor(state_seq_buffer, dtype=tf.float32)
                state_fnn_buffer = tf.convert_to_tensor(state_fnn_buffer, dtype=tf.float32)
                action_buffer = tf.convert_to_tensor(action_buffer, dtype=tf.float32)
                action_logprob_buffer = tf.convert_to_tensor(action_logprob_buffer, dtype=tf.float32)
                kl = self.cal_kl(state_seq_buffer, state_fnn_buffer, action_buffer, action_logprob_buffer)

        utils.log_actor_critic_info(actor_loss, critic_loss)
        self.buffer.clear()

    def load(self, run=1):
        self.obs_norm.load(dir=os.path.join('.', 'rms', 'PPO', self.sequence_model_type, str(run)))
        self.load_models(dir=os.path.join('.', 'model_weights', 'PPO', self.sequence_model_type, str(run)))

    def load_models(self, dir):
        print('... Loading Models ...')
        self.actor.load_weights(os.path.join(dir, 'actor_weights'))
        self.critic.load_weights(os.path.join(dir, 'critic_weights'))
        
    def model_info(self):
        self.actor.summary()
        self.critic.summary()

    def policy(self, net, state):
        state_seq, state_fnn = self.obs_norm.normalize(state, update=False)
        state_seq = tf.expand_dims(tf.convert_to_tensor(state_seq, dtype=tf.float32), axis=0) 
        state_fnn = tf.expand_dims(tf.convert_to_tensor(state_fnn, dtype=tf.float32), axis=0)

        action_mean, action_std = self.actor([state_seq, state_fnn])
        action_std = action_std if self.training else 0.
        action_dist = self.actor_dist(action_mean, action_std)
        nn_action = action_dist.sample()
        nn_action = np.clip(nn_action, -NN_BOUND, NN_BOUND)
        action_logprob = self.cal_logprob(action_dist, nn_action)
        state_value = self.critic([state_seq, state_fnn])

        # remove batch dim
        nn_action = tf.squeeze(nn_action, axis=0).numpy()
        # reduce to single value
        action_logprob = tf.squeeze(action_logprob).numpy()
        state_value = tf.squeeze(state_value).numpy()

        # mg action
        p_b5_min = max(P_B5_MIN, (SOC_MIN - self.bat5_soc) * self.bat5_max_e_mwh / HOUR_PER_TIME_STEP)
        p_b5_max = min(P_B5_MAX, (SOC_MAX - self.bat5_soc) * self.bat5_max_e_mwh / HOUR_PER_TIME_STEP)
        p_b10_min = max(P_B10_MIN, (SOC_MIN - self.bat10_soc) * self.bat10_max_e_mwh / HOUR_PER_TIME_STEP)
        p_b10_max = min(P_B10_MAX, (SOC_MAX - self.bat10_soc) * self.bat10_max_e_mwh / HOUR_PER_TIME_STEP)

        # invalid action masking
        self.min_action[ACTION_IDX.get('p_b5')] = p_b5_min
        self.min_action[ACTION_IDX.get('p_b10')] = p_b10_min
        self.max_action[ACTION_IDX.get('p_b5')] = p_b5_max
        self.max_action[ACTION_IDX.get('p_b10')] = p_b10_max
        mg_action = utils.scale_to_mg(nn_action, self.min_action, self.max_action)

        return mg_action, nn_action, action_logprob, state_value

    def reset(self):
        # reset internal states
        self.state = None
        self.state_value = None
        self.action_logprob = None
        self.bat5_soc = 0.
        self.bat10_soc = 0.
        self.rewards = []
        self.costs = []
        self.last_time_step = None
        self.applied = False
        self.history = {
            'price': [],
            'excess': [],
            'nn_bat5_p_mw': [],
            'nn_bat10_p_mw': [],
            'bat5_p_mw': [],
            'bat10_p_mw': [],
            'bat5_soc': [], 
            'bat10_soc': []}

    def save(self, run=1):
        self.obs_norm.save(dir=os.path.join('.', 'rms', 'PPO', self.sequence_model_type, str(run)))
        self.save_models(dir=os.path.join('.', 'model_weights', 'PPO', self.sequence_model_type, str(run)))

    def save_models(self, dir):
        print('... Saving Models ...')
        self.actor.save_weights(os.path.join(dir, 'actor_weights'))
        self.critic.save_weights(os.path.join(dir, 'critic_weights'))

    @tf.function
    def update_actor(self, state_seq_batch, state_fnn_batch, action_batch, action_logprob_batch, advantage_batch):
        with tf.GradientTape() as tape:
            action_means, action_stds = self.actor([state_seq_batch, state_fnn_batch])
            action_dists = self.actor_dist(action_means, action_stds)
            action_logprobs = self.cal_logprob(action_dists, action_batch)
            prob_ratio = tf.math.exp(action_logprobs - action_logprob_batch)

            surrogate_obj = prob_ratio * advantage_batch
            clipped_surrogate_obj = tf.clip_by_value(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage_batch
            actor_loss = -tf.math.reduce_mean(
                tf.math.minimum(surrogate_obj, clipped_surrogate_obj)
            )
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        return actor_loss

    @tf.function
    def update_critic(self, state_seq_batch, state_fnn_batch, return_batch):
        huber_loss = keras.losses.Huber()
        with tf.GradientTape() as tape:
            critic_values = self.critic([state_seq_batch, state_fnn_batch])
            critic_loss = huber_loss(return_batch, critic_values)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return critic_loss

    @tf.function
    def update_sequence_model(self, state_seq_batch, state_fnn_batch, return_batch):
        huber_loss = keras.losses.Huber()
        with tf.GradientTape() as tape:
            critic_values = self.critic([state_seq_batch, state_fnn_batch])
            critic_loss = huber_loss(return_batch, critic_values)
        seq_grads = tape.gradient(critic_loss, self.sequence_model.trainable_variables)
        seq_grads = [tf.clip_by_norm(g, 1.0) for g in seq_grads]
        self.sequence_model.optimizer.apply_gradients(zip(seq_grads, self.sequence_model.trainable_variables))

    def time_step(self, net, t):
        # action selection
        self.state = self.get_state(net, t)
        mg_action, nn_action, self.action_logprob, self.state_value = self.policy(net, self.state)
        self.bat5_p_mw, self.bat10_p_mw = mg_action
        self.action = nn_action    
        # utils.log_trans_info(self.state, nn_action, t)

        # history
        self.history['price'].append(round(self.price_profile['price'][t], 3))
        excess = net.res_sgen['p_mw'].sum() - net.res_load['p_mw'].sum()
        self.history['excess'].append(round(excess, 3))
        self.history['nn_bat5_p_mw'].append(round(self.action[ACTION_IDX['p_b5']], 3))
        self.history['nn_bat10_p_mw'].append(round(self.action[ACTION_IDX['p_b10']], 3))
        self.history['bat5_p_mw'].append(round(self.bat5_p_mw, 3))
        self.history['bat10_p_mw'].append(round(self.bat10_p_mw, 3))
        self.history['bat5_soc'].append(round(self.bat5_soc, 3))
        self.history['bat10_soc'].append(round(self.bat10_soc, 3))
        
        self.applied = False
        self.last_time_step = t