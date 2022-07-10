'''
class:
    - TransformerEncoder
    - ActorModel
    - CriticModel
'''

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.layers import SpectralNormalization
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from setting import *

class TransformerEncoder(layers.Layer):
    def __init__(self, key_dim=64, num_heads=2, dense_dim=32, sequence_length=SEQ_LENGTH, **kwargs):
        super().__init__(**kwargs)
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dense_dim = dense_dim
        self.sequence_length = sequence_length

        self.dense_inputs = layers.Dense(key_dim)
        self.position_embeddings = layers.Embedding(sequence_length, key_dim)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation='relu'),
            layers.Dense(key_dim)
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        attention_input = self.dense_inputs(inputs) + self.position_embeddings(positions)
        attention_output = self.attention(attention_input, attention_input)
        proj_input = self.layernorm1(attention_input + attention_output)
        proj_output = self.dense_proj(proj_input)
        outputs = self.layernorm2(proj_input + proj_output)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'key_dim': self.key_dim,
            'num_heads': self.num_heads,
            'dense_dim': self.dense_dim,
            'sequence_length': self.sequence_length
        })
        return config

class SequenceModel(keras.Model):
    def __init__(self, sequence_model_type='rnn', activation='relu', **kwargs):
        super().__init__()
        self.dense_proj = get_dense_proj_seq(activation=activation)
        if sequence_model_type == 'conv1d':
            self.seq = get_conv1d_model()
        elif sequence_model_type == 'rnn':    
            self.seq = get_rnn_model()
        elif sequence_model_type == 'transformer':
            self.seq = get_transformer_model()
        else:
            self.seq = layers.Flatten()
    
    def call(self, inputs):
        state_seq = self.dense_proj(inputs)
        state_seq = self.seq(state_seq)
        # state_seq = self.seq(inputs)

        return state_seq

class ActorMuModel(keras.Model):
    def __init__(self, n_action, **kwargs):
        super().__init__()
        # self.dense_proj = get_dense_proj_fnn()
        self.concat = layers.Concatenate()
        self.fc = keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(64, activation='tanh', kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001)),
            layers.LayerNormalization(),
            layers.Dense(64, activation='tanh', kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001)),
            layers.LayerNormalization(),
        ]) 

        self.action = layers.Dense(n_action, activation='tanh', kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001))

    def call(self, state_seq, state_fnn):
        # state_fnn = self.dense_proj(state_fnn)
        state = self.concat([state_seq, state_fnn])
        state = self.fc(state)
        action = self.action(state)

        return action

class ActorPiModel(keras.Model):
    def __init__(self, n_action, logstd_init=0., **kwargs):
        super().__init__()
        # self.dense_proj = get_dense_proj_fnn(activation='tanh')
        self.concat = layers.Concatenate()
        self.fc = keras.Sequential([
            layers.Dense(64, activation='tanh', kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001)),
            layers.Dense(64, activation='tanh', kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001)),
        ]) 
        self.action_mean = layers.Dense(n_action, activation='tanh', kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001))
        
        self.action_logstd = tf.Variable(logstd_init * tf.ones(n_action), trainable=True)
    
    def call(self, state_seq, state_fnn):
        # state_fnn = self.dense_proj(state_fnn)
        state = self.concat([state_seq, state_fnn])
        state = self.fc(state)
        action_mean = self.action_mean(state)

        action_std = tf.math.exp(self.action_logstd)
        return action_mean, action_std

class CriticQModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        # self.dense_proj_fnn = get_dense_proj_fnn()
        # self.dense_proj_a = get_dense_proj_a()
        self.concat = layers.Concatenate()
        self.dense = keras.Sequential([
            layers.Dense(64, activation='tanh', kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001)),
            SpectralNormalization(layers.Dense(64, activation='tanh', kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001))),
        ])
        self.q = layers.Dense(1)

    def call(self, state_seq, state_fnn, action):
        # state_fnn = self.dense_proj_fnn(state_fnn)
        # action = self.dense_proj_a(action)
        state_action = self.concat([state_seq, state_fnn, action])
        state_action = self.dense(state_action)
        q_value = self.q(state_action)

        return q_value
    
class CriticVModel(keras.Model):
    def __init__(self, name='critic', **kwargs):
        super().__init__()
        # self.dense_proj_fnn = get_dense_proj_fnn(activation='tanh')
        self.concat = layers.Concatenate()
        self.dense = keras.Sequential([
            SpectralNormalization(layers.Dense(64, activation='tanh', kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001))),
            SpectralNormalization(layers.Dense(64, activation='tanh', kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001))),
        ])
        self.v = layers.Dense(1)
    
    def call(self, state_seq, state_fnn):
        # state_fnn = self.dense_proj_fnn(state_fnn)
        state = self.concat([state_seq, state_fnn])
        state = self.dense(state)
        state_value = self.v(state)

        return state_value

def get_dense_proj_a(proj_dim=DENSE_DIM_A):
    inputs = keras.Input(shape=(N_ACTION,))
    outputs = keras.Sequential([
        layers.Dense(proj_dim, activation='relu')
    ])(inputs)

    model = keras.Model(inputs, outputs, name='dense_proj_a')
    return model

def get_dense_proj_fnn(proj_dim=DENSE_DIM_FNN, activation='relu'):
    inputs = keras.Input(shape=STATE_FNN_SHAPE)
    outputs = layers.Dense(proj_dim, activation=activation, kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001))(inputs)

    model = keras.Model(inputs, outputs, name='dense_proj_fnn')
    return model

def get_dense_proj_seq(proj_dim=DENSE_DIM_SEQ, activation='tanh'):
    inputs = keras.Input(shape=STATE_SEQ_SHAPE)
    outputs = layers.Dense(proj_dim, activation=activation, kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001))(inputs)

    model = keras.Model(inputs, outputs, name='dense_proj_seq')
    return model

def get_conv1d_model():
    inputs = keras.Input(shape=(SEQ_LENGTH, DENSE_DIM_SEQ))
    outputs = keras.Sequential([
        SpectralNormalization(layers.Conv1D(8, 5, activation='tanh')),
        layers.MaxPooling1D(2),
        layers.GRU(16)
        # layers.Conv1D(8, 3, activation='tanh'),
        # layers.GlobalMaxPooling1D(),
    ])(inputs)

    model = keras.Model(inputs, outputs, name='sequence_model')
    return model

def get_rnn_model():
    # inputs = keras.Input(shape=(SEQ_LENGTH, DENSE_DIM_SEQ))
    # lstm_in = layers.LayerNormalization()(inputs)
    # lstm_out = layers.LSTM(DENSE_DIM_SEQ, return_sequences=True)(lstm_in)
    # outputs = layers.Add()([inputs, lstm_out]) # residual connection
    # outputs = layers.LayerNormalization()(outputs)
    # outputs = layers.LSTM(16)(outputs)
    inputs = keras.Input(shape=(SEQ_LENGTH, DENSE_DIM_SEQ))
    outputs = layers.GRU(32)(inputs)

    model = keras.Model(inputs, outputs, name='sequence_model')
    return model

# TODO
def get_transformer_model():
    pass

# deterministic actor
def get_mu_actor(sequence_model, actor_model):
    input_seq = keras.Input(shape=STATE_SEQ_SHAPE)
    state_seq = sequence_model(input_seq)
    input_fnn = keras.Input(shape=STATE_FNN_SHAPE)
    state_fnn = get_dense_proj_fnn()(input_fnn)
    action = actor_model(state_seq, state_fnn)

    actor = keras.Model([input_seq, input_fnn], action)
    return actor

# stochastic actor
def get_pi_actor(sequence_model, actor_model):
    input_seq = keras.Input(shape=STATE_SEQ_SHAPE)
    state_seq = sequence_model(input_seq)
    state_fnn = keras.Input(shape=STATE_FNN_SHAPE)
    action_mean, action_std = actor_model(state_seq, state_fnn)

    actor = keras.Model([input_seq, state_fnn], [action_mean, action_std])
    return actor

# q value critic
def get_q_critic(sequence_model, critic_model):
    input_seq = keras.Input(shape=STATE_SEQ_SHAPE)
    state_seq = sequence_model(input_seq)
    state_fnn = keras.Input(shape=STATE_FNN_SHAPE)
    action = keras.Input(shape=(N_ACTION,))
    q_value = critic_model(state_seq, state_fnn, action)

    critic = keras.Model([input_seq, state_fnn, action], q_value)
    return critic

# state value critic
def get_v_critic(sequence_model, critic_model):
    input_seq = keras.Input(shape=STATE_SEQ_SHAPE)
    state_seq = sequence_model(input_seq)
    state_fnn = keras.Input(shape=STATE_FNN_SHAPE)
    state_value = critic_model(state_seq, state_fnn)

    critic = keras.Model([input_seq, state_fnn], state_value)
    return critic