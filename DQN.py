"""
DDQN agent
"""
import numpy as np
from collections import deque
from tensorflow.keras.layers import Dense, Flatten, Concatenate
import tensorflow as tf
import random


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
    #
    def _build_model(self):
        inputA = tf.keras.Input(shape= self.state_size[0], name="InputA")
        inputB = tf.keras.Input(shape= self.state_size[1], name="InputB")
        x_A = Flatten()(inputA)
        x_con = Dense(128, activation='linear')(x_A)
        x_temp_A = tf.keras.activations.swish(x_con)
        for iRes in range(3):
            x_con = Dense(128, activation='linear')(x_temp_A)
            x_con = tf.keras.activations.swish(x_con)
            x_con = Dense(128, activation='linear')(x_con)
            x_temp_A = tf.keras.layers.Add()([x_temp_A, x_con])
        x_A = Dense(128, activation='linear')(x_temp_A)
        x_A = tf.keras.activations.swish(x_A)
        ModelX = tf.keras.Model(inputs=inputA, outputs=x_A)
        x_B = Flatten()(inputB)
        x_con = Dense(128, activation='linear')(x_B)
        x_temp_B = tf.keras.activations.swish(x_con)
        for iRes in range(3):
            x_con = Dense(128, activation='linear')(x_temp_B)
            x_con = tf.keras.activations.swish(x_con)
            x_con = Dense(128, activation='linear')(x_con)
            x_temp_B = tf.keras.layers.Add()([x_temp_B, x_con])
        x_B = Dense(128, activation='linear')(x_temp_B)
        x_B = tf.keras.activations.swish(x_B)
        ModelY = tf.keras.Model(inputs=inputB, outputs=x_B)
        combinedXY = Concatenate(axis=-1)([ModelX.output, ModelY.output])
        combinedXY = Dense(256, activation='linear')(combinedXY)
        combinedXY = tf.keras.activations.swish(combinedXY)
        outputs = Dense(self.action_size, activation='linear', name="Outputs")(combinedXY)
        model = tf.keras.Model(inputs=[ModelX.input, ModelY.input], outputs=[outputs])
        model.compile(loss='mse', optimizer='adam')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())  # copy weights from model to target_model

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            act_type = 'random'
            return random.randrange(self.action_size), act_type
        act_values = self.model.predict(state, verbose=0)
        act_type = 'RL'
        return np.argmax(act_values[0]), act_type # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        minibatch[0] = self.memory[len(self.memory)-1]
        state_batch_0, state_batch_1, target_batch = [], [], []
        for state, action, reward, next_state in minibatch:
            ActionIndex = np.argmax(self.model.predict(next_state, verbose=0)[0]) ### Selection using DQN
            target = reward + self.gamma * self.target_model.predict(next_state, verbose=0)[0][ActionIndex]   # Evaluation using target network
            target_f = self.model.predict(state, verbose=0) ## DQN
            target_f[0][action] = target
            state_batch_0.append(state[0]) ## Sub-state 1 -- Estimated Equivalent Wireless Channel H
            state_batch_1.append(state[1]) ## Sub-state 2 -- Current Reflection Pattern
            target_batch.append(target_f[0]) ## target batch
            InputA_data = np.reshape(np.array(state_batch_0), (batch_size, -1))
            InputB_data = np.reshape(np.array(state_batch_1), (batch_size, -1))
        history = self.model.fit( {"InputA": InputA_data, "InputB": InputB_data }, {"Outputs":np.array(target_batch)}, epochs=1, verbose=0)

        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
