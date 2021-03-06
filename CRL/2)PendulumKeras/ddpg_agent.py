import numpy as np
import random
import copy
from collections import namedtuple, deque
from replay_buffer import ReplayBuffer

import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Input, Add, Activation, LeakyReLU
from keras.layers import GaussianNoise, Dropout, Concatenate
from keras.models import Model
from keras.optimizers import Adam

NUM_DIM=2
NUM_ACTIONS = NUM_DIM+2
TREE_DEPTH=8
BUFFER_SIZE = int(1e3)  # replay buffer size
BATCH_SIZE = 64        # minibatch size
GAMMA = 0.9            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic

# Fully Connected Layer's size was set to
# 2*NUM_ACTIONS
FC_ACTOR = 80*NUM_ACTIONS
FC_CRITIC = 80*NUM_ACTIONS


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self,env, sess, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.env = env
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        # Noise process
        self.noise = OUNoise(action_size, random_seed)



        # Actor Network (w/ Target Network)
        self.actor_local = self.Actor(state_size, action_size, random_seed,FC_ACTOR)
        self.actor_target = self.Actor(state_size, action_size, random_seed,FC_ACTOR)
        self.actor_local.compile(loss='mse', optimizer=Adam(lr=LR_ACTOR))

        # where we will feed de/dC (from critic)
        self.actor_state_input = self.actor_local.input
        # The line below means that we there will be rows (batch size in number)
        # and columns (action_size in number) where each i,j means the ith sample's
        # jth gradient
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, action_size])
        self.actor_local_weights = self.actor_local.trainable_weights

        # dC/dA (from actor)
        self.actor_grads = tf.gradients(self.actor_local.output, self.actor_local_weights, -self.actor_critic_grad)

        grads = zip(self.actor_grads, self.actor_local_weights)
        self.optimize = tf.train.AdamOptimizer(LR_ACTOR).apply_gradients(grads)
        # Critic Network (w/ Target Network)
        self.critic_local = self.Critic(state_size, action_size, random_seed,FC_CRITIC)
        self.critic_target = self.Critic(state_size, action_size, random_seed,FC_CRITIC)
        self.critic_local.compile(loss='mse', optimizer=Adam(lr=LR_CRITIC))

        #Critic Gradients
        self.critic_state_input,self.critic_action_input=self.critic_local.input
        self.critic_grads = tf.gradients(self.critic_local.output, self.critic_action_input)
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Initialize for later gradient calculations
        self.sess.run(tf.global_variables_initializer())

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.predict(next_states)
        Q_targets_next = self.critic_target.predict([next_states, actions_next])
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        loss = self.critic_local.train_on_batch([states,actions], Q_targets )

        # ---------------------------- update actor ---------------------------- #

        actions_pred = self.actor_local.predict(states)
        # Doubt: look into if the [0] at the end of grads
        #       is necessary

        # The below evaluates self.critic_grads
        # By feeding the inputs
        grads = self.sess.run(self.critic_grads, feed_dict={
            self.critic_state_input:  states,
            self.critic_action_input: actions_pred
        })[0]

        self.sess.run(self.optimize, feed_dict={
            self.actor_state_input: states,
            self.actor_critic_grad: grads
        })

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)




    def Critic(self, state_size, action_size, seed, fc_units=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (int): Number of nodes in the hidden layers
        """

        S = Input(shape=[state_size])
        A = Input(shape=[action_size])

        S1 = Dense(fc_units)(S)
        S1 = GaussianNoise(0.1)(S1)
        S1 = Activation('relu')(S1)
        S1 = Dropout(0.25)(S1)
        S2 = Dense(fc_units)(S1)
        S2 = GaussianNoise(0.1)(S2)
        S2 = Activation('relu')(S2)
        A1 = Dense(fc_units)(A)
        A1 = GaussianNoise(0.1)(A1)
        A1 = Activation('relu')(A1)

        h0 = Add()([A1,S2])
        h1 = Dense(int(fc_units/4), activation='relu')(h0)
        h1 = GaussianNoise(0.1)(h1)
        h1 = Activation('relu')(h1)
        h1 = Dropout(0.25)(h1)
        finalOutput = Dense(action_size)(h1)
        finalOutput = LeakyReLU(alpha=0.1)(finalOutput)
        return Model([S,A],finalOutput)


    def Actor(self, state_size, action_size, seed, fc_units=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (int): Number of nodes in the hidden layers

        """


        S = Input(shape=[state_size])
        h0 = Dense(fc_units)(S)
        h0 = GaussianNoise(0.1)(h0)
        h0 = Activation('relu')(h0)
        h0 = Dropout(0.25)(h0)
        h1 = Dense(fc_units)(h0)
        h1 = GaussianNoise(0.1)(h1)
        h1 = Activation('relu')(h1)
        h1 = Dropout(0.25)(h1)

        Policy = Dense(action_size,activation='tanh')(h1)
        # Note: The policy has NUM_DIM + 1 in length
        #       The last node is for done action

        return Model(S,Policy)




    def NetworkSummary(self):
        print("Actor Summary")
        self.actor_target.summary()
        print("Critic Summary")
        self.critic_target.summary()

    def load_network(self, path, extension):
        self.actor_local.load_weights(path+'actor_local_'+extension)
        self.actor_target.load_weights(path+'actor_target_'+extension)
        self.critic_local.load_weights(path+'critic_local_'+extension)
        self.critic_target.load_weights(path+'critic_target_'+extension)
        print("Successfully saved network.")

    def save_network(self, path, extension):
        # Saves model at specified extension as h5 file

        self.actor_local.save(path+'actor_local_'+extension)
        self.actor_target.save(path+'actor_target_'+extension)
        self.critic_local.save(path+'critic_local_'+extension)
        self.critic_target.save(path+'critic_target_'+extension)
        #print("Successfully saved network.")

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise):
        """Returns actions for given state as per current policy."""
        state=np.array([state])

        action = self.actor_local.predict(state)[0]
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, 0, 1)




    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        local_weights = local_model.get_weights()
        target_weights = target_model.get_weights()

        for i in range(len(local_weights)):
            target_weights[i] = tau * local_weights[i] + (1 - tau)* target_weights[i]

        target_model.set_weights(target_weights)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
