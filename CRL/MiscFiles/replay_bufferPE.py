from collections import namedtuple, deque
import random
import numpy as np
from SumTree import SumTree

class ReplayBufferPE:
    """Fixed-size buffer to store experience tuples."""
    # This is according to the formula (error + e)**alpha
    epsilon = 0.01
    alpha = 0.6

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.max_size=buffer_size
        self.memory = SumTree(buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def _getPriority(self, error):
        return (error + self.epsilon) ** self.alpha

    # Look into: This is the only thing that changed
    def add(self, error, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        p = self._getPriority(error)
        self.tree.add(p, e)


    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = []
        segment = self.tree.total() / self.batch_size

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            experiences.append( data )


        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)

        return (states, actions, rewards, next_states, dones)


    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


    #
    #     errors = numpy.zeros(len(batch))
    #
    #     for i in range(len(batch)):
    #         o = batch[i][1]
    #         s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
    #
    #         t = p[i]
    #         oldVal = t[a]
    #         if s_ is None:
    #             t[a] = r
    #         else:
    #             t[a] = r + GAMMA * pTarget_[i][ numpy.argmax(p_[i]) ]  # double DQN
    #
    #         x[i] = s
    #         y[i] = t
    #         errors[i] = abs(oldVal - t[a])
    #
    #     return (x, y, errors)
    #
    # def replay(self):
    #     batch = self.memory.sample(BATCH_SIZE)
    #     x, y, errors = self._getTargets(batch)
    #
    #     #update errors
    #     for i in range(len(batch)):
    #         idx = batch[i][0]
    #         self.memory.update(idx, errors[i])
    #
    #     self.brain.train(x, y)
