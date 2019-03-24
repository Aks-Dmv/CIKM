from collections import deque
import random
import numpy as np

class ReplayBuffer:
    """Constructs a buffer object that stores the past moves
    and samples a set of subsamples"""

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a,upDown,regressor, r, d, s2):
        """Add an experience to the buffer"""
        """We either append or pop the oldest experience and then append"""
        # S represents current state, a is action,
        # r is reward, d is whether it is the end,
        # and s2 is next state
        experience = (s, a,upDown,regressor, r, d, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample(self, batch_size):
        """Samples a total of elements equal to batch_size from buffer
        if buffer contains enough elements. Otherwise return all elements"""

        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)

        else:
            batch = random.sample(self.buffer, batch_size)

        # Maps each experience in batch in batches of states, actions, rewards
        # and new states
        print(batch)
        # this print is working
        # print("map",map(np.array, list(zip(*batch))))
        # the map is also working but we don't know if it is correct
        #print(list(map(np.array, list(zip(*batch)))))
        s_batch, a_batch,upDown_batch,reg_batch, r_batch, d_batch, s2_batch = list(map(np.array, list(zip(*batch))))

        # it was observed that s_batch initially had dim (batchSize,1,TREE_DEPTH,NUM_ACTIONS)
        #print("before",s_batch.shape)
        #s_batch=s_batch.squeeze()
        #print("after",s_batch.shape)
        return s_batch, a_batch,upDown_batch,reg_batch, r_batch, d_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0



if __name__ == "__main__":
    print("ReplayBuffer")
    # This is just a hardcoded test case
    # a=ReplayBuffer(10)
    # a.add(1,2,3,4,5)
    # a.add(11,2,3,4,5)
    # a.add(111,2,3,4,5)
    # a.add(2,2,3,4,5)
    # a.add(22,2,3,4,5)
    # a.add(222,2,3,4,5)
    # print(a.sample(3))
    # a.clear()
    # a.add(3,2,3,4,5)
    # a.add(33,2,3,4,5)
    # a.add(333,2,3,4,5)
    # a.add(3333,2,3,4,5)
    # print(a.sample(3))
