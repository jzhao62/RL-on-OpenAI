import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque
from scipy.misc import imresize
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class DQN_Agent:
    def __init__(self, problem, buffer_size, observe_before_train, load_prev_model, model_path, index):
        self.index = index
        self.buffer_size = buffer_size

        self.env = gym.make(problem)
        assert self.env.action_space.__repr__().startswith('Discrete')
        self.num_action = self.env.action_space.n

        self.replay_memory = deque()
        self.time_step = 0
        self.initialize_replay_memory(observe_before_train)
        self.collect_state()

        self.cumulative_training = 0
        self.loss_value = 0
        self.model_path = model_path
        self.initialize_model(load_prev_model)


    def read_observation(self, observation, action, reward, terminal, need_train = True):

        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        else:
            reward = 0

        state = self.image_preproc(observation)
        new_state = np.stack((state, self.curr_state[:,:,0], self.curr_state[:,:,1], self.curr_state[:,:,2]), axis = 2)
        self.replay_memory.append((self.curr_state, action, reward, new_state, terminal))

        if len(self.replay_memory) > self.buffer_size:
            self.replay_memory.popleft()

        self.curr_state = new_state

        if need_train:
            self.time_step += 1
            self.experience_replay()

        #fig = plt.figure(figsize=(16, 12), dpi=80)
        #a=fig.add_subplot(2,3,1)
        #imgplot = plt.imshow(self.curr_state[:,:,0])
        #a.set_title('0')
        #a=fig.add_subplot(2,3,2)
        #imgplot = plt.imshow(self.curr_state[:,:,1])
        #a.set_title('2')
        #a=fig.add_subplot(2,3,3)
        #imgplot = plt.imshow(self.curr_state[:,:,2])
        #a.set_title('3')
        #a=fig.add_subplot(2,3,4)
        #imgplot = plt.imshow(self.curr_state[:,:,3])
        #a.set_title('4')
        #a=fig.add_subplot(2,3,5)
        #imgplot = plt.imshow(np.sum(self.curr_state,axis = 2))
        #a.set_title('all')

        #plt.show()

    def initialize_replay_memory(self, observe_before_train):
        observe_before_train = min(observe_before_train, self.buffer_size)
        observation = self.env.reset()
        self.initial_state(observation)
        t = time.time()
        for i in range(0, observe_before_train):

            action = self.env.action_space.sample()
            observation, reward, done, _ = self.env.step(action)
            self.read_observation(observation, action, reward, done, need_train = False)
            if done:
                observation = self.env.reset()
                self.initial_state(observation)

            t_e = time.time() - t
            if i % 10000 == 0:
                print "Observed %i steps, still need %i steps, memory size %i, speed %i s" %(i, observe_before_train - i, len(self.replay_memory), t_e)

        print "Finish observation, start training"


    def next_action(self):
	self.e = max(0.1, - 0.9 * self.time_step / 1000000 + 1)
        if np.random.rand(1)[0] < self.e:
            action = self.env.action_space.sample()
        else:
            QValue = self.QValue.eval(feed_dict= {self.state_input:[self.curr_state]})[0]
            action = np.argmax(QValue)

        return action

    def experience_replay(self):

        batch_size = 32
        assert len(self.replay_memory) > 0.5 * self.buffer_size

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_memory, batch_size)
        state_batch = [data[0] for data in minibatch]
        action_idx_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]
        terminal_batch = [data[4] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        action_batch =[]
        QValue_next_state_batch = self.QValue.eval(feed_dict={self.state_input:nextState_batch})
        for i in range(0,batch_size):
            action = np.zeros(self.num_action)
            action[action_idx_batch[i]] = 1
            action_batch.append(action)
            if terminal_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + 0.99 * (self.e * np.mean(QValue_next_state_batch[i]) + (1 - self.e) * np.max(QValue_next_state_batch[i])))

        _, self.loss_value = self.session.run([self.trainStep, self.cost],
                                feed_dict={self.yInput:y_batch, self.action_input:action_batch, self.state_input:state_batch})

        self.cumulative_training += batch_size
        if self.cumulative_training > 200000 * 32:
            self.cumulative_training = 0
            self.index += 1
            save_path = self.saver.save(self.session, self.model_path + "_" + str(self.index) + ".ckpt")
            print("Model saved in file: %s" % save_path)


    def collect_state(self):

        observation = self.env.reset()
        state = self.image_preproc(observation)
        curr_state = np.stack((state, state, state, state), axis = 2)
        self.evoluation_set = [curr_state]

        for i in range(0,999):
            action = self.env.action_space.sample()
            observation, reward, done, _ = self.env.step(action)
            state = self.image_preproc(observation)
            next_state = np.stack((state, curr_state[:,:,0], curr_state[:,:,1], curr_state[:,:,2]), axis = 2)
            self.evoluation_set.append(next_state)
            curr_state = next_state
            if done:
                observation = self.env.reset()
                state = self.image_preproc(observation)
                curr_state = np.stack((state, state, state, state), axis = 2)


    def evoluate(self):
        QValue = self.QValue.eval(feed_dict={self.state_input:self.evoluation_set})
        mean = 0
        for i in range(0, 1000):
            mean += np.max(QValue[i])
        return mean/1000

    def initialize_model(self, load_prev_model):
        # network weights
        W_conv1 = self.weight_variable([8,8,4,16])
        b_conv1 = self.bias_variable([16])

        W_conv2 = self.weight_variable([4,4,16,32])
        b_conv2 = self.bias_variable([32])

        # input layer
        self.state_input = tf.placeholder("float",[None,84,84,4])

        # convolutional layers
        h_conv1 = tf.nn.relu(self.conv2d(self.state_input,W_conv1,4) + b_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1,W_conv2,2) + b_conv2)

        # vectorization layer
        flat_size = h_conv2.shape[1].value * h_conv2.shape[2].value * h_conv2.shape[3].value
        flat_input = tf.reshape(h_conv2,[-1,flat_size])

        # flat hidden layer
        W_fc1 = self.weight_variable([flat_size,256])
        b_fc1 = self.bias_variable([256])
        h_fc1 = tf.nn.relu(tf.matmul(flat_input,W_fc1) + b_fc1)

        # Q Value layer
        W_fc2 = self.weight_variable([256,self.num_action])
        b_fc2 = self.bias_variable([self.num_action])
        self.QValue = tf.matmul(h_fc1,W_fc2) + b_fc2

        #self.yInput = tf.placeholder("float", [None, num_action])
        #self.cost = tf.reduce_sum(tf.square(self.yInput - self.QValue))
        #self.trainStep = tf.train.RMSPropOptimizer(0.00001).minimize(self.cost)

        self.action_input = tf.placeholder("float",[None,self.num_action])
        self.yInput = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.QValue, self.action_input), reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

        if load_prev_model:
            self.saver.restore(self.session, self.model_path + "_" + str(self.index) + ".ckpt")
            print "Successfully loaded:", self.model_path + "_" + str(self.index) + ".ckpt"
        else:
            print "New Model"

    def initial_state(self, observation):
        state = self.image_preproc(observation)
        self.curr_state = np.stack((state, state, state, state), axis = 2)

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    def image_preproc(self, observation):
        observation = (0.33 * observation[:,:,0] + 0.33 * observation[:,:,1] + 0.33 * observation[:,:,2])/255
        observation = imresize(observation, (110, 84))
        observation = observation[18:102, :]
        return observation
