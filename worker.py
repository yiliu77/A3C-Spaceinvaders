from spaceinvaders.model import A3CModel
import tensorflow as tf
import gym
import numpy as np


def update_graph(transfer_id, target_id):
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target_id)
    transfer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, transfer_id)

    assign_ops = []
    for transfer_var, target_var in zip(transfer_vars, target_vars):
        assign_ops.append(target_var.assign(transfer_var))
    return assign_ops


def process(image):
    image = image[::2, ::2, :]
    image = image[24:104, :80, :]
    image = np.mean(image, axis=2)
    image = image / 128 - 1
    return np.reshape(image, [1, 80, 80, 1])


class Worker:
    def __init__(self, sess, agent_id, device_id, env_obs_n, env_act_n, n_max_iter):
        self.sess = sess
        self.env = gym.make('SpaceInvaders-v0')
        self.env._max_episode_steps = n_max_iter
        self.n_max_iter = n_max_iter
        self.env_obs_n = env_obs_n
        self.env_act_n = env_act_n

        self.model = A3CModel(sess, env_obs_n, env_act_n, agent_id, device_id)
        self.update_worker_ops = update_graph('global', agent_id)

    def work(self, coord):
        while not coord.should_stop():
            self.sess.run(self.update_worker_ops)
            # noinspection PyRedeclaration
            obs = process(self.env.reset())

            for tick in range(self.n_max_iter):
                action = self.model.get_rand_action(obs)
                new_obs, reward, done, info = self.env.step(action)
                new_obs = process(new_obs)

                self.model.train(obs, action, reward, new_obs, int(done))

                obs = new_obs
                if done:
                    break

