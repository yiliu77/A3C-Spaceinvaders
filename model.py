import tensorflow as tf
import random


class A3CModel:
    def __init__(self, sess, input_size, output_size, agent_id, device_id=None):
        with tf.variable_scope(agent_id) and tf.device(device_id):
            self.sess = sess

            self.n_in = input_size
            self.n_out = output_size

            self.lr = 0.00005
            self.gamma = 0.99
            self.entropy_beta = 0.01
            self.rand_chance = 0.15

            self.s = tf.placeholder(tf.float32, [1, self.n_in[0], self.n_in[1], 1], "state")  # grayscale 1
            self.a = tf.placeholder(tf.int32, None, "action")
            self.r = tf.placeholder(tf.float32, None, "reward")
            self.s_ = tf.placeholder(tf.float32, [1, self.n_in[0], self.n_in[1], 1], "next_state")
            self.v_ = tf.placeholder(tf.float32, None, "next_v")
            self.end = tf.placeholder(tf.float32, None, "done")

            conv1 = tf.layers.conv2d(self.s, 32, [3, 3], padding="SAME", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            conv2 = tf.layers.conv2d(pool1, 32, [3, 3], padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            conv3 = tf.layers.conv2d(pool2, 64, [3, 3], padding="SAME", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

            conv4 = tf.layers.conv2d(pool3, 64, [3, 3], padding="SAME", activation=tf.nn.relu)
            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

            flat = tf.reshape(pool4, [-1, self.n_in[0] * self.n_in[1] * 64 // (16 ** 2)])  # TODO
            self.shared = tf.layers.dense(flat, 200, activation=tf.nn.relu)

            self.hidden_v = tf.layers.dense(self.shared, 200, activation=tf.nn.relu)
            self.v = tf.squeeze(tf.layers.dense(self.hidden_v, 1))

            self.hidden_policy = tf.layers.dense(self.shared, 200, activation=tf.nn.relu)
            self.policy = tf.layers.dense(self.hidden_policy, self.n_out)

            if agent_id != "global":
                self.probs = tf.nn.softmax(self.policy)
                self.real_action = tf.argmax(self.probs[0, :], 0)
                self.rand_action = tf.squeeze(tf.multinomial(tf.log(self.probs), num_samples=1))

                self.critic_loss = self.r + self.gamma * self.v_ * (1.0 - self.end) - self.v

                self.log_prob = tf.log(self.probs[0, self.a] + 1e-6)
                self.actor_loss = -tf.reduce_mean(self.log_prob * tf.stop_gradient(self.critic_loss))

                entropy = tf.reduce_sum(self.probs * tf.log(self.probs + 1e-6))
                self.entropy_loss = self.entropy_beta * entropy

                self.total_loss = 0.5 * tf.square(self.critic_loss) + self.actor_loss + self.entropy_loss

                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

                agent_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, agent_id)
                self.gradients = tf.gradients(self.total_loss, agent_vars)
                self.var_norms = tf.global_norm(agent_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = optimizer.apply_gradients(zip(grads, global_vars))

    def train(self, s, a, r, s_, end):
        v_ = self.get_value(s_)

        feed_dict = {self.s: s, self.a: a, self.r: r, self.s_: s_, self.v_: v_, self.end: end}
        _, training_error = self.sess.run([self.apply_grads, self.total_loss], feed_dict=feed_dict)
        return training_error

    def get_value(self, s):
        return self.sess.run(self.v, feed_dict={self.s: s})

    def get_rand_action(self, s):
        if random.random() < self.rand_chance:
            return random.randint(0, self.n_out - 1)
        return self.sess.run(self.rand_action, feed_dict={self.s: s})

    def get_real_action(self, s):
        return self.sess.run(self.real_action, feed_dict={self.s: s})