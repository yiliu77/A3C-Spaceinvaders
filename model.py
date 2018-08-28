class A3C:
    def __init__(self, sess, input_size, output_size):
        self.sess = sess

        self.n_in = input_size
        self.n_out = output_size

        self.lr = 0.0001
        self.gamma = 0.99
        self.entropy_beta = 0.01

        self.lstm_cells = 64

        self.s = tf.placeholder(tf.float32, [None, self.n_in[0], self.n_in[1], 1], "state")  # grayscale 1
        self.a = tf.placeholder(tf.int32, None, "action")
        self.r = tf.placeholder(tf.float32, None, "reward")
        self.s_ = tf.placeholder(tf.float32, [None, self.n_in[0], self.n_in[1], 1], "next_state")
        self.v_ = tf.placeholder(tf.float32, None, "next_v")
        self.end = tf.placeholder(tf.float32, None, "done")

        with tf.variable_scope("network"):
            conv1 = tf.layers.conv2d(self.s, 32, [5, 5], padding="SAME", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            conv2 = tf.layers.conv2d(pool1, 32, [5, 5], padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            conv3 = tf.layers.conv2d(pool2, 32, [5, 5], padding="SAME", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

            conv4 = tf.layers.conv2d(pool3, 32, [5, 5], padding="SAME", activation=tf.nn.relu)
            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

            # LSTM
            flat = tf.reshape(pool4, [-1, self.n_in[0] * self.n_in[1] * 32 // (4 ** 4)])
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_cells, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]

            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = [c_in, h_in]

            rnn_in = tf.expand_dims(flat, [0])
            state_in = tf.contrib.rnn.LSTMStateTuple(c_init, h_init)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=state_in, time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            self.shared = tf.reshape(lstm_outputs, [-1, self.lstm_cells])

            # value and policy
            self.v = tf.squeeze(tf.layers.dense(self.shared, 1))
            self.policy = tf.layers.dense(self.shared, self.n_out)

        self.probs = tf.nn.softmax(self.policy)
        self.real_action = tf.argmax(self.probs[0, :], 0)
        self.rand_action = tf.squeeze(tf.multinomial(tf.log(self.probs), num_samples=1))

        with tf.variable_scope("training"):
            self.critic_loss = self.r + self.gamma * self.v_ * (1.0 - self.end) - self.v

            one_hot = tf.one_hot(self.a, self.n_out)
            self.log_prob = tf.reduce_sum(tf.log(self.probs * one_hot + 1e-6), axis=1)
            self.actor_loss = -tf.reduce_mean(tf.stop_gradient(self.critic_loss) * self.log_prob)

            entropy = tf.reduce_sum(self.probs * tf.log(self.probs + 1e-6))
            self.entropy_loss = self.entropy_beta * entropy

            self.total_loss = tf.reduce_mean(0.5 * tf.square(self.critic_loss) + self.actor_loss + self.entropy_loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.minimize(self.total_loss)

    def train(self, s, a, r, s_, end, v_):
        feed_dict = {self.s: s, self.a: a, self.r: r, self.s_: s_, self.v_: v_, self.end: end}
        _, training_error = self.sess.run([self.train_op, self.total_loss], feed_dict=feed_dict)
        return training_error

    def get_value(self, s):
        return self.sess.run(self.v, feed_dict={self.s: s})

    def get_rand_action(self, s, rnn):
        return self.sess.run([self.rand_action, self.v, self.state_out], feed_dict={self.s: s,
                                                                                    self.state_in[0]: rnn[0],
                                                                                    self.state_in[1]: rnn[1]})

    def get_real_action(self, s, rnn):
        return self.sess.run(self.real_action, feed_dict={self.s: s, self.state_in[0]: rnn[0],
                                                          self.state_in[1]: rnn[1]})
