from spaceinvaders.worker import Worker
from spaceinvaders.model import A3CModel
import tensorflow as tf
import os
import time
import threading as thread

devices = ['/cpu:0', '/cpu:0']
n_max_iter = 10000
env_obs_n = [80, 80]
env_act_n = 6

save_index = 0

session = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()

# TODO preprocessing missing last row of invaders
checkpoint = "./run-" + str(save_index) + ".ckpt"
if os.path.isfile(checkpoint + ".meta"):
    saver.restore(session, checkpoint)
elif save_index != 0:
    raise Exception("Session data not found!!")


root_logdir = "tf_logs/"
logdir = "{}/run-{}/".format(root_logdir, save_index + 1)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

master = A3CModel(session, env_obs_n, env_act_n, 'global')
workers = []
for index, device in enumerate(devices):
    workers.append(Worker(session, 'worker' + str(index), device, env_obs_n, env_act_n, n_max_iter))

coord = tf.train.Coordinator()
worker_threads = []
for worker in workers:
    def worker_work():
        worker.work(coord)

    t = thread.Thread(target=worker_work)
    t.start()
    time.sleep(0.5)
    worker_threads.append(t)
coord.join(worker_threads)