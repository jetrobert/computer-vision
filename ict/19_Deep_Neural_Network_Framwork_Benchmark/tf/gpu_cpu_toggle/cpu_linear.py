import tensorflow as tf
import numpy as np
import os 

# 20% of gpu 0 volume is used by session 
os.environ['CUDA_VISIBLE_DEVICES']=''
'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.2
sess = tf.InteractiveSession(config=config)
'''
sess = tf.InteractiveSession()
x_data = np.random.rand(100).astype("float32")
y_data = x_data * 0.1 + 0.3

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

for step in xrange(20001):
    sess.run(train)
    if step % 20 ==0:
        print(step, sess.run(W), sess.run(b))
		