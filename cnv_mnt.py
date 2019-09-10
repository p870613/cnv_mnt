from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
import time

#arg
learning_rate = 0.0001
train_epochs = 100
batch_size = 100
display_step = 1

def con2d(input, w_shape, b_shape):
     incoming = w_shape[0] * w_shape[1] * w_shape[1]
     w_init = tf.random_normal_initializer(stddev = (2.0/incoming) ** 0.5)
     b_init = tf.constant_initializer(value = 0)
     
     w = tf.get_variable("w", w_shape, initializer = w_init)
     b = tf.get_variable("b", b_shape, initializer = b_init)

     # w is filter, padding:SAME 可以自己補0
     # value may have any number of dimensions
     return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, w, strides = [1, 1, 1 ,1], padding = 'SAME'), b))
     
def max_pool(input, k = 2):
     # ksize = [batch, height, width, channels]
     return tf.nn.max_pool(input, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = 'SAME')

def layer(input, w_shape, b_shape):
     w_init = tf.random_normal_initializer(stddev = (2.0/w_shape[0])**0.5)
     b_init = tf.constant_initializer(value = 0)

     w = tf.get_variable("w", w_shape, initializer = w_init)
     b = tf.get_variable("b", b_shape, initializer = b_init)
     
     return tf.nn.relu(tf.matmul(input, w) + b)

def inference(x, keep_prob):

     x = tf.reshape(x, shape = [-1, 28, 28, 1])
     #捲機1
     with tf.variable_scope("conv_1"):
          conv_1 = con2d(x, [5, 5, 1, 32], [32])
          print(tf.shape(conv_1))
          pool_1 = max_pool(conv_1)
          print(tf.shape(pool_1))
          
     #捲機2
     with tf.variable_scope("conv_2"):
          conv_2 = con2d(pool_1, [5, 5, 32, 64], [64])
          print(tf.shape(conv_2))
          pool_2 = max_pool(conv_2)
          print(tf.shape(pool_2))
          
     #全連接
     with tf.variable_scope("dense"):
          pool_2_flat = tf.reshape(pool_2, [-1, 7* 7* 64])
          fc_1 = layer(pool_2_flat, [7*7*64, 1024], [1024])

          #dropout是把一些神經元丟掉另外以p的機率留神經元
          fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

     # softmax
     with tf.variable_scope("output"):
          output = layer(fc_1_drop, [1024, 10], [10])

     return output

def loss(output, y):
     xentropy = xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)  
     loss = tf.reduce_mean(xentropy)
     return loss

def train(cost, global_step):
     #summary record
     tf.summary.scalar("cost", cost)

     optimizer = tf.train.AdamOptimizer(learning_rate)
     train_op = optimizer.minimize(cost, global_step = global_step)

     return train_op


def evaluate(output, y):
     #計算accuracy
     correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
     ac = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     
     #record
     tf.summary.scalar("error", (1.0 - ac))
     return ac

if __name__ == '__main__':
     with tf.variable_scope("mnist_conv_model"):
          x = tf.placeholder("float", [None, 784])
          y = tf.placeholder("float", [None, 10])
          keep_prob = tf.placeholder(tf.float32)
     
          output = inference(x, keep_prob)

          cost = loss(output, y)

          global_step = tf.Variable(0, name = 'global_step', trainable = False)
          train_op = train(cost, global_step)

          eval_op = evaluate(output, y)

          summary_op = tf.summary.merge_all()

          saver = tf.train.Saver()

          with tf.Session() as sess:
               summary_writer = tf.summary.FileWriter("logistic_logs/",graph_def=sess.graph_def)

               init_op = tf.global_variables_initializer()
               sess.run(init_op)

               for epoch in range(train_epochs):
                    avg_cost = 0
                    total_batch = int(mnist.train.num_examples/batch_size)

                    for i in range(total_batch):
                         batch_x, batch_y = mnist.train.next_batch(batch_size)
                         sess.run(train_op, feed_dict = {x:batch_x, y : batch_y, keep_prob:0.5})
                         avg_cost = avg_cost + sess.run(cost, feed_dict = {x:batch_x, y : batch_y, keep_prob:0.5})/total_batch

                    if epoch % display_step == 0:
                         print("Epoch: ", epoch+1, ", cost: ", avg_cost)

                    ac = sess.run(eval_op, feed_dict={x: mnist.validation.images, y: mnist.validation.labels, keep_prob: 1})
                    print("ac", ac)

                    summary_str = sess.run(summary_op, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
                    summary_writer.add_summary(summary_str, sess.run(global_step))
                    saver.save(sess, "conv_mnist_logs/model-checkpoint", global_step=global_step)
               print("finish")

               accuracy = sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1})

               print ("Test Accuracy:", accuracy)

