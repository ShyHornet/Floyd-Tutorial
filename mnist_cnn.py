import argparse
import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import math
FLAGS = None
#下载mnist数据

def train():
    mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
    sess = tf.InteractiveSession()
    with tf.name_scope('input'):
        X   = tf.placeholder(tf.float32,[None,28,28,1],name='X')
        Y_ = tf.placeholder(tf.float32,[None,10])
        #None:mini-batch中图像编号
        #28,28手写数字图像大小
        W  = tf.Variable(tf.truncated_normal([784,10],stddev=0.1))
        #785个图像,10个权值，分别对应数字0~9
    with tf.name_scope('input_reshaped'):
        XX = tf.reshape(X, [-1, 28*28])
        tf.summary.image('input',X, 10)
    def weight_variable(shape):
        init =  tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init)
    def bias_variable(shape):
        init = tf.ones(shape)/10
        return tf.Variable(init)
    def variable_summaries(var):
        with tf.name_scope('summaries'):
          mean = tf.reduce_mean(var)
          tf.summary.scalar('mean', mean)
          with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
          tf.summary.scalar('stddev', stddev)
          tf.summary.scalar('max', tf.reduce_max(var))
          tf.summary.scalar('min', tf.reduce_min(var))
          tf.summary.histogram('histogram', var)
    def cnn_layer(input_tensor, shape,depth,stride, layer_name, fully_connected=False):
        act=tf.nn.relu
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable(shape)
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([depth])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                if fully_connected!=True:
                    preactivate = tf.nn.conv2d(input_tensor,weights, strides=[1, stride, stride, 1], padding='SAME')+biases
                    tf.summary.histogram('pre_activations', preactivate)
                else:
                    preactivate = tf.matmul(input_tensor, weights) + biases
                    tf.summary.histogram('pre_activations_fully_connected', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
        return activations
    K = 4  # first convolutional layer output depth
    L = 8  # second convolutional layer output depth
    M = 12  # third convolutional layer
    N = 200  # fully connected layer
    hidden1 = cnn_layer(X,[5, 5, 1, K],K,1,'SAME',"hidden_layer1")
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)
    hidden2 = cnn_layer(dropped,[5, 5, K, L],L,2,"hidden_layer2")
    hidden3 = cnn_layer(hidden2,[4, 4, L, M],M,2,"hidden_layer3")
    hidden4_input = tf.reshape(hidden3, shape=[-1, 7 * 7 * M])
    hidden4 = cnn_layer(hidden4_input ,[7 * 7 * M, N],N,1,"hidden_layer4",True)
    with tf.name_scope('output'):
        Ylogits = tf.matmul(hidden4,weight_variable([N,10])) + bias_variable([10])
        Y = tf.nn.softmax(Ylogits)

    with tf.name_scope('cross_entropy'):
        #定义损失函数(交叉熵)
        # cross_entropy = -tf.reduce_sum(Y_*tf.log(Y))
        #更安全的计算方式(避免计算log(0),导致准确率断崖式下跌)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,labels= Y_)
        #正则化交叉熵
        cross_entropy = tf.reduce_mean(cross_entropy)*100
        #计算单个批次中正确结果的百分比

    with tf.name_scope('accuracy'):
         with tf.name_scope('correct_prediction'):
            is_correct = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
         with tf.name_scope('accuracy'):
            accuracy =  tf.reduce_mean(tf.cast(is_correct,tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    with tf.name_scope('train'):
        with tf.name_scope('learning_rate'):
            lr = tf.placeholder(tf.float32)
        optimizer = tf.train.AdamOptimizer(lr)
        train_step = optimizer.minimize(cross_entropy)



    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/output', sess.graph)
    tf.global_variables_initializer().run()
    for i in range(FLAGS.steps):
        #读取当前批次的图像和正确结果
        batch_X,batch_Y = mnist.train.next_batch(100)

        #实现学习率衰减
        max_learning_rate = 0.003
        min_learning_rate = 0.0001
        decay_speed = 2000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
        k = 0.9
        train_data = {X: batch_X, Y_: batch_Y, lr: learning_rate,keep_prob:FLAGS.dropout}
        #开始训练

        if i%10 == 0:
            summary,a,c,_ = sess.run([merged,accuracy,cross_entropy,train_step],feed_dict=train_data)
            print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
            train_writer.add_summary(summary, i)
        if i%100 == 0:
            test_data={X: mnist.test.images, Y_: mnist.test.labels,lr: learning_rate,keep_prob:k}
            _,a,c,_ = sess.run([merged,accuracy, cross_entropy,train_step], feed_dict=test_data)
            print(str(i) + " ********* test accuracy:" + str(a) + " test loss: " + str(c)+ " (lr:" + str(learning_rate) + ")")
        sess.run(train_step,train_data)
    train_writer.close()

def main(_):
    train()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000,
                      help='Number of steps to run trainer.')
    parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
