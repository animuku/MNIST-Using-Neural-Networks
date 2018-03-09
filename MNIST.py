import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tensorflow.python.framework import ops
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)


batch_size=100

def create_placeholders(n_x,n_y):
   X=tf.placeholder(tf.float32,shape=(None,n_x))
   Y=tf.placeholder(tf.float32,shape=(None,n_y))
   
   return X,Y
   



def initialize_parameters():
    
    W1 = tf.get_variable("W1",[784,500],initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1",[500],initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2",[500,500],initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2",[500],initializer=tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable("W3",[500,10],initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3",[10],initializer=tf.contrib.layers.xavier_initializer())
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    return parameters

    


def forward_propagation(X,parameters):
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
    W3=parameters["W3"]
    b3=parameters["b3"]

    Z1=tf.add(tf.matmul(X,W1),b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(A1,W2),b2)                                              
    A2 = tf.nn.relu(Z2)                                              
    Z3 = tf.add(tf.matmul(Z2,W3),b3) 
    
    return Z3
    

    
def compute_cost(Z3,Y):
    
    
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))
    
    return cost


tf.reset_default_graph()

X,Y=create_placeholders(784,10)

parameters=initialize_parameters()

Z3=forward_propagation(X,parameters)

cost=compute_cost(Z3,Y)

optimizer=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

init=tf.global_variables_initializer()

n_epochs=15
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        epoch_loss=0
        for z in range(int(mnist.train.num_examples/batch_size)):
            
            epoch_x,epoch_y=mnist.train.next_batch(batch_size)
            _,m_cost=sess.run([optimizer,cost],feed_dict={X:epoch_x,Y:epoch_y})
            epoch_loss=epoch_loss+m_cost
        
        
        print epoch_loss
    #parameters=sess.run(parameters)
    
    correct_prediction=tf.equal(tf.argmax(Z3,1),tf.argmax(Y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))
    
    print ("Test Accuracy:", accuracy.eval({X:mnist.test.images , Y: mnist.test.labels}))   





    
        


    



