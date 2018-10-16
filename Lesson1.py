# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:02:55 2018

@author: 寻ME
"""

import tensorflow as tf

a=tf.add(3,5)
with tf.Session() as sess:
    print(sess.run(a))

#当然可以使用更加复杂的计算图如下
    
x=2
y=3
add_op=tf.add(x,y)
mul_op=tf.multiply(x,y)
useless=tf.multiply(x,add_op)
pow_op=tf.pow(add_op,mul_op)
with tf.Session() as sess:
    z,not_useless=sess.run([pow_op,useless])
    
#可以将计算图放在特定的GPU或者CPU之下
    
with tf.device('/gpu:2'):
    a=tf.constant([[1.0,2.0,3.0][4.0,5.0,6.0]],name='a')
    b=tf.constant([[1.0,2.0],[3.0,4.0],[5.0,6.0]],name='b')
    c=tf.matmul(a,b)


