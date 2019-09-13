#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 09:22:34 2019

@author: niyin
"""

import os
from flask import Flask, request, redirect, url_for
from werkzeug import secure_filename
from flask import send_from_directory
import tensorflow as tf
import matplotlib.pyplot as plt
#import cv2
from PIL import Image, ImageFilter

UPLOAD_FOLDER = '/app'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 


"""
Load the model2.ckpt file
file is stored in the same directory as this python script is started
Use the model to predict the integer. Integer is returend as list.

Based on the documentatoin at
https://www.tensorflow.org/versions/master/how_tos/variables/index.html
"""


    
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('model', filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Welcome to my recognizing page</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
    

#@app.route('/uploads/<filename>')
#def uploaded_file(filename):
#    return send_from_directory(app.config['UPLOAD_FOLDER'],
#                               filename)

@app.route('/uploads/<filename>')
def model(filename):
    file_name=os.path.join(app.config['UPLOAD_FOLDER'], filename)#导入自己的图片地址
    #in terminal 'mogrify -format png *.jpg' cvert jpg to png
    im = Image.open(file_name)
    im = im.resize((28, 28), Image.ANTIALIAS).convert('L')
    #im.save("/home/niyin/桌面/docker/sample.png")
    plt.imshow(im)
    plt.show()
    tv = list(im.getdata()) #get pixel values
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    #print(tva)
    result=tva
    
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
        # Define the model (same as when creating the model file)
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    init_op = tf.initialize_all_variables()
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "/app/model.ckpt")#这里使用了之前保存的模型参数
    #print ("Model restored.")
        prediction=tf.argmax(y_conv,1)
        predint=prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)
        print(h_conv2)
        print('recognize result:')
        print(predint[0])
        finalresult=str(predint[0])
    return 'the regognized result of your picture is:'+finalresult




#@app.route('/uploads/shibie/result')
#def result(predint):
#    return predint[0]




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=100,debug=True)
    



