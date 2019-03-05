import tensorflow as tf



from keras.datasets import cifar10
from keras.utils import np_utils


(x_train,y_train),(x_test,y_test)=cifar10.load_data()


print(y_train)

y_test

y_train_onehot_encoded=np_utils.to_categorical(y_train,10)
y_test_one_hot_encoded=np_utils.to_categorical(y_test,10)


y_train_onehot_encoded

y_test_one_hot_encoded

y_train_onehot_encoded.shape[1]

x_train.shape

y_train.shape


x_test.shape


y_test.shape

num_classes=10
img_size=x_train.shape[1]
num_channels= x_train.shape[3]
img_size_flat=x_train.shape[1]*x_train.shape[2]*x_train.shape[3]

print("num_classes:",num_classes)
print("num_channels:",num_channels)
print("img_size:",img_size)
print("img_size_flat:",img_size_flat)

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import SGD

labels =  ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
print(labels[0])

num_pixels=x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
print(num_pixels)

import matplotlib.pyplot as plt
plt.imshow(x_train[0])


x_train

print(labels[int(y_train[0])])
print(int(y_train[0]))
ps=y_train
print(ps.shape)

#labels_train=tf.one_hot(y_train, depth=10,
        #   on_value=1.0, off_value=0.0,
         #              axis=-1)
#labels_train=tf.reshape(labels_train,shape=[50000,10])
#print(labels_train.shape)
#labels_train=tf.cast(labels_train,tf.int32)
#print(labels_train)
#print(len(y_train))

"""
indices = [0, 2, -1, 1]
depth = 3
print(tf.one_hot(indices, depth,
           on_value=1.0, off_value=0.0,
           axis=-1)) 
"""

#print(labels_train)

fig=plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.imshow(x_train[i])
  a=labels[int(y_train[i])]
  plt.title("True:{}".format(a))
  plt.xticks([])
  plt.yticks([])
 

print("labels:{0}".format(labels[int(y_train[0])]))

def plot_images(images,cls_true,cls_pred=None):
  assert len(images)==len(cls_true)==9
  fig,axes=plt.subplots(3,3)
  fig.subplots_adjust(hspace=0.3,wspace=0.3)
  
  for i,ax in enumerate(axes.flat):
    ax.imshow(images[i,:,:,:],interpolation='nearest')
    cls_true_name=labels[int(cls_true[i])]
    
    if cls_pred is None:
      xlabel="True:{0}".format(cls_true_name)
      
    else:
      class_pred_name=labels[int(cls_pred[i])]
      xlabel="True:{0} predicted:{1}".format(cls_true_name,cls_pred_name)
       
    ax.set_xlabel(xlabel)
    ax.set_xticks([])
    ax.set_yticks([])
  plt.show()

images=x_train[0:9]
cls_true=y_train[0:9]
plot_images(images=images,cls_true=cls_true)

images=x_test[0:9]
cls_true=y_test[0:9]
plot_images(images=images,cls_true=cls_true)

#placeholders
x=tf.placeholder(tf.float32,shape=[None,img_size,img_size,num_channels],name='x')
x_image=tf.reshape(x,[-1,img_size,img_size,num_channels])
y_true=tf.placeholder(tf.float32,shape=[None,num_classes],name='y_true')
#argmax
y_true_cls=tf.argmax(y_true,axis=1)

#preprocessing
x_train=x_train/255
x_test=x_test/255

def new_weights(shape):
  return tf.Variable(tf.truncated_normal(shape,stddev=0.05,dtype=tf.float32))

def new_biases(length):
  return tf.Variable(tf.constant(0.05,shape=[length]))


#convolution layer 1
filter_size1=3
num_filters1=64
#convolution layer 2
filter_size2=3
num_filters2=128
#convolution layer 3
filter_size3=5
num_filters3=256
#convolution layer 4
filter_size4=5
num_filters4=512
#fully_connected layer1
fc_layer_size1=128
#fully_connected layer1
fc_layer_size2=256
#fully_connected layer1
fc_layer_size3=512
#fully_connected layer1
fc_layer_size4=1024


def new_conv_layer(input,num_input_channels,filter_size,num_filters,use_pooling=True):#,dropout_layer=False):
  #shape of filters
  shape=[filter_size,filter_size,num_input_channels,num_filters]
  
  weights=new_weights(shape)
  biases=new_biases(length=num_filters)
  
  layer=tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME')
  layer=layer+biases
  
  layer=tf.nn.relu(layer)

  if use_pooling:
    layer=tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
  #if dropout_layer:
   # layer=tf.nn.dropout(layer,keep_prob=0.25)
    
  print(layer.get_shape())  
    
  return layer,weights

def flatten_layer(layer):
  layer_shape=layer.get_shape()
  
  #layer_shape=[num_input_chanels,img_height,img_width,num_output_channels]
  
  print("layer_shape=",layer_shape)
  
  #num_fateures=img_height*img_width*num_output_channels 
  #using tensor flow fn to calculate this
    
  num_features=layer_shape[1:4].num_elements()
  
  layer_flat=tf.reshape(layer,[-1,num_features])
  
  print("num_features=",num_features)
  
  return layer_flat,num_features

def fc_layer(input,num_inputs,num_outputs,use_relu=True):
  
  weights=new_weights([num_inputs,num_outputs])
  biases=new_biases(length=num_outputs)
  
  layer=tf.matmul(input,weights)+biases
  
  if use_relu:
    layer=tf.nn.relu(layer)
    
  return layer

y_true_cls.shape

#conv1
layer_conv1,weights_conv1 =new_conv_layer(input=x_image, num_input_channels=num_channels,filter_size=filter_size1,num_filters=num_filters1,use_pooling=False)


print("layer_conv1=",layer_conv1)
print("weights_conv1=",weights_conv1)

#conv2
layer_conv2,weights_conv2= new_conv_layer(input=layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)


print("layer_conv2=",layer_conv2)
print("weights_conv2=",weights_conv2)

#conv3
#layer_conv3,weights_conv3= new_conv_layer(input=layer_conv2, num_input_channels=num_filters2, filter_size=filter_size3, num_filters=num_filters3, use_pooling=True)


#print("layer_conv3=",layer_conv3)
#print("weights_conv3=",weights_conv3)

#conv4
layer_conv4,weights_conv4= new_conv_layer(input=layer_conv2, num_input_channels=num_filters2, filter_size=filter_size4, num_filters=num_filters4, use_pooling=True)

print("layer_conv4=",layer_conv4)
print("weights_conv4=",weights_conv4)

#layer_flat, num_features=flatten_layer(layer_conv2)
layer_flat, num_features=flatten_layer(layer_conv4)

print("layer_flat=",layer_flat)
print("num_features=",num_features)

#fc1
layer_fc1=fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=fc_layer_size1,use_relu=True)

print("layer_fc1=",layer_fc1)

#fc2
layer_fc2=fc_layer(input=layer_fc1,num_inputs=fc_layer_size1,num_outputs=fc_layer_size2, use_relu=True)

print("layer_fc2=",layer_fc2)

#fc3
#layer_fc3=fc_layer(input=layer_fc2, num_inputs=fc_layer_size2, num_outputs=fc_layer_size3, use_relu=True)

#print("layer_fc3=",layer_fc3)

#fc4
layer_fc4=fc_layer(input=layer_fc2, num_inputs=fc_layer_size2, num_outputs=fc_layer_size4, use_relu=True)

print("layer_fc4=",layer_fc4)

#final lyaer fuly connected one
layer_fc5=fc_layer(input=layer_fc4, num_inputs=fc_layer_size4, num_outputs=num_classes, use_relu=False)

print("layer_fc5=",layer_fc5)

y_pred=tf.nn.softmax(layer_fc5)

y_pred_cls=tf.argmax(y_pred,axis=1)


#cross_entropy
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=y_true)

#loss
cost=tf.reduce_mean(cross_entropy)

#optimizer
#optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
optimizer=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

#correct_prediction
correct_prediction=tf.equal(y_pred_cls,y_true_cls)

#accuracy
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess=tf.Session()
init=sess.run(tf.global_variables_initializer())


train_batch_size=69

import numpy as np
def random_batch():
    # Number of images in the training-set.
    num_images = len(x_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = x_train[idx, :, :, :]
    y_batch = y_train_onehot_encoded[idx, :]

    return x_batch, y_batch

#total_iterations = 0

def optimize(num_iterations):
  #global total_iterations
  #for i in range(total_iterations, total_iterations+num_iterations):
   for i in range(num_iterations):
      x_true_batch,y_true_batch=random_batch()
      feed_dict_train={x_image:x_true_batch,y_true:y_true_batch}
      sess.run(optimizer,feed_dict=feed_dict_train)
      loss=sess.run(cost,feed_dict=feed_dict_train)
      acc=sess.run(accuracy,feed_dict=feed_dict_train)
      if(i%100==0):
        print("optimization iteration:{0:>6}\t training accuracy:{1:>6.1%}\t loss:{2}".format(i+1,acc,loss))
        #print("loss:{}".format(cost))
  #total_iterations += num_iterations

optimize(5000)

batch_size=256

def print_test_accuray(images,labels,cls_true):
    num_images=len(images)
    cls_pred=np.zeros(shape=num_images,dtype=np.int)
    #next batch sarting images
    i=0
    while(i<num_images):
        #ending of next batch images
        j=min(i+batch_size,num_images)
        feed_dict1={x:images[i:j:],y_true:labels[i:j:]}
        cls_pred[i:j]=sess.run(y_pred_cls,feed_dict=feed_dict1)
        #set start index of next batch =end of prev batch
        i=j
    
    correct=(y_pred_cls==cls_true)
    incorrect=(y_pred_cls!=cls_true)
    correct_sum=correct.sum()
    acc=float(correct_sum)/num_images
    
    print("accuracy on test set:{0:.1%}  ({1}/{2})".format(acc,coorect_sum,num_images)) 
    print("correct prediction::")
    plot_examples(cls_pred=cls_pred,uwant=correct)
    print("incorrect prediction::")
    plot_examples(cls_pred=cls_pred,uwant=incorrect)

def plot_examples(cls_pred,uwant):
    images=x_test[uwant]
    cls_pred=cls_pred[uwant]
    cls_true=y_test[uwant]
    
    plot_images(images=images[0:9],
                cls_true=y_test[0:9],
               cs_pred=cls_pred[0:9])

#def predict_cls_test():
 #   return pred_cls(images=x_test,labels=y_test_onehot_encoded,cls_true=y_test)

pint_test_accuracy(images=x_test,labels=y_test_one_hot_encoded,cls_true=y_test)

#plot weights
import math
def plot_conv_weights(weights, input_channel=0):
   
    w = sess.run(weights)

    # lowest and highest values for the weights.to corect colour intensity across images so they can be compared
  
    w_min = np.min(w)
    w_max = np.max(w)

    # No. of filters used in the convlayer
    num_filters = w.shape[3]

    # Number of grids to plot
    # Rounded-up, square-root of the number of filters
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights
        if i<num_filters:
            # Get the weights for the i filter of the input channel
           
            img = w[:, :, input_channel, i]

            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest',cmap='gray')# cmap='seismic')
        
        ax.set_xticks([])
        ax.set_yticks([])
   
    plt.show()

#plot convlayer
def plot_conv_layer(layer, image):
   
    feed_dict0 = {x: [image]}

    values = sess.run(layer, feed_dict=feed_dict0)

    num_filters = values.shape[3]

    num_grids = math.ceil(math.sqrt(num_filters))
   
    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='jet')
     
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()

plot_conv_weights(weights=weights_conv1)

def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='jet')
    plt.axis('off')

    plt.show()

image1 = x_test[0]
plot_image(image1)

plot_conv_layer(layer=layer_conv1, image=image1)

#y_pred_proba=estimator.predict_proba(x_test)

from sklearn.metrics import classification_report,accuracy_score,f1_score,confusion_matrix

print(classification_report(y_true=y_test,y_pred=cls_pred))
print(confusion_matrix(y_true=y_test,y_pred=cls_pred))
print(accuracy_score(y_true=y_test,y_pred=cls_pred))
print(f1_score(y_true=y_test,y_pred=cls_pred))
