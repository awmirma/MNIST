import os
import cv2

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)


model = tf.keras.models.load_model('handwritten.model')


# loss , accuracy = model.evaluate(x_test,y_test)

# print(loss)
# print(accuracy)

image_number = 271



# /home/awmirma/Documents/AI/practice/MNIST/mnist_png/testing/0/3.png
for i in range(10) :
    while os.path.isfile(f"mnist_png/testing/0/{image_number}.png"):
        print("ure in")
        try : 
            img = cv2.imread(f"mnist_png/testing/0/{image_number}.png")[:,:,0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            
            # print (f"this number is probably a {np.argmax(prediction)} --> {image_number}")
            print("prediction\tactual number")
            print(f"{np.argmax(prediction)}\t\t{i}")
            # plt.imshow(img[0], cmap = plt.cm.binary)
            # plt.show()
        except :
            print("error!!")

        finally :
            image_number += 1 
