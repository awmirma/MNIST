import tensorflow as tf
import glob

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics=['accuracy'])



models = []
# weights_epoch = 0

for i in range(5):
    m = model.fit(x_train,y_train,epochs=i)
    models.append(m)

histories = glob.glob("dropout_0.2/history/*model_{}*".format(i))

histories = sorted(histories)
print(histories)
# model.fit(x_train,y_train,epochs=)
# model.save(f'handwrittenfinal.model')
# loss , accuracy = model.evaluate(x_test,y_test)
# print(loss)
# print(accuracy)
