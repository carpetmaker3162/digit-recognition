'''
train on MNIST dataset

parameters saved in model.weights.h5 file
'''

import tensorflow as tf

def main():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # normalize pixel values
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(48, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy: {}'.format(test_acc))

    model.save('model.weights.h5')

if __name__ == '__main__':
    main()