import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import logging
import utils


# Define a loss function. We only care about the first of the two output layers.
def loss(model, x, y, loss_object, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y1_, y2_ = model(x, training=training)
    assert y1_.shape == y2_.shape
    return loss_object(y_true=y, y_pred=y1_)


# Calculate the gradient
def grad(model, inputs, targets, loss_object):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, loss_object, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def test_current_accuracy(model, test_dataset):
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()

    for (x, y) in test_dataset:
        y1_, y2_ = model(x, training=False)
        test_accuracy(y1_, y)

    logging.info(f'Test set accuracy: {test_accuracy.result():.3%}')


def train_experimental_model(train_images, train_labels, test_images, test_labels):
    logging.info('Training network with two output layers...')
    model = utils.create_model('VGG_8_simple_with_two_outputs')

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(64)

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = SGD(lr=0.001, momentum=0.9)

    train_loss_results = []
    train_accuracy_results = []
    num_epochs = 200

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        for x, y in train_dataset:
            # Optimize the model
            loss_value, grads = grad(model, x, y, loss_object)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss

            y1_, y2_ = model(x, training=True)
            epoch_accuracy.update_state(y, y1_)

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        logging.info(f'Epoch {epoch:03d}: Loss: {epoch_loss_avg.result():.3f}, '
                     f'Accuracy: {epoch_accuracy.result():.3%}')

        # Do a test after some epochs, and also at the end
        if epoch % 10 == 0 or epoch == num_epochs-1:
            test_current_accuracy(model, test_dataset)
