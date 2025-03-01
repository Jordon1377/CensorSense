import tensorflow as tf
from tensorflow.keras import layers, models

def create_pnet(input_shape=(12, 12, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    
    # First Conv layer
    x = layers.Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(inputs)
    x = layers.PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = layers.MaxPooling2D((2, 2), strides=2, padding='same', name='pool1')(x)
    
    # Second Conv layer
    x = layers.Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = layers.PReLU(shared_axes=[1, 2], name='PReLU2')(x)
    
    # Third Conv layer
    x = layers.Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = layers.PReLU(shared_axes=[1, 2], name='PReLU3')(x)
    
    # Face classification branch
    face_class = layers.Conv2D(2, (1, 1), strides=1, padding='valid', name='conv4-1')(x)
    face_class = layers.Reshape((2,), name="face_class_reshaped")(face_class)  # Flatten output
    face_class = layers.Softmax(name='face_class')(face_class)
    
    # Bounding box regression branch
    bbox_reg = layers.Conv2D(4, (1, 1), strides=1, padding='valid', name='conv4-2')(x)
    
    # Facial landmark localization branch (not sure if needed rn)
    landmark_reg = layers.Conv2D(10, (1, 1), strides=1, padding='valid', name='conv4-3')(x)
    
    model = models.Model(inputs, [face_class, bbox_reg, landmark_reg])
    
    return model

pnet_model = create_pnet()
pnet_model.summary()
