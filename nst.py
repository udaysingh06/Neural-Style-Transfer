import tensorflow as tf
from tensorflow.keras.applications import vgg19
from utils import load_and_process, deprocess, save_image
import numpy as np

# Paths
CONTENT_PATH = 'content_images/your_content.jpg'
STYLE_PATH   = 'style_images/your_style.jpg'
OUTPUT_PATH  = 'output/generated.jpg'

# Load images
content = load_and_process(CONTENT_PATH)
style = load_and_process(STYLE_PATH)

# Feature extraction model
style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
content_layers = ['block5_conv2']
all_layers = style_layers + content_layers

vgg = vgg19.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False
outputs = [vgg.get_layer(name).output for name in all_layers]
feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=outputs)

# Extract features
style_features = feature_extractor(style)[:-1]
content_feature = feature_extractor(content)[-1]

# Initialize target as variable
target = tf.Variable(content, dtype=tf.float32)

# Gram matrix function
def gram_matrix(tensor):
    t = tf.transpose(tensor, perm=[0,3,1,2])
    features = tf.reshape(t, [tf.shape(t)[0], tf.shape(t)[1], -1])
    Gram = tf.matmul(features, features, transpose_b=True)
    return Gram

# Weights
style_weight = 1e-2
content_weight = 1e4

optimizer = tf.optimizers.Adam(learning_rate=0.005)

@tf.function()
def train_step():
    with tf.GradientTape() as tape:
        outputs = feature_extractor(target)
        style_outs = outputs[:-1]
        content_out = outputs[-1]

        s_loss = tf.add_n([
            tf.reduce_mean((gram_matrix(style_outs[i]) - gram_matrix(style_features[i]))**2)
            for i in range(len(style_layers))
        ]) / len(style_layers)

        c_loss = tf.reduce_mean((content_out - content_feature)**2)
        loss = style_weight * s_loss + content_weight * c_loss

    grads = tape.gradient(loss, target)
    optimizer.apply_gradients([(grads, target)])
    target.assign(tf.clip_by_value(target, -103.939, 255.0-123.68))

# Run optimization
steps = 1000
for i in range(steps):
    train_step()
    if i % 100 == 0:
        print(f"Step {i}/{steps}")

# Save output
final_img = deprocess(target.numpy())
save_image(final_img, OUTPUT_PATH)
print("Saved at", OUTPUT_PATH)