import tensorflow as tf
import numpy as np
import PIL.Image
import time
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(e)
# ==========================================
# 1. CONFIGURATION (Edit these paths)
# ==========================================
CONTENT_PATH = 'Content_Industry1.jpeg'  
STYLE_PATH = 'Style6.jpeg'       
OUTPUT_FILENAME = 'Generated_Image1.jpg'

# IMAGE SETTINGS
IMG_MAX_DIM = 1000

# WEIGHTS (Balanced to prevent "Grey Blob")
CONTENT_WEIGHT = 1e4  
STYLE_WEIGHT = 1e4   
TV_WEIGHT = 1e3       

# TRAINING SETTINGS
EPOCHS = 100
STEPS_PER_EPOCH = 10

# ==========================================
# 2. SETUP & GPU CHECK
# ==========================================
print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU Detected: {gpus}")
else:
    print("WARNING: No GPU detected. Running on CPU (will be slow).")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)
def load_img(path_to_img):
        with tf.device('/CPU:0'):
            max_dim = IMG_MAX_DIM
            img = tf.io.read_file(path_to_img)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)

            shape = tf.cast(tf.shape(img)[:-1], tf.float32)
            long_dim = max(shape)
            scale = max_dim / long_dim

            new_shape = tf.cast(shape * scale, tf.int32)
            img = tf.image.resize(img, new_shape)
            img = img[tf.newaxis, :]
            return img

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    
    num_locations = tf.cast(input_shape[1]*input_shape[2]*input_shape[3], tf.float32)
    return result / num_locations

# ==========================================
# 4. MODEL DEFINITION
# ==========================================
def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        
        content_dict = {content_name: value
                        for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value
                      for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
# Load Images
print(f"Loading images...\nContent: {CONTENT_PATH}\nStyle: {STYLE_PATH}")
try:
    content_image = load_img(CONTENT_PATH)
    style_image = load_img(STYLE_PATH)
except Exception as e:
    print(f"Error loading images: {e}")
    print("Please check filenames in the CONFIGURATION section.")
    exit()

# Define Layers
content_layers = ['block4_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Initialize Model & Targets
extractor = StyleContentModel(style_layers, content_layers)
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# Initialize Variable to Train
#image = tf.Variable(content_image)
print("Using Random Noise Initialization!")
random_noise = tf.random.uniform(tf.shape(content_image), minval=0.0, maxval=1.0)
image = tf.Variable(random_noise)

# Define Loss Function
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= STYLE_WEIGHT / len(style_outputs)

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= CONTENT_WEIGHT / len(content_outputs)
    
    loss = style_loss + content_loss
    return loss

# Optimizer
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
#SGD with Momentum
#opt = tf.optimizers.SGD(learning_rate=0.05, momentum=0.9)
#RMSprop
#opt = tf.optimizers.RMSprop(learning_rate=0.02)

# Training Step (Includes TV Loss)
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        
        # Add Total Variation (Smoothing) Loss
        loss += TV_WEIGHT * tf.cast(tf.image.total_variation(image)[0], tf.float32)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    return loss

# ==========================================
# 6. TRAINING LOOP
# ==========================================
start_time = time.time()
print(f"\nStarting Training for {EPOCHS} epochs...")
print(f"Configuration: Content W={CONTENT_WEIGHT}, Style W={STYLE_WEIGHT}, TV W={TV_WEIGHT}")

for n in range(EPOCHS):
    for m in range(STEPS_PER_EPOCH):
        loss = train_step(image)
    
    # Print progress every 10 epochs
    if (n+1) % 10 == 0:
        print(f"Epoch {n+1}/{EPOCHS} complete. Loss: {loss.numpy():.2e}")

end_time = time.time()
print(f"\nTraining finished in {end_time - start_time:.1f} seconds")

# ==========================================
# 7. SAVE RESULT
# ==========================================
file_name = OUTPUT_FILENAME
tensor_to_image(image).save(file_name)
print(f"Success! Image saved as: {os.path.abspath(file_name)}")
