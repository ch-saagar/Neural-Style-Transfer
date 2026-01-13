# Neural-Style-Transfer
Neural Style Transfer is a deep learning technique that takes two images, a Content Image and a Style Image and blends them together.

The specific objectives are:

1.Generate a New Image: Create a third image (the "Generated Image" x) that keeps the structure/shapes of the Content Image but adopts the colors and textures of the Style Image.

2.Use a Pre-trained Network: Instead of training a new neural network from scratch, we will use VGG19, a powerful pre-trained Convolutional Neural Network (CNN) that already knows how to recognize features in images.

3.Optimize Pixels, Not Weights: Unlike standard machine learning where we update the network's parameters (weights) to learn a task, here we freeze the network and update the pixels of the generated image until it minimizes a specific loss function.

4.Master Advanced Keras Functional API, Subclassing API, and custom training loops with automatic differentiation.

