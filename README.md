# Dog vs Cat Classification Using Transfer Learning (MobileNet)

This project is designed for learning and practicing deep learning techniques, specifically focusing on transfer learning and image recognition. It demonstrates how to use a pretrained MobileNet model to classify images of cats and dogs, and covers essential steps in image preprocessing, model building, training, and evaluation.

---


## 1. Dataset Introduction

This project uses the [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) dataset from Kaggle, which contains 12,500 images each of cats and dogs. The dataset is split into training and testing sets using the following code:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

- `test_size=0.2` means 20% of the data is used for testing and 80% for training.
- `random_state=42` ensures reproducibility.

---


## 2. Image Processing

The original images are processed as follows:

- All images are resized to 224x224 pixels
- Only 1000 cat and 1000 dog images are selected
- Images are converted to RGB format and saved to a new folder

Relevant code:

```python
from PIL import Image
import os

os.makedirs('/content/resized_images', exist_ok=True)
origin_folder = '/content/train'
resized_folder = '/content/resized_images'

dog_count = 0
cat_count = 0
target_count = 1000

for filename in os.listdir(origin_folder):
  if dog_count < target_count and filename.startswith('dog'):
    img_path = origin_folder + '/' + filename
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    newImgPath = resized_folder + '/' + filename
    img.save(newImgPath)
    dog_count += 1

  elif cat_count < target_count and filename.startswith('cat'):
    img_path = origin_folder + '/' + filename
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    newImgPath = resized_folder + '/' + filename
    img.save(newImgPath)
    cat_count += 1

  if dog_count == target_count and cat_count == target_count:
    break
```


**Explanation**:  
- This code iterates through the original image folder, processes the first 1000 cat and 1000 dog images to 224x224 RGB format, and saves them to the `/content/resized_images` folder.

---


## 3. Model Introduction

This project uses transfer learning with MobileNetV3 as a feature extractor (with frozen weights), and only the final dense layer is trained.

Key code:

```python
import tf_keras
import tensorflow_hub as hub

mobilenet_model = 'https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5'
pretrained_model = hub.KerasLayer(mobilenet_model, input_shape=(224, 224, 3), trainable=False)
num_of_classes = 2

model = tf_keras.Sequential([
    pretrained_model,
    tf_keras.layers.Dense(num_of_classes)
])

model.summary()

model.compile(
    optimizer='adam',
    loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc']
)
history = model.fit(X_train_scaled, Y_train, epochs=5)
```


**Training Process Explanation**:  
- MobileNetV3 pretrained model is used as a feature extractor with frozen weights.
- A dense layer is added for 2-class (cat/dog) output.
- The loss function is `SparseCategoricalCrossentropy`, and the optimizer is `adam`.
- Trained for 5 epochs.

**Training Log and Test Accuracy**:

```
Epoch 1/5
50/50 [==============================] - 62s 1s/step - loss: 0.2050 - acc: 0.9162
Epoch 2/5
50/50 [==============================] - 52s 1s/step - loss: 0.0784 - acc: 0.9731
Epoch 3/5
50/50 [==============================] - 50s 1s/step - loss: 0.0604 - acc: 0.9812
Epoch 4/5
50/50 [==============================] - 53s 1s/step - loss: 0.0475 - acc: 0.9837
Epoch 5/5
50/50 [==============================] - 63s 1s/step - loss: 0.0386 - acc: 0.9906

13/13 [==============================] - 14s 961ms/step - loss: 0.0625 - acc: 0.9825
Test loss: 0.06254494190216064
Test accuracy: 0.9825000166893005
```

**Result Visualization**:

<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/db97f22a-27c0-4711-8ae6-7560c06615c0" />


---


## 4. Summary and Training Effect Analysis

- With transfer learning, the model achieves high accuracy in just a few epochs, showing that MobileNetV3 features are very effective for cat vs. dog classification.
- The training accuracy and loss curves show good convergence and no obvious overfitting.
- The test set accuracy reaches 98% , indicating strong generalization.
- For further improvement, consider data augmentation, adjusting the model structure, or increasing the number of epochs.

---


## 5. Reference

- [Kaggle Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats)
- [TensorFlow Hub: MobileNetV3](https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5)
- [tf.keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)

---
> **All model code, logs, and result plots are based on the original notebook and project files.  
> For any questions or suggestions, please open an issue.**
