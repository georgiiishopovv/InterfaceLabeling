import keras._tf_keras.keras
from keras._tf_keras.keras import datasets
from keras._tf_keras.keras.datasets import fashion_mnist
import numpy as np
from  keras._tf_keras.keras.utils import to_categorical
from t import *
from image_processing import *
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras._tf_keras.keras.preprocessing.image import img_to_array, load_img
import cv2

def main():
    image_paths = [
    "C:/Users/Georgi/Desktop/InternshipSpain2024/car1.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/car2.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/car3.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/car4.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/car5.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/car6.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/car7.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/car8.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/plane1.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/plane3.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/plane6.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/plane7.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/plane8.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/plane9.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/plane10.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/bike1.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/bike2.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/bike3.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/bike4.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/bike5.jpg",
    "C:/Users/Georgi/Desktop/InternshipSpain2024/bike6.jpg"
    ]
    target_size = (28, 28)

    #print(f"Second image pixel values range: {images_np[1].min()} - {images_np[1].max()}")

    labels = ['car', 'car', 'car', 'car', 'car', 'car', 'car', 'car', 'plane', 'plane','plane', 'plane', 'plane', 'plane', 'plane', 'bike', 'bike', 'bike', 'bike', 'bike', 'bike']

    def augment_images(image_paths, labels, num_augmented_images, target_size):

        datagen = ImageDataGenerator(
        rotation_range=10, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
        )

        augmented_images = []
        augmented_labels = []
        for image_path, label in zip(image_paths, labels):
            img = load_img(image_path, target_size=target_size)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)  #Reshape to (1, height, width, channels)

            #Generate augmented images
            count = 0
            for batch in datagen.flow(x, batch_size=1):
                augmented_images.append(batch[0])  
                augmented_labels.append(label)   
                count += 1
                if count >= num_augmented_images:
                    break
        return augmented_images, augmented_labels
            
    a_images, a_labels = augment_images(image_paths, labels, 10, target_size)
    #a_images = [load_and_preprocess_image(path, target_size) for path in a_images]
    grayscale = False
    for i in range(len(a_images)):
        a_images[i] = cv2.resize(a_images[i], target_size)
        a_images[i] = np.array(a_images[i]) / 255.0
        if grayscale:
            pil_image1 = Image.fromarray((a_images[i] * 255).astype(np.uint8))  # Convert NumPy array to PIL Image
            pil_image1 = pil_image1.convert('L')  # Convert to grayscale
            a_images[i] = np.array(pil_image1) / 255.0  # Convert back to NumPy array and normalize
            
    #images = [load_and_preprocess_image(path, target_size) for path in image_paths]
    #images_np = np.array(images)
    a_images_np = np.array(a_images)
    print("Shape of the new array:", a_images_np.shape)
    #combined_array = np.concatenate((images_np, a_images_np), axis=0)

    label_map = {'car': 0, 'plane': 1, 'bike': 2}
    # labels_np = np.array([label_map[label] for label in labels])
    # labels_np = labeling(labels_np)
    a_labels_np = np.array([label_map[label] for label in a_labels])
    a_labels_np = labeling(a_labels_np)
    # labels_np = np.array(labels_np)
    a_labels_np = np.array(a_labels_np)
    #combined_labels = np.concatenate((labels_np, a_labels_np), axis=0)
    #combined_array = np.array(combined_array)
    #combined_labels = np.array(combined_labels)

    # print("====================")
    # print(combined_array)
    # print("====================")
    # print(combined_labels)

    indices = np.arange(len(a_images_np))
    np.random.shuffle(indices)
    a_images_np = a_images_np[indices]
    a_labels_np = a_labels_np[indices]
  
    # Network initialization
    conv = Conv(a_images_np[1].shape, filters_size=1, filters_count=8)
    pool = MaxPool(pool_size=2)
    full = CNN(input_size=14 * 14 * 8, output_size=3)

    train(a_images_np, a_labels_np, conv, pool, full)
    
    image_paths = [
        "C:/Users/Georgi/Desktop/InternshipSpain2024/test_car1.jpg",
        "C:/Users/Georgi/Desktop/InternshipSpain2024/test_car2.jpg",
        "C:/Users/Georgi/Desktop/InternshipSpain2024/bike1.jpg",
        "C:/Users/Georgi/Desktop/InternshipSpain2024/plane2.jpg",
    ]

    test_images = [load_and_preprocess_image(path, target_size) for path in image_paths]
    if grayscale:
            pil_image1 = Image.fromarray((test_images[i] * 255).astype(np.uint8))  # Convert NumPy array to PIL Image
            pil_image1 = pil_image1.convert('L')  # Convert to grayscale
            test_images[i] = np.array(pil_image1) / 255.0  # Convert back to NumPy array and normalize
    test_images_np = np.array(test_images)

    # Split data into train and test sets
    predictions = []
    for data in test_images_np:
        pred = predict(data, conv, pool, full)  # Assuming you have a predict function
        predictions.append(pred)

    predicted_labels = []
    for pred in predictions:
        max_index = np.argmax(pred)
        label = list(label_map.keys())[max_index]  # Map index back to original label
        predicted_labels.append(label)

    for i, label in enumerate(predicted_labels):
        print(f"Test image {i + 1}: Predicted label - {label}")

if __name__ == "__main__":
    main()

#=======================
# print("My: ", images_np[0])
# print("My label: ", labels_np[0])

# (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# X_train = train_images[:500] / 255.0
# y_train = train_labels[:500]

# X_test = train_images[500:1000] / 255.0
# y_test = train_labels[500:1000]

# y_train = to_categorical(y_train)
# print("Mnist: ", X_train[1])
# print("Mnist label", y_train[1])
#print(X_train.shape)

# y_test = to_categorical(y_test)
# #print("Initial labels", y_train)

# conv = Conv(X_train[0].shape, 6, 1)
# pool = MaxPool(2)
# full = CNN(121, 10)

# train(X_train, y_train, conv, pool, full)
