import numpy as np
from scipy.signal import correlate2d
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from PIL import Image

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Conv:
    def __init__(self, input_pixels, filters_size, filters_count):
        self.input_pixels = input_pixels
        self.kernel_size = filters_size
        self.filters_count = filters_count
        input_h = input_pixels[0]
        input_w = input_pixels[1]
        
        self.kernel_shape = (filters_count, self.kernel_size, self.kernel_size)
        self.end_shape = (filters_count, input_h - self.kernel_size + 1, input_w - self.kernel_size + 1)
        
        self.filters = np.random.rand(*self.kernel_shape)
        self.biases = np.random.rand(*self.end_shape)
        
    def forward(self, input):
        
        num_dimensions = len(input.shape)
        
        if num_dimensions == 3:
            input = np.mean(input, axis=2)
        
        self.input = input
        temp_output = np.zeros(self.end_shape)
        if not isinstance(self.input, np.ndarray):
            self.input = np.array(self.input)
        for i in range(len(self.filters)):
            temp_output[i] = correlate2d(self.input, self.filters[i], mode="valid")
            #temp_output[i] += self.biases[i]
            
        output = np.maximum(temp_output, 0)
        #print("CONV", output.shape)
        return output
        
    def backward(self, learning_rate, der_output):
        der_input = np.zeros_like(self.input)
        der_filters = np.zeros_like(self.filters)  
        
        for i in range(self.filters_count):
            der_input += correlate2d(der_output[i], self.filters[i], mode="full")
            der_filters[i] = correlate2d(self.input, der_output[i], mode="valid")
                        
        self.filters -= learning_rate * der_filters
        self.biases -= learning_rate * der_output

        return der_input
    
class MaxPool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        
    def forward(self, input):
        self.input = input
        if not isinstance(self.input, np.ndarray):
            self.input = np.array(self.input)
        self.num = self.input.shape[0]
        self.input_h = self.input.shape[1]
        self.input_w = self.input.shape[2]
        
        self.output_h = self.input_h // self.pool_size
        self.output_w = self.input_w // self.pool_size
        
        self.output = (self.num, self.output_h, self.output_w)
        self.output = np.zeros(self.output)      
        
        for a in range(self.num):
            for b in range(self.output_h):
                for c in range(self.output_w):
                    start1 = b * self.pool_size
                    start2 = c * self.pool_size
                    
                    end1 = start1 + self.pool_size
                    end2 = start2 + self.pool_size
                    
                    patch = self.input[a, start1:end1, start2:end2]
                    self.output[a, b, c] = np.max(patch)
                 
        return self.output
    
    def backward(self, der_output):
        der_input = np.zeros_like(self.input)
        for a in range(self.num):
            for b in range(self.output_h):
                for c in range(self.output_w):
                    start1 = b * self.pool_size
                    start2 = c * self.pool_size
                    
                    end1 = start1 + self.pool_size
                    end2 = start2 + self.pool_size
                    
                    patch = self.input[a, start1:end1, start2:end2]
                    temp = (patch == np.max(patch))
                    der_input[a, start1:end1, start2:end2] = der_output[a, b, c] * temp
                    
        return der_input

class CNN:
    # def __init__(self, input_size, output_size):
    #     self.input_size = input_size
    #     self.output_size = output_size
    #     self.weights = np.random.randn(self.output_size, self.input_size)
    #     self.weights2 = np.random.randn(self.output_size, self.input_size)
    #     self.biases = np.random.randn(self.output_size, 1)

    # def softmax(self, input):
    #     shift = input - np.max(input)
    #     exp = np.exp(shift)
    #     sum = np.sum(exp)
    #     #print("------------")
    #     #print(sum)
    #     sum1 = np.sum(exp, axis=0)
    #     #print(sum1)
    #     probability = exp / sum1
    #     return probability

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        
        self.biases = np.random.randn(output_size, 1)

    def softmax(self, input):
        shift = input - np.max(input)
        exp = np.exp(shift)
        return exp / np.sum(exp, axis=0, keepdims=True)
    
    def forward(self, input):
        self.input = input
        self.flattened_input = input.flatten().reshape(1, -1)
        z = np.dot(self.weights, self.flattened_input.T) + self.biases
        self.output = self.softmax(z)
        return self.output
    
    def softmax_derivative(self, input):
        der = np.diagflat(input) - np.dot(input, input.T)
        return der

    # def forward(self, input):
    #     self.input = input
    #     if not isinstance(input, np.ndarray):
    #         input = np.array(input)
    #     #print("----------")
    #     #print(input.shape)
    #     self.flattened_input = input.flatten().reshape(1, -1)
        
        # #print("----------")
        # #print(self.flattened_input.shape)
        # z = np.dot(self.weights, self.flattened_input.T) + self.biases
        # #print("----------")
        # #print(z.shape)
        # self.output = self.softmax(z)
        # return self.output
    
    def backward(self, learning_rate, der_output):
        #print("X", self.output.shape)
        #print("N", self.softmax_derivative(self.output).shape)
        #print("Z", der_output.shape)
        der_dz = np.dot(self.softmax_derivative(self.output), der_output)
        #print("Ji", der_dz.shape)
        der_biases = der_dz
        der_weights = np.dot(der_dz, self.flattened_input)
        
        der_input = np.dot(self.weights.T, der_dz)
        if not isinstance(der_input, np.ndarray):
            der_input = np.array(der_input)
        if not isinstance(self.input, np.ndarray):
            self.input = np.array(self.input)
        der_input = der_input.reshape(self.input.shape)
        
        self.weights -= der_weights * learning_rate
        self.biases -= der_biases * learning_rate
        
        return der_input  

def cross_entropy_loss(predictions, expected_labels):
    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    #print("Predictions", predictions.shape)
    #print("Expected labels", expected_labels.shape)
    loss = -np.sum(expected_labels * np.log(predictions)) / expected_labels.shape[0]
    return loss
    
def cross_entropy_loss_backward(labels, predicted_probabilities):
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    classes = labels.shape[0]
    derivative = -labels / (predicted_probabilities + 1e-7) / classes
    return derivative

def train(train_array1, labels, conv: Conv, pool: MaxPool, cnn: CNN, learning_rate=0.01, epochs=20):
    original_labels = np.copy(labels)
    
    for e in range(epochs):
        total_loss = 0.0
        correct_predictions = 0

        for passes in range(len(train_array1)):
            convolution_result = conv.forward(train_array1[passes])
            pool_result = pool.forward(convolution_result)
            cnn_result = cnn.forward(pool_result)
            #print(passes, "/", len(train_array1))

            loss = cross_entropy_loss(cnn_result.flatten(), labels[passes])
            total_loss += loss

            one_hot_pred = np.zeros_like(cnn_result)
            pred_index = np.argmax(cnn_result)
            one_hot_pred[pred_index] = 1
            one_hot_pred = one_hot_pred.flatten()
            pred_index = np.argmax(one_hot_pred)

            if np.argmax(labels[passes]) == pred_index:
                correct_predictions += 1

            derivative = cross_entropy_loss_backward(labels[passes], cnn_result.flatten()).reshape((-1, 1))
            cnn_back = cnn.backward(learning_rate, derivative)
            pool_back = pool.backward(cnn_back)
            conv_back = conv.backward(learning_rate, pool_back)

            # # Debugging intermediate values
            # print(f"Epoch {e + 1}, Pass {passes + 1}")
            # print("Convolution result shape:", convolution_result.shape)
            # print("Pooling result shape:", pool_result.shape)
            # print("CNN result shape:", cnn_result.shape)
            # print("Loss:", loss)
            # print("Predicted labels:", np.argmax(cnn_result.flatten()))
            # print("True labels:", np.argmax(labels[passes]))
            # print("Learning rate:", learning_rate)
        
        print("Correct predictions: ", correct_predictions)
        average_loss = total_loss / len(train_array1)
        accuracy = correct_predictions / len(train_array1) * 100
        print(f"Epoch {e + 1}/{epochs} - Loss: {average_loss:.8f} - Accuracy: {accuracy:.2f}%")
        
        # Additional check after each epoch
        if not np.array_equal(labels, original_labels):
            raise ValueError("Labels array has changed during training")

        # if e > 0 and average_loss > previous_loss:
        #     learning_rate *= 0.5  # Reduce learning rate if loss increases
        
        # previous_loss = average_loss

# def softmax_predict(x):
#     exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
#     return exp_x / exp_x.sum(axis=0, keepdims=True)

def predict(input, conv, pool, cnn):
    conv_out = conv.forward(input)
    pool_out = pool.forward(conv_out)
    flattened = pool_out.flatten()
    predictions = cnn.forward(flattened)
    return predictions

# class YOLO:
#     def __init__(self, input_size, grid_size, num_boxes, num_classes):
#         self.input_size = input_size
#         self.grid_size = grid_size
#         self.num_boxes = num_boxes
#         self.num_classes = num_classes
        
#         # Example CNN structure, can be adjusted
#         self.conv1 = Conv(input_size, 3, 16)
#         self.pool1 = MaxPool(2)
#         self.conv2 = Conv((input_size[0]//2, input_size[1]//2), 3, 32)
#         self.pool2 = MaxPool(2)
#         self.conv3 = Conv((input_size[0]//4, input_size[1]//4), 3, 64)
#         self.pool3 = MaxPool(2)
#         self.fc_input_size = (input_size[0]//8) * (input_size[1]//8) * 64
#         self.fc = CNN(self.fc_input_size, grid_size * grid_size * (num_boxes * 5 + num_classes))
    
#     def forward(self, input):
#         x = self.conv1.forward(input)
#         x = self.pool1.forward(x)
#         x = self.conv2.forward(x)
#         x = self.pool2.forward(x)
#         x = self.conv3.forward(x)
#         x = self.pool3.forward(x)
#         output = self.fc.forward(x)
#         return output

#     def backward(self, learning_rate, der_output):
#         der_fc = self.fc.backward(learning_rate, der_output)
#         der_pool3 = self.pool3.backward(der_fc)
#         der_conv3 = self.conv3.backward(learning_rate, der_pool3)
#         der_pool2 = self.pool2.backward(der_conv3)
#         der_conv2 = self.conv2.backward(learning_rate, der_pool2)
#         der_pool1 = self.pool1.backward(der_conv2)
#         der_conv1 = self.conv1.backward(learning_rate, der_pool1)
#         return der_conv1
    
# def yolo_loss(predictions, labels, S, B, C):
#     # Reshape predictions and labels
#     predictions = predictions.reshape((S, S, B * (5 + C)))
#     labels = labels.reshape((S, S, B * (5 + C)))
    
#     # Extract components from predictions and labels
#     pred_conf = predictions[:, :, 0:B]
#     pred_bbox = predictions[:, :, B:5*B].reshape((S, S, B, 4))
#     pred_classes = predictions[:, :, 5*B:].reshape((S, S, C))
    
#     true_conf = labels[:, :, 0:B]
#     true_bbox = labels[:, :, B:5*B].reshape((S, S, B, 4))
#     true_classes = labels[:, :, 5*B:].reshape((S, S, C))
    
#     # Localization loss (MSE)
#     loc_loss = np.sum((true_bbox - pred_bbox) ** 2)
    
#     # Confidence loss (MSE)
#     conf_loss = np.sum((true_conf - pred_conf) ** 2)
    
#     # Classification loss (cross-entropy)
#     class_loss = -np.sum(true_classes * np.log(pred_classes + 1e-7))
    
#     total_loss = loc_loss + conf_loss + class_loss
#     return total_loss

# def yolo_loss_backward(labels, predictions, S, B, C):
#     predictions = predictions.reshape((S, S, B * (5 + C)))
#     labels = labels.reshape((S, S, B * (5 + C)))
    
#     der_predictions = 2 * (predictions - labels) / labels.size
#     return der_predictions

# def train(train_images, labels, yolo_model: YOLO, learning_rate=0.01, epochs=100):
#     original_labels = np.copy(labels)
#     S = yolo_model.grid_size
#     B = yolo_model.num_boxes
#     C = yolo_model.num_classes
    
#     for e in range(epochs):
#         total_loss = 0.0
#         correct_predictions = 0

#         for i in range(len(train_images)):
#             output = yolo_model.forward(train_images[i])
#             loss = yolo_loss(output.flatten(), labels[i].flatten(), S, B, C)
#             total_loss += loss
            
#             # Check if predictions are correct
#             pred_classes = np.argmax(output.reshape((S, S, B * (5 + C)))[:, :, 5*B:], axis=-1)
#             true_classes = np.argmax(labels[i].reshape((S, S, B * (5 + C)))[:, :, 5*B:], axis=-1)
#             correct_predictions += np.sum(pred_classes == true_classes)
            
#             # Backward pass
#             der_output = yolo_loss_backward(labels[i], output.flatten(), S, B, C).reshape(output.shape)
#             yolo_model.backward(learning_rate, der_output)
        
#         average_loss = total_loss / len(train_images)
#         accuracy = correct_predictions / (len(train_images) * S * S) * 100
#         print(f"Epoch {e + 1}/{epochs} - Loss: {average_loss:.8f} - Accuracy: {accuracy:.2f}%")
        
#         if not np.array_equal(labels, original_labels):
#             raise ValueError("Labels array has changed during training")
