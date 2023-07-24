import numpy as np
from train_model import load_mnist_data, build_model, evaluate_model

def predict_images(model, test_images, num_images=10):
    # Make predictions on the test images
    predictions = model.predict(test_images[:num_images])
    predicted_labels = [np.argmax(prediction) for prediction in predictions]
    print("Predicted labels:", predicted_labels)

if __name__ == "__main__":
    _, _, test_images, _ = load_mnist_data()
    model = build_model()
    model.load_weights('model_weights.h5')  # Load the pre-trained model weights (if available)
    test_accuracy = evaluate_model(model, test_images)
    print("Test accuracy:", test_accuracy)

    # Example: Print predictions for the first 10 test images
    predict_images(model, test_images)
