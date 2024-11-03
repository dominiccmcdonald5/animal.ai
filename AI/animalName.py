import numpy as np
import os
import cv2  # OpenCV for image processing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def training():
    dataset_path = 'C:\\Users\\dorkd\\TestProject\\pythonProject1\\AI\\Data\\val'

    # Initialize lists for images and labels
    images = []
    labels = []

    # List the classes (assuming folder names match the class labels)
    classes = os.listdir(dataset_path)

    for label in classes:
        class_path = os.path.join(dataset_path, label)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            # Load the image
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))  # Resize to a fixed size
            images.append(img)
            labels.append(label)

    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    # Flatten the images
    X = X.reshape(X.shape[0], -1)  # Flatten the images

    # Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Create a pipeline with scaling and the classifier
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the validation set
    y_pred = model.predict(X_val)

    return label_encoder, model

def predict_image(img_path, label_encoder, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))  # Resize to the same size as the training images
    img = img.reshape(1, -1)  # Flatten the image
    prediction = model.predict(img)
    return label_encoder.inverse_transform(prediction)[0]

