import joblib
import numpy as np

# Load trained model
model = joblib.load('models/model.pkl')

def make_prediction(features):
    """
    Make prediction using trained model
    """

    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)

    return int(prediction[0])


# Example local test
if __name__ == "__main__":

    sample_data = [1, 0, 34, 45000]

    result = make_prediction(sample_data)

    print(f"Prediction: {result}")