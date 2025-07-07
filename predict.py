from model import predict

# This function makes a prediction using the learned parameters
def predict_price(rooms, theta0, theta1, mean, std):
    rooms_scaled = (rooms - mean) / std
    return predict(rooms_scaled, theta0, theta1)
