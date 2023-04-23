import sys
import pickle
import time as time
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


def knn_model(x_train, x_test, x_val, y_train, y_test, y_val):

    print(f'Shape: {x_train.shape}')
    print(f'Observation: \n{x_train[0]}')
    print(f'Labels: {y_train}')

    print("Scaling the data...")
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    x_val_scaled = scaler.transform(x_val)

    grid_params = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    print("Training the model...")
    model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1)
    model.fit(x_train_scaled, y_train)

    print(f'Model Score Training Data: {model.score(x_test_scaled, y_test)}')

    y_predict = model.predict(x_test_scaled)
    print(f'Confusion Matrix for training data: \n{confusion_matrix(y_predict, y_test)}')

    val_predict = model.predict(x_val)
    print(f'Model Score Validation Data: {model.score(x_val_scaled, y_val)}')
    print(f'Confusion Matrix for Validation data: \n{confusion_matrix(val_predict, y_val)}')
    
    start = time.time()
    print(f'Predicted label: {model.predict(x_test_scaled[0].reshape(1, -1))}')
    end = time.time()
    print(f'Prediction time for that one label: {end - start}')
    
    print("Getting the size of the model...")
    p = pickle.dumps(model)
    print(f'Size of the model: {sys.getsizeof(p)}')