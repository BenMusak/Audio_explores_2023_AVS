import sys
import pickle
import time as time
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


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

    y_predict = model.predict(x_test_scaled)
    print(f'Model Score Test Data: {model.score(x_test_scaled, y_test)}')
    print(f'Accuracy Score for Test Data: {accuracy_score(y_predict, y_test)}')
    print(f'Confusion Matrix for Test Data: \n{confusion_matrix(y_predict, y_test)}')

    val_predict = model.predict(x_val)
    print(f'Model Score Validation Data: {model.score(x_val_scaled, y_val)}')
    print(f'Accuracy Score for Validation Data: {accuracy_score(val_predict, y_val)}')
    print(f'Confusion Matrix for Validation Data: \n{confusion_matrix(val_predict, y_val)}')
    
    train_predict = model.predict(x_train)
    print(f'Model Score Train Data: {model.score(x_train_scaled, y_train)}')
    print(f'Accuracy Score for Train Data: {accuracy_score(train_predict, y_train)}')
    print(f'Confusion Matrix for Train Data: \n{confusion_matrix(train_predict, y_train)}')
    
    start = time.time()
    print(f'Predicted label: {model.predict(x_test_scaled[0].reshape(1, -1))}')
    end = time.time()
    print(f'Prediction time for that one label: {end - start}')
    
    print("Getting the size of the model...")
    p = pickle.dumps(model)
    print(f'Size of the model: {sys.getsizeof(p)}')