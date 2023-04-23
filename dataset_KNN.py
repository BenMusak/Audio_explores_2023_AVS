import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(data_path, label_path):
    print("Loading the dataset...")
    X = np.load(data_path) # data
    y = np.load(label_path) # label

    return X, y


def reshape_data(X):
    print("Reshaping the dataset...")
    X = np.asarray(X)
    X_reshaped = X.reshape(-1, X.shape[1] * X.shape[2])
    return X_reshaped


def split_dataset(X, y, training_ratio, validation_ratio, test_ratio):
    print("Splitting the dataset...")
    
    training_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1

    if training_ratio + validation_ratio + test_ratio != 1:
        raise ValueError('Training, validation, and test ratios must sum to 1.')

    train_size = int(training_ratio * len(X))
    val_size = int(validation_ratio * len(X))
    test_size = len(X) - train_size - val_size

    train_spectrograms, val_spectrograms, train_labels, val_labels = train_test_split(X, y, test_size=val_size, random_state=42)
    train_spectrograms, test_spectrograms, train_labels, test_labels = train_test_split(train_spectrograms, train_labels, test_size=test_size, random_state=42)
    
    train_spectrograms = np.asarray(train_spectrograms)
    train_labels = np.asarray(train_labels)
    val_spectrograms = np.asarray(val_spectrograms)
    test_spectrograms = np.asarray(test_spectrograms)
    test_labels = np.asarray(test_labels)

    print('Training samples:', train_spectrograms.shape[0])
    print('Validation samples:', val_spectrograms.shape[0])
    print('Test samples:', test_spectrograms.shape[0])
    
    return train_spectrograms, train_labels, test_spectrograms, test_labels, val_spectrograms, val_labels


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def extract_delta_MFCCs(y, order):
    delta_MFCCs = librosa.feature.delta(y, order=order)
    return delta_MFCCs


def convert_to_mfcc(data, dim_reduction=True):
    mfccs = []
    print("Converting to MFCCs...")
    for y in data:
        converted = librosa.db_to_power(y)
        mfcc = librosa.feature.mfcc(S=converted)
        #visualize_MFCCs_Mel(mfcc, y, 44100)
        mfcc_delta = extract_delta_MFCCs(mfcc, 1)
        mfcc_delta2 = extract_delta_MFCCs(mfcc, 2)
        comprehensive_mfccs = np.concatenate([mfcc, mfcc_delta, mfcc_delta2])

        if dim_reduction:
            comprehensive_mfccs = np.sum(comprehensive_mfccs, axis=1)

        mfccs.append(comprehensive_mfccs)
        
    if not dim_reduction:
        mfccs = reshape_data(mfccs)
        
    return mfccs


def visualize_MFCCs_Mel(MFCCs, Mel, sr):
    print("Visualizing MFCCs...")
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img_mel = librosa.display.specshow(Mel,
                               x_axis='time', y_axis='mel', fmax=8000,
                               ax=ax[0])
    fig.colorbar(img_mel, ax=[ax[0]])
    ax[0].set(title='Mel spectrogram')
    ax[0].label_outer()
    img_MFCCs = librosa.display.specshow(MFCCs, x_axis='time', sr=sr)
    fig.colorbar(img_MFCCs, ax=[ax[1]])
    ax[1].set(title='MFCC')
    plt.title('MFCCs')
    plt.show()


def test_model_dataset(data_path, label_path, training_ratio, validation_ratio, test_ratio):
    X, y = load_data(data_path, label_path)
    mfccs = convert_to_mfcc(X)
    X_train, X_test, y_train, y_test, x_val, y_val = split_dataset(mfccs, y, training_ratio, validation_ratio, test_ratio)

    return X_train, X_test, y_train, y_test, x_val, y_val


def create_dataset(train_path, train_label_path, test_path):
    # Load the training data
    print("Creating the train dataset...")
    X_train, y_train = load_data(train_path, train_label_path)
    mfccs_train = convert_to_mfcc(X_train)
    train_data = mfccs_train
    train_labels = y_train
    #labed_parts = list(zip(X, y))
    
    # Load the test data
    print("Creating the test dataset...")
    X_test, y_test = load_data(test_path)
    mfccs_test = convert_to_mfcc(X_test)
    test_data = mfccs_test
    test_labels = y_test
    
    #X_train, X_test, y_train, y_test = split_dataset(mfccs, y, test_split)

    return train_data, train_labels, test_data, test_labels
