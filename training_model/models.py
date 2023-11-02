import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Conv1D, Flatten, MaxPool2D


def NN(
    original_data,
    original_X,
    original_Y,
    combined_training_data,
    x_train1,
    x_train2,
    x_train,
    x_test,
    x_val,
    y_train1,
    y_train2,
    y_train,
    y_test,
    y_val,
):
    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(
        Dense(units=15, activation="relu", input_dim=len(original_X.columns))
    )
    # Adding the second hidden layer
    classifier.add(Dense(units=8, activation="relu"))
    classifier.add(Dense(units=5, activation="relu"))
    # Adding the output layer
    classifier.add(Dense(units=1, activation="sigmoid"))
    # Compiling the ANN

    classifier.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    # Fitting the ANN to the Training set
    classifier.fit(x_train, y_train, batch_size=10, epochs=50)

    # Making the predictions and evaluating the model
    # Predicting the Test set results
    y_pred = classifier.predict(x_val)
    y_pred = y_pred > 0.5
    y_pred = pd.DataFrame(y_pred, columns=["defects"])
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_val, y_pred)
    from sklearn.metrics import accuracy_score

    accuracy_score(y_val, y_pred)

    return classifier


def random_forest(
    original_data,
    original_X,
    original_Y,
    combined_training_data,
    x_train1,
    x_train2,
    x_train,
    x_test,
    x_val,
    y_train1,
    y_train2,
    y_train,
    y_test,
    y_val,
):
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    clf.fit(x_train, y_train.values.ravel())
    return clf


def svm(
    original_data,
    original_X,
    original_Y,
    combined_training_data,
    x_train1,
    x_train2,
    x_train,
    x_test,
    x_val,
    y_train1,
    y_train2,
    y_train,
    y_test,
    y_val,
):
    clf = SVC(gamma="auto")
    clf.fit(x_train, y_train.values.ravel())
    return clf


def XGBoost(
    original_data,
    original_X,
    original_Y,
    combined_training_data,
    x_train1,
    x_train2,
    x_train,
    x_test,
    x_val,
    y_train1,
    y_train2,
    y_train,
    y_test,
    y_val,
):
    clf = xgb = XGBClassifier(
        max_depth=9,
        learning_rate=0.01,
        n_estimators=500,
        reg_alpha=1.1,
        colsample_bytree=0.9,
        subsample=0.9,
        n_jobs=5,
    )
    clf.fit(
        x_train,
        y_train.values.ravel(),
        eval_set=[(x_val, y_val.values.ravel())],
        early_stopping_rounds=50,
    )
    return clf


def cnn(
    original_data,
    original_X,
    original_Y,
    combined_training_data,
    x_train1,
    x_train2,
    x_train,
    x_test,
    x_val,
    y_train1,
    y_train2,
    y_train,
    y_test,
    y_val,
):
    x_train_matrix = x_train.values
    x_val_matrix = x_val.values
    y_train_matrix = y_train.values
    y_val_matrix = y_val.values

    ytrainseries = y_train["defects"]
    yvalseries = y_val["defects"]

    img_rows, img_cols = 1, len(original_X.columns)

    x_train1 = x_train_matrix.reshape(x_train_matrix.shape[0], img_rows, img_cols, 1)
    x_val1 = x_val_matrix.reshape(x_val_matrix.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    # add model layers
    # conv layers
    model.add(Conv2D(64, kernel_size=1, activation="relu", input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=1, activation="relu"))
    model.add(Conv2D(16, kernel_size=1, activation="relu"))
    # desne layer
    model.add(Flatten())
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    # compile model using accuracy to measure model performance
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # train the model
    model.fit(x_train1, y_train_matrix, epochs=50)
    y_pred = model.predict(x_val1) > 0.5
    y_pred_df = pd.DataFrame(y_pred)

    return model
