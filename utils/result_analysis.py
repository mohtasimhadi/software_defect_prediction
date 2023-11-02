import numpy as np
from sklearn.metrics import *
from training_model.models import *


def calculate_defect_identified(model, df_x, df_y, nn_clf, cnn_clf):
    if model == nn_clf:
        result = model.predict(df_x) > 0.5

    elif model == cnn_clf:
        df_matrix = df_x.values
        df_val1 = df_matrix.reshape(df_matrix.shape[0], 1, len(df_x.columns), 1)
        result = model.predict(df_val1) > 0.5

    else:
        result = model.predict(df_x)

    actual = list(df_y["defects"])
    result = list(result)
    total_pos = df_y.value_counts()[1]
    count = 0

    for i in range(len(actual)):
        if actual[i] == True:
            if actual[i] == result[i]:
                count += 1

    return count / total_pos


# metrics method
def metrics_calculate(model_name, y_val, y_pred):
    """
    0. basic metrics values ['accuracy', 'precision', 'recall', 'fpr', 'fnr', 'auc']
    1. classification report
    2. confusion matrix
    """
    y_val = np.reshape(y_val, -1).astype(np.int32)
    y_pred = np.where(np.reshape(y_pred, -1) > 0.5, 1, 0)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    fpr = fp / (tn + fp)
    fnr = fn / (tp + fn)
    auc = roc_auc_score(y_val, y_pred)
    # print('Model:%s Acc:%.8f Prec:%.8f Recall:%.8f FNR:%.8f FPR:%.8f AUC:%.8f' % (model_name, accuracy, precision, recall, fnr, fpr, auc))
    print(model_name, "classification report:\n", classification_report(y_val, y_pred))
    print(model_name, "confusion_matrix:\n", confusion_matrix(y_val, y_pred))
    print(
        "\n%s FNR:%.8f FPR:%.8f\n%s accuracy:%.8f"
        % (model_name, fnr, fpr, model_name, accuracy_score(y_pred, y_val))
    )


def print_accuracy(model, nn_clf, cnn_clf, x_val, x_test, y_val, y_test):
    if model == nn_clf:
        y_pred_on_val = model.predict(x_val) > 0.5
        y_pred_on_test = model.predict(x_test) > 0.5

    elif model == cnn_clf:
        x_val_matrix = x_val.values
        x_val1 = x_val_matrix.reshape(x_val_matrix.shape[0], 1, len(x_val.columns), 1)
        y_pred_on_val = model.predict(x_val1) > 0.5
        x_test_matrix = x_test.values
        x_test1 = x_test_matrix.reshape(
            x_test_matrix.shape[0], 1, len(x_test.columns), 1
        )
        y_pred_on_test = model.predict(x_test1) > 0.5
    else:
        y_pred_on_val = model.predict(x_val)
        y_pred_on_test = model.predict(x_test)

    print("******", str(model), "******")
    print("<Validation Set>")
    print("Accuracy:", balanced_accuracy_score(y_val, y_pred_on_val))
    print("Avg Precision:", average_precision_score(y_val, y_pred_on_val))
    print(
        "f1_score:",
        f1_score(
            y_val, y_pred_on_val, average="weighted", labels=np.unique(y_pred_on_val)
        ),
    )
    print(
        "Precision:",
        precision_score(y_val, y_pred_on_val, labels=np.unique(y_pred_on_val)),
    )
    print(
        "Recall:", recall_score(y_val, y_pred_on_val, labels=np.unique(y_pred_on_val))
    )
    print("ROC_AUC:", roc_auc_score(y_val, y_pred_on_val))
    print("\n\n")

    y_pred_on_val_df = pd.DataFrame(y_pred_on_val, columns=["defects1"])
    y_pred_on_test_df = pd.DataFrame(y_pred_on_test, columns=["defects1"])
    val_result = pd.concat(
        [y_val["defects"].reset_index(drop=True), y_pred_on_val_df["defects1"]], axis=1
    )
    val_result = val_result.rename(
        columns={"defects": "val_actual", "defects1": "val_predict"}
    )
    test_result = pd.concat(
        [y_test["defects"].reset_index(drop=True), y_pred_on_test_df["defects1"]],
        axis=1,
    )
    test_result = test_result.rename(
        columns={"defects": "test_actual", "defects1": "test_predict"}
    )
    return val_result, test_result
