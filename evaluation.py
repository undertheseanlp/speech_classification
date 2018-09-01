import pandas as pd
from sklearn.metrics import classification_report, accuracy_score


def extract_id(row):
    return int(row["id"].split(".")[0])
    print(0)


def extract_label(row):
    gender, accent = row['gender'], row['accent']
    return f"{gender},{accent}"


def evaluation(output_file, true_file):
    y_pred = pd.read_csv(output_file)
    y_pred["id"] = y_pred.apply(extract_id, axis=1)
    y_pred = y_pred.sort_values("id").reset_index()
    y_pred = y_pred.apply(extract_label, axis=1)
    y_true = pd.read_csv(true_file)
    y_true = y_true.apply(extract_label, axis=1)
    print(classification_report(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))


if __name__ == '__main__':
    evaluation("submission.csv", "data/public_test_gt.csv")
