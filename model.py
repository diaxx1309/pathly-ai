import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

def train_model():
    data = pd.read_csv("data.csv")

    le_stream = LabelEncoder()
    le_interest = LabelEncoder()
    le_exam = LabelEncoder()
    le_college = LabelEncoder()

    data["stream"] = le_stream.fit_transform(data["stream"])
    data["interest"] = le_interest.fit_transform(data["interest"])
    data["exam"] = le_exam.fit_transform(data["exam"])
    data["college"] = le_college.fit_transform(data["college"])

    X = data[["marks", "stream", "interest"]]
    y_exam = data["exam"]
    y_college = data["college"]

    exam_model = DecisionTreeClassifier()
    college_model = DecisionTreeClassifier()

    exam_model.fit(X, y_exam)
    college_model.fit(X, y_college)

    return exam_model, college_model, le_stream, le_interest, le_exam, le_college
