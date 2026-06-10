import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import streamlit as st

@st.cache_resource
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

    X_train, X_test, y_exam_train, y_exam_test = train_test_split(
        X, y_exam, test_size=0.2, random_state=42
    )
    _, _, y_col_train, y_col_test = train_test_split(
        X, y_college, test_size=0.2, random_state=42
    )

    exam_model = DecisionTreeClassifier(max_depth=6, random_state=42)
    college_model = DecisionTreeClassifier(max_depth=7, random_state=42)

    exam_model.fit(X_train, y_exam_train)
    college_model.fit(X_train, y_col_train)

    exam_acc = round(exam_model.score(X_test, y_exam_test) * 100, 2)
    college_acc = round(college_model.score(X_test, y_col_test) * 100, 2)

    return exam_model, college_model, le_stream, le_interest, le_exam, le_college, exam_acc, college_acc