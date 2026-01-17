import streamlit as st
from model import train_model

st.set_page_config(page_title="PATHLY AI", page_icon="🎓")

st.title("🎓 PATHLY AI")
st.subheader("Personalized Learning & Career Recommendation System")

exam_model, college_model, le_stream, le_interest, le_exam, le_college = train_model()

marks = st.slider("Enter your marks (%)", 40, 100, 75)
stream = st.selectbox("Select Stream", ["Science", "Commerce", "Arts"])
interest = st.selectbox(
    "Select Interest",
    ["Engineering", "Medical", "Business", "Law", "Design", "Finance", "Computer Science", "Management"]
)

if st.button("Get My Career Path 🚀"):
    stream_encoded = le_stream.transform([stream])[0]
    interest_encoded = le_interest.transform([interest])[0]

    prediction_input = [[marks, stream_encoded, interest_encoded]]

    exam_pred = exam_model.predict(prediction_input)
    college_pred = college_model.predict(prediction_input)

    exam_result = le_exam.inverse_transform(exam_pred)[0]
    college_result = le_college.inverse_transform(college_pred)[0]

    st.success("🎯 Recommended Career Path")
    st.write("📘 Recommended Exam:", exam_result)
    st.write("🏫 Suggested College Type:", college_result)

    st.info("📚 Learning Path")
    st.write("- Strengthen basics of your stream")
    st.write("- Practice exam-level questions")
    st.write("- Learn skill-based courses")
