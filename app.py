import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = 'best_model.pkl'
FEATURE_ORDER = [
    'age',
    'gender',
    'study_hours_per_day',
    'social_media_hours',
    'netflix_hours',
    'part_time_job',
    'attendance_percentage',
    'sleep_hours',
    'diet_quality',
    'exercise_frequency',
    'parental_education_level',
    'internet_quality',
    'mental_health_rating',
    'extracurricular_participation',
]

GENDER_MAP = {'Male': 0, 'Female': 1, 'Other': 2}
YES_NO_MAP = {'No': 0, 'Yes': 1}
DIET_MAP = {'Fair': 0, 'Good': 1, 'Poor': 2}
PARENT_EDU_MAP = {'High School': 0, 'Bachelor': 1, 'Master': 2}
INTERNET_MAP = {'Poor': 0, 'Average': 1, 'Good': 2}


def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def build_input_dataframe() -> pd.DataFrame:
    st.sidebar.header('Student Inputs')

    age = st.sidebar.slider('Age', min_value=15, max_value=30, value=20)
    gender = st.sidebar.selectbox('Gender', list(GENDER_MAP.keys()))
    study_hours_per_day = st.sidebar.slider('Study Hours Per Day', 0.0, 12.0, 4.0, 0.5)
    social_media_hours = st.sidebar.slider('Social Media Hours', 0.0, 12.0, 2.0, 0.5)
    netflix_hours = st.sidebar.slider('Netflix Hours', 0.0, 12.0, 1.0, 0.5)
    part_time_job = st.sidebar.selectbox('Part-Time Job', list(YES_NO_MAP.keys()))
    attendance_percentage = st.sidebar.slider('Attendance Percentage', 0.0, 100.0, 85.0, 1.0)
    sleep_hours = st.sidebar.slider('Sleep Hours', 0.0, 12.0, 7.0, 0.5)
    diet_quality = st.sidebar.selectbox('Diet Quality', list(DIET_MAP.keys()))
    exercise_frequency = st.sidebar.slider('Exercise Frequency', 0, 14, 3)
    parental_education_level = st.sidebar.selectbox('Parental Education Level', list(PARENT_EDU_MAP.keys()))
    internet_quality = st.sidebar.selectbox('Internet Quality', list(INTERNET_MAP.keys()))
    mental_health_rating = st.sidebar.slider('Mental Health Rating', 1, 10, 7)
    extracurricular_participation = st.sidebar.selectbox('Extracurricular Participation', list(YES_NO_MAP.keys()))

    data = {
        'age': age,
        'gender': GENDER_MAP[gender],
        'study_hours_per_day': study_hours_per_day,
        'social_media_hours': social_media_hours,
        'netflix_hours': netflix_hours,
        'part_time_job': YES_NO_MAP[part_time_job],
        'attendance_percentage': attendance_percentage,
        'sleep_hours': sleep_hours,
        'diet_quality': DIET_MAP[diet_quality],
        'exercise_frequency': exercise_frequency,
        'parental_education_level': PARENT_EDU_MAP[parental_education_level],
        'internet_quality': INTERNET_MAP[internet_quality],
        'mental_health_rating': mental_health_rating,
        'extracurricular_participation': YES_NO_MAP[extracurricular_participation],
    }

    return pd.DataFrame([data], columns=FEATURE_ORDER)


def main():
    st.set_page_config(page_title='Student Performance Predictor', page_icon='📘', layout='wide')

    st.title('📘 Student Performance Predictor')
    st.write(
        'This Streamlit app uses your notebook\'s linear regression workflow to predict a student\'s exam score '
        'from study habits, lifestyle, attendance, and related factors.'
    )

    model = load_model()

    if model is None:
        st.error(
            'Model file not found. Put `best_model.pkl` in the same folder as this app, or run `train_model.py` first.'
        )
        st.stop()

    input_df = build_input_dataframe()

    left, right = st.columns([1, 1])
    with left:
        st.subheader('Encoded Model Input')
        st.dataframe(input_df, use_container_width=True)

    with right:
        st.subheader('Category Encoding Used')
        st.markdown(
            '- **Gender:** Male=0, Female=1, Other=2\n'
            '- **Part-Time Job:** No=0, Yes=1\n'
            '- **Diet Quality:** Fair=0, Good=1, Poor=2\n'
            '- **Parental Education:** High School=0, Bachelor=1, Master=2\n'
            '- **Internet Quality:** Poor=0, Average=1, Good=2\n'
            '- **Extracurricular Participation:** No=0, Yes=1'
        )

    if st.button('Predict Exam Score', type='primary'):
        try:
            prediction = float(model.predict(input_df)[0])
            st.success(f'Predicted Exam Score: {prediction:.2f}')

            if prediction >= 85:
                band = 'Excellent'
            elif prediction >= 70:
                band = 'Good'
            elif prediction >= 50:
                band = 'Average'
            else:
                band = 'Needs improvement'

            st.info(f'Performance Band: **{band}**')
        except Exception as exc:
            st.exception(exc)

    with st.expander('Optional: Batch prediction from a CSV'):
        uploaded_file = st.file_uploader('Upload a CSV with the same feature columns', type=['csv'])
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                missing = [col for col in FEATURE_ORDER if col not in batch_df.columns]
                if missing:
                    st.error(f'Missing required columns: {missing}')
                else:
                    preds = model.predict(batch_df[FEATURE_ORDER])
                    result_df = batch_df.copy()
                    result_df['predicted_exam_score'] = preds
                    st.dataframe(result_df, use_container_width=True)
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        'Download predictions as CSV',
                        data=csv,
                        file_name='student_score_predictions.csv',
                        mime='text/csv',
                    )
            except Exception as exc:
                st.exception(exc)


if __name__ == '__main__':
    main()
