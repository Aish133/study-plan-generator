import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate mock data
def generate_mock_data(num_samples=100):
    data = {
        'How many hours do you study each day (outside of classes)?': pd.Series(pd.np.random.randint(1, 10, size=num_samples)),
        'What is your preferred learning style?': pd.Series(pd.np.random.choice(['Visual', 'Auditory', 'Kinesthetic'], size=num_samples)),
        'How do you organize your study schedule?': pd.Series(pd.np.random.choice(['Daily', 'Weekly', 'Flexible'], size=num_samples)),
        'Current CGPA or percentage': pd.Series(pd.np.random.uniform(50, 100, size=num_samples))
    }
    return pd.DataFrame(data)

# Load the dataset
df = generate_mock_data()

# Clean column names by stripping leading/trailing spaces
df.columns = df.columns.str.strip()

# Selecting relevant columns for clustering
X = df[['How many hours do you study each day (outside of classes)?',
        'What is your preferred learning style?',
        'How do you organize your study schedule?',
        'Current CGPA or percentage']]

# Handle missing values
X['How many hours do you study each day (outside of classes)?'] = X['How many hours do you study each day (outside of classes)?'].fillna(0)
X = X.fillna('Unknown')

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X, drop_first=True)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Define the number of clusters
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
df['Study Plan Cluster'] = kmeans.fit_predict(X_scaled)

# Define questions for each subject to determine knowledge level
knowledge_questions = {
    'DBMS': {
        'What are indexes in a database?': "Indexes are used to optimize database query performance by reducing the time complexity of search operations.",
        'What are the different types of joins?': "The main types of joins are INNER JOIN, LEFT JOIN, RIGHT JOIN, and FULL JOIN."
    },
    'AI': {
        'What is overfitting?': "Overfitting happens when a model performs well on training data but poorly on new unseen data.",
        'What is supervised learning?': "Supervised learning is when a model is trained using labeled data to predict the correct output."
    },
    'OS': {
        'What are race conditions?': "Race conditions occur when two or more processes access shared data concurrently, leading to unpredictable outcomes.",
        'Explain system calls in detail.': "System calls provide the interface between a process and the operating system, allowing processes to request services from the kernel."
    }
}

# Define study plans with more detailed subjects and follow-up questions based on level
detailed_plans = {
    'Intensive Study Plan': {
        'Beginner': {
            'DBMS': 'Start with foundational topics like database design, ER models, and relational algebra. Practice creating basic SQL queries for table creation and data retrieval.',
            'OS': 'Focus on understanding the basics of operating systems, including process management, memory allocation, and I/O systems. Begin with simple OS programming exercises.',
            'AI': 'Learn the basic concepts of AI, such as search algorithms and logic, while practicing simple coding examples in Python.',
            'Exercises': 'Solve 10-15 basic SQL queries for creating tables, inserting data, and simple SELECT queries.'
        },
        'Intermediate': {
            'DBMS': 'Focus on normalization, joins, and transaction management. Understand indexing and query optimization. Work on complex SQL queries.',
            'OS': 'Learn about process scheduling, synchronization, and virtual memory. Implement simple algorithms for CPU scheduling.',
            'AI': 'Study machine learning algorithms, especially supervised learning (e.g., decision trees, SVMs). Work on hands-on exercises in Python.',
            'Exercises': 'Solve SQL queries with multiple joins, subqueries, and implement small projects like designing a library database.'
        },
        'Advanced': {
            'DBMS': 'Dive into advanced topics like distributed databases, concurrency control, and database security. Work on indexing strategies and query optimization.',
            'OS': 'Study distributed operating systems, security protocols, and advanced memory management techniques. Work on projects like implementing a small OS kernel.',
            'AI': 'Master deep learning concepts, including neural networks, and implement AI models using frameworks like TensorFlow or PyTorch.',
            'Exercises': 'Work on DBMS case studies, optimize complex queries, and implement a mini-project such as a dynamic web-based application that interacts with databases.'
        }
    }
}

# Function to evaluate user knowledge based on answers
def evaluate_knowledge(subject, answers):
    questions = knowledge_questions[subject]
    correct_answers = 0
    total_questions = len(questions)

    for question, correct_answer in questions.items():
        user_answer = answers.get(question, "").strip().lower()
        if user_answer == correct_answer.lower():
            correct_answers += 1

    if correct_answers == total_questions:
        return 'Advanced'
    elif correct_answers >= total_questions // 2:
        return 'Intermediate'
    else:
        return 'Beginner'

# Function to predict study plan and get personalized study plan for the selected subject
def predict_study_plan(study_hours, learning_style, study_schedule, cgpa, subject, answers):
    user_input = pd.DataFrame([[study_hours, learning_style, study_schedule, cgpa]],
                              columns=['How many hours do you study each day (outside of classes)?',
                                       'What is your preferred learning style?',
                                       'How do you organize your study schedule?',
                                       'Current CGPA or percentage'])

    user_input_encoded = pd.get_dummies(user_input).reindex(columns=X_encoded.columns, fill_value=0)
    user_input_scaled = scaler.transform(user_input_encoded)
    cluster = kmeans.predict(user_input_scaled)[0]

    level = evaluate_knowledge(subject, answers)
    detailed_plan = detailed_plans['Intensive Study Plan'][level].get(subject, 'No specific plan available')

    return df['Study Plan Cluster'].unique()[cluster], detailed_plan

# Streamlit App Interface
st.title("Personalized Study Plan Generator")

study_hours = st.number_input("How many hours do you study each day (outside of classes)?", min_value=0.0, max_value=24.0)
learning_style = st.text_input("What is your preferred learning style?")
study_schedule = st.text_input("How do you organize your study schedule?")
cgpa = st.number_input("What is your current CGPA or percentage?", min_value=0.0, max_value=100.0)
subject = st.selectbox("Choose a subject", ["DBMS", "OS", "AI"])

st.write("Answer the following questions to evaluate your knowledge:")
answers = {}
for question in knowledge_questions[subject]:
    answers[question] = st.text_input(question)

if st.button("Generate Study Plan"):
    if study_hours and learning_style and study_schedule and cgpa and subject:
        cluster, detailed_plan = predict_study_plan(study_hours, learning_style, study_schedule, cgpa, subject, answers)
        st.write(f"Predicted Study Plan Cluster: {cluster}")
        st.write(f"Detailed Plan for {subject}: {detailed_plan}")
    else:
        st.write("Please fill in all fields.")

