# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Step 1: Load the dataset
training_dataset = pd.read_csv('Training.csv')
test_dataset = pd.read_csv('Testing.csv')

# Step 2: Prepare data
X = training_dataset.iloc[:, 0:132].values
y = training_dataset.iloc[:, -1].values

# Encode disease names
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

# Step 3: Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Step 4: Save symptom names
symptom_columns = training_dataset.columns[:-1]

# Step 5: Define doctor profiles
doctor_profiles = {
    'Dr. Amarpreet Singh Riar': ['fever', 'stomach_pain', 'body_ache', 'throat_infection', 'typhoid', 'hives', 'swelling', 'cold', 'infectious_disease', 'diabetes', 'type_2_diabetes', 'stress'],
    'Dr. (Maj.) Sharad Shrivastava': ['diabetes', 'chest_pain', 'asthma', 'thyroid', 'cough', 'pneumothorax', 'heatstroke'],
    'Dr. Anirban Biswas': ['chest_pain', 'typhoid', 'stomach_pain', 'type_1_diabetes'],
    'Dr. Aman Vij': ['fever', 'viral_fever', 'blood_pressure', 'chickenpox', 'thyroid', 'anemia', 'dizziness', 'stomach_pain'],
    'Dr. Mansi Arya': ['stress', 'skin_disease', 'mental_health']
}

# Step 6: Intelligent follow-up questions for symptoms
def ask_symptom_specific_questions(symptom_list):
    follow_up_qna = {
        "fever": "Do you feel feverish more at night or all day?",
        "cough": "Is your cough dry or productive (with mucus)?",
        "chest_pain": "Is the chest pain sharp, dull, or tight? Does it worsen with breathing?",
        "dizziness": "Does the dizziness occur when standing up quickly?",
        "fatigue": "Is your fatigue persistent even after resting?",
        "nausea": "Do you feel nauseous mostly in the morning or after eating?",
        "stomach_pain": "Is your stomach pain localized or spread out?",
        "headache": "Is it a throbbing headache or a pressure-like pain?",
        "sore_throat": "Do you have difficulty swallowing as well?",
        "body_pain": "Is the pain in joints or muscles?"
    }

    print("\n💬 Follow-up questions based on your symptoms:\n")
    for symptom in symptom_list:
        question = follow_up_qna.get(symptom.lower().replace(" ", "_"))
        if question:
            answer = input(f"❓ [{symptom.replace('_', ' ').capitalize()}] {question}: ").strip()
            if not answer:
                print("❌ Invalid response, skipping.")
            else:
                print(f"✅ Got it. Response: {answer}")
        else:
            print(f"ℹ No follow-up available for: {symptom}")

# Step 7: Get user input
def get_user_input():
    print("🤖 Let's identify your symptoms.")
    symptoms = []
    found_in_profiles = False  # Initialize variable

    # Normalize input symptom to match dataset format (lowercase, underscores)
    main_symptom = input("❓ What is your main symptom? ").strip().lower().replace(" ", "_")
    symptoms.append(main_symptom)

    # Check if the main symptom exists in the dataset columns or doctor profiles
    if main_symptom not in symptom_columns:
        print("⚠ Symptom not found in dataset, but checking doctor profiles.")
        # Check if the symptom exists in any doctor's profile
        for doctor, symptoms_list in doctor_profiles.items():
            if main_symptom in [symptom.lower().replace(" ", "_") for symptom in symptoms_list]:
                found_in_profiles = True
                break
        if not found_in_profiles:
            print("⚠ Symptom not found in doctor profiles either.")
            return symptoms  # Only the main symptom provided
    else:
        found_in_profiles = True
        # Proceed with normal symptom processing (when found in dataset)
        symptom_series = training_dataset[main_symptom]
        correlations = {}

        for col in symptom_columns:
            if col != main_symptom:
                correlation = training_dataset[col].corr(symptom_series)
                if not np.isnan(correlation):
                    correlations[col] = correlation

        # Pick top 5 related symptoms
        follow_up_symptoms = sorted(correlations, key=correlations.get, reverse=True)[:5]

        print("\n💬 Please answer a few follow-up questions related to your symptom:\n")
        for symptom in follow_up_symptoms:
            answer = input(f"❓ Do you also have '{symptom.replace('_', ' ')}'? (yes/no): ").strip().lower()
            if answer in ['yes', 'y']:
                symptoms.append(symptom)

    # Only ask specific questions for symptoms that exist in the dataset or doctor profiles
    if found_in_profiles:
        ask_symptom_specific_questions(symptoms)

    return symptoms

# Step 8: Build input vector
def build_input_vector(user_symptoms):
    input_vector = [0] * len(symptom_columns)
    for idx, symptom in enumerate(symptom_columns):
        if symptom.strip().lower().replace(" ", "_") in user_symptoms:
            input_vector[idx] = 1
    return input_vector

# Step 9: Predict disease (FIXED - was commented out)
def predict_disease(input_vector):
    input_vector = np.array(input_vector).reshape(1, -1)
    predicted_label = classifier.predict(input_vector)[0]
    predicted_disease = labelencoder.inverse_transform([predicted_label])[0]
    return predicted_disease

# Step 10: Suggest doctor
def suggest_doctor(predicted_disease, user_symptoms):
    predicted_disease_lower = predicted_disease.lower().replace(" ", "_")
    best_match = None
    max_overlap = 0

    for doctor, symptoms in doctor_profiles.items():
        overlap = len(set(symptoms).intersection(user_symptoms + [predicted_disease_lower]))
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = doctor

    return best_match

# Step 11: Run chatbot (FIXED - corrected main guard)
if __name__ == "__main__":
    user_symptoms = get_user_input()

    if len(user_symptoms) > 0:  # Changed to > 0 to handle single symptom cases
        input_vector = build_input_vector(user_symptoms)
        predicted_disease = predict_disease(input_vector)

        print("\n✅ Based on your symptoms, you may have:", predicted_disease)

        # Suggest doctor
        suggested_doctor = suggest_doctor(predicted_disease, user_symptoms)
        if suggested_doctor:
            print("👨‍⚕ Recommended Doctor:", suggested_doctor)
        else:
            print("⚠ No matching doctor found.")
    else:
        print("⚠ No symptoms provided. Please provide at least one symptom.")