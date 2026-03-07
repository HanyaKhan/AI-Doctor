import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import webbrowser

# Load training data
training_dataset = pd.read_csv('Training.csv')
X = training_dataset.iloc[:, :-1].values
y = training_dataset.iloc[:, -1].values

# Encode disease labels
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

# Train Decision Tree model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

symptom_columns = list(training_dataset.columns[:-1])

# Load doctor info
try:
    doc_dataset = pd.read_csv('medicaldoctors_dataset.csv', names=['Name', 'Description'])
    diseases = training_dataset['prognosis'].unique()
    doctors = pd.DataFrame({
        'disease': diseases,
        'name': doc_dataset['Name'],
        'link': doc_dataset['Description']
    })
except:
    doctors = pd.DataFrame(columns=['disease', 'name', 'link'])

# Chatbot GUI class
class ChatSymptomChecker:
    def __init__(self, root):
        self.root = root
        self.root.title("Symptom Checker ChatBot")
        self.root.geometry("700x500")
        self.root.configure(bg="lightblue")

        self.chatbox = scrolledtext.ScrolledText(root, width=85, height=25, wrap=tk.WORD)
        self.chatbox.pack(pady=10)
        self.chatbox.insert(tk.END, "?? Hello! What symptom are you experiencing first?\n")
        self.chatbox.config(state='disabled')

        self.entry = tk.Entry(root, width=80)
        self.entry.pack(pady=5)
        self.entry.bind('<Return>', self.process_input)

        self.user_symptoms = []
        self.follow_up_count = 0

    def process_input(self, event):
        user_text = self.entry.get().strip().lower().replace(" ", "_")
        self.entry.delete(0, tk.END)

        self.chatbox.config(state='normal')
        self.chatbox.insert(tk.END, f"?? You: {user_text}\n")

        if user_text:
            self.user_symptoms.append(user_text)
            self.follow_up_count += 1

            if self.follow_up_count == 1:
                self.chatbox.insert(tk.END, f"?? How long have you had {user_text.replace('_', ' ')}?\n")
            elif self.follow_up_count == 2:
                self.chatbox.insert(tk.END, f"?? Are you also experiencing fatigue, nausea, or dizziness?\n")
            elif self.follow_up_count == 3:
                self.chatbox.insert(tk.END, f"?? Thank you! Let me diagnose based on your symptoms...\n")
                self.diagnose()
            else:
                self.chatbox.insert(tk.END, f"?? Anything else? You can keep listing symptoms or close the app.\n")

        self.chatbox.config(state='disabled')
        self.chatbox.yview(tk.END)

    def diagnose(self):
        input_vector = [1 if symptom.lower() in self.user_symptoms else 0 for symptom in symptom_columns]
        input_vector = np.array(input_vector).reshape(1, -1)
        predicted_label = classifier.predict(input_vector)[0]
        predicted_disease = labelencoder.inverse_transform([predicted_label])[0]

        self.chatbox.insert(tk.END, f"? Based on your symptoms, you may have: {predicted_disease}\n")

        doctor_row = doctors[doctors['disease'] == predicted_disease]
        if not doctor_row.empty:
            name = doctor_row['name'].values[0]
            link = doctor_row['link'].values[0]
            self.chatbox.insert(tk.END, f"????? Consult: {name}\n")
            self.chatbox.insert(tk.END, f"?? Visit: {link}\n")
            tk.Button(self.root, text="Open Link", command=lambda: webbrowser.open(link)).pack(pady=5)

        self.chatbox.insert(tk.END, f"?? You may enter more symptoms if you'd like to refine the diagnosis.\n")

if __name__ == '__main__':
    root = tk.Tk()
    app = ChatSymptomChecker(root)
    root.mainloop()