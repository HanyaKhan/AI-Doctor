import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime
from PIL import Image, ImageTk
import io
import requests
import random


class MedicalDiagnosisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DoctorBot - AI Diagnosis Chat")
        self.root.geometry("800x700")
        self.root.configure(bg='#fff5ec')

        # Load and train model
        self.load_data()
        self.train_model()

        # Initialize chat history
        self.chat_history = []

        # Create UI
        self.create_widgets()

        # Add initial bot message
        self.add_bot_message("👋 Hello! Describe your symptoms to get started.")

    def load_data(self):
        """Load and prepare the datasets"""
        df1 = pd.read_csv(r'C:\Users\PMLS\Desktop\GoogleCollabAi\realistic_patient_data.csv')
        df2 = pd.read_csv(r'C:\Users\PMLS\Desktop\GoogleCollabAi\extended_mock_patient_data.csv')

        df1.columns = df1.columns.str.strip().str.lower()
        df2.columns = df2.columns.str.strip().str.lower()

        self.df = pd.concat([df1, df2], ignore_index=True)

        # Ensure required columns exist
        required_cols = ['patient_id', 'symptoms', 'medical_history', 'family_history', 'diagnosis_code',
                         'diagnosis_name']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")

        self.df = self.df.dropna(subset=['diagnosis_code'])

        # Prepare features and target
        features = self.df[['symptoms', 'medical_history', 'family_history']].fillna('')
        self.target = self.df['diagnosis_code']

        # Text vectorization
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
        X_symptoms = self.vectorizer.fit_transform(features['symptoms'])
        X_history = self.vectorizer.transform(features['medical_history'])
        X_family = self.vectorizer.transform(features['family_history'])
        self.X = hstack([X_symptoms, X_history, X_family])

        # Create diagnosis mapping
        self.diagnosis_mapping = dict(zip(self.df['diagnosis_code'], self.df['diagnosis_name']))
        self.suggestions = {
            "G43": "Take rest, pain relievers, and reduce screen exposure.",
            "E11": "Monitor blood sugar, take insulin, and manage diet.",
            "I10": "Reduce salt, regular exercise, and BP medication.",
            "J45": "Use inhalers and avoid allergens.",
            "U07.1": "Isolate, stay hydrated, and use paracetamol.",
            "D50": "Take iron supplements and eat iron-rich food.",
            "J10": "Rest, fluids, and consult for antivirals.",
            "F32": "Seek therapy, maintain routines, and consider medication."
        }


    def train_model(self):
        """Train the machine learning model"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.target, test_size=0.2, random_state=42
        )

        clf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        self.best_clf = grid_search.best_estimator_

        train_acc = accuracy_score(y_train, self.best_clf.predict(X_train))
        test_acc = accuracy_score(y_test, self.best_clf.predict(X_test))
        print(f"Training Accuracy: {train_acc:.2f}")
        print(f"Testing Accuracy: {test_acc:.2f}")

    def create_widgets(self):
        """Create all the UI widgets"""
        # Header
        header_frame = tk.Frame(self.root, bg='#fff5ec')
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        try:
            # Try to load image from URL
            response = requests.get("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", stream=True)
            img_data = response.content
            img = Image.open(io.BytesIO(img_data))
            img = img.resize((80, 80), Image.Resampling.LANCZOS)
            self.logo_img = ImageTk.PhotoImage(img)
            logo_label = tk.Label(header_frame, image=self.logo_img, bg='#fff5ec')
            logo_label.pack(side=tk.LEFT, padx=10)
        except:
            # Fallback if image can't be loaded
            pass

        title_label = tk.Label(
            header_frame,
            text="🩺 DoctorBot - AI Diagnosis Chat",
            font=('Comic Sans MS', 16, 'bold'),
            bg='#fff5ec',
            fg='black'
        )
        title_label.pack(side=tk.LEFT, padx=10)

        # Chat display
        self.chat_frame = tk.Frame(self.root, bg='#fefcf9', bd=2, relief=tk.GROOVE)
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            width=60,
            height=20,
            font=('Comic Sans MS', 11),
            bg='#fefcf9',
            fg='black',
            padx=10,
            pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)

        # Input area
        input_frame = tk.Frame(self.root, bg='#fff5ec')
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        self.user_input = tk.Entry(
            input_frame,
            width=50,
            font=('Comic Sans MS', 11),
            bg='white',
            fg='black',
            relief=tk.GROOVE
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.user_input.bind('<Return>', lambda e: self.process_input())

        send_btn = tk.Button(
            input_frame,
            text="Send",
            command=self.process_input,
            font=('Comic Sans MS', 10, 'bold'),
            bg='#e5cfcf',
            fg='black',
            relief=tk.GROOVE
        )
        send_btn.pack(side=tk.LEFT, padx=5)

        # Bottom buttons
        btn_frame = tk.Frame(self.root, bg='#fff5ec')
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        clear_btn = tk.Button(
            btn_frame,
            text="🧹 Clear Chat",
            command=self.clear_chat,
            font=('Comic Sans MS', 10),
            bg='#e5cfcf',
            fg='black',
            relief=tk.GROOVE
        )
        clear_btn.pack(side=tk.LEFT, padx=5)

        report_btn = tk.Button(
            btn_frame,
            text="📄 Generate Report",
            command=self.generate_report,
            font=('Comic Sans MS', 10),
            bg='#e5cfcf',
            fg='black',
            relief=tk.GROOVE
        )
        report_btn.pack(side=tk.LEFT, padx=5)

    def add_user_message(self, message):
        """Add a user message to the chat"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"You: {message}\n", 'user')
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def add_bot_message(self, message):
        """Add a bot message to the chat"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"DoctorBot: {message}\n\n", 'bot')
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def process_input(self):
        """Process user input and generate response"""
        user_input = self.user_input.get().strip()
        if not user_input:
            return

        self.add_user_message(user_input)
        self.user_input.delete(0, tk.END)

        # Add to chat history
        self.chat_history.append((user_input, ""))

        if len(self.chat_history) < 5:
            # Ask follow-up questions
            bot_response = self.follow_up_questions(user_input)
            self.chat_history[-1] = (user_input, bot_response)
            self.add_bot_message(bot_response)
        else:
            # Make diagnosis
            combined = " ".join([m[0] for m in self.chat_history[-2:]] + [user_input])
            symptom_vec = self.vectorizer.transform([combined])
            empty = self.vectorizer.transform([""])
            X_input = hstack([symptom_vec, empty, empty])

            try:
                pred_code = self.best_clf.predict(X_input)[0]
                pred_name = self.diagnosis_mapping.get(pred_code, "Unknown")
                advice = self.suggestions.get(pred_code, "Please consult your doctor for detailed advice.")

                bot_response = f"🩺 Predicted Disease: {pred_name}\n💊 Advice: {advice}"
                self.chat_history[-1] = (user_input, bot_response)
                self.add_bot_message(bot_response)

                # Add closing messages
                self.add_bot_message("✅ Your report is ready. Stay healthy!")
                self.add_bot_message("❤ Take care of your health!")

                # Enable report generation
                self.last_diagnosis = pred_name
                self.last_advice = advice
            except Exception as e:
                self.add_bot_message(f"❌ Error: {str(e)}")

    def follow_up_questions(self, symptom):
        """Return relevant follow-up question based on the symptom"""
        symptom = symptom.lower()

        # Respiratory & Cold-like
        if "fever" in symptom:
            return "Do you feel feverish more at night or throughout the day?"
        elif "cough" in symptom:
            return "Is it a dry cough or with phlegm?"
        elif "cold" in symptom or "chills" in symptom:
            return "Do you also have sneezing or runny nose?"
        elif "sore throat" in symptom:
            return "Is it painful to swallow or speak?"
        elif "runny nose" in symptom:
            return "Is the discharge clear or colored?"
        elif "sneezing" in symptom:
            return "Do you experience sneezing fits in the morning or throughout the day?"
        elif "shortness of breath" in symptom:
            return "Does it occur during rest or only with activity?"
        elif "wheezing" in symptom:
            return "Is the wheezing worse at night or during exertion?"

        # Neurological
        elif "headache" in symptom:
            return "Is your headache accompanied by dizziness or nausea?"
        elif "dizziness" in symptom:
            return "Do you feel lightheaded or is the room spinning?"
        elif "numbness" in symptom:
            return "Is the numbness constant or comes and goes?"
        elif "tingling" in symptom:
            return "Where do you feel the tingling sensation?"
        elif "fainting" in symptom:
            return "Did you lose consciousness or feel like you were about to faint?"

        # Pain-related
        elif "pain" in symptom:
            return "Is the pain localized or spreads? When does it worsen?"
        elif "back pain" in symptom:
            return "Is the pain in the lower or upper back?"
        elif "chest pain" in symptom:
            return "Is it sharp, dull, or does it radiate to your arms?"
        elif "abdominal pain" in symptom or "stomach pain" in symptom:
            return "Is the pain cramping, sharp, or dull?"
        elif "joint pain" in symptom:
            return "Which joints are affected and does it worsen in the morning?"

        # Digestive
        elif "nausea" in symptom:
            return "Do you also experience vomiting or stomach cramps?"
        elif "vomiting" in symptom:
            return "How frequently are you vomiting and what triggers it?"
        elif "diarrhea" in symptom:
            return "How many times a day do you have diarrhea?"
        elif "constipation" in symptom:
            return "How many days has it been since your last bowel movement?"
        elif "bloating" in symptom:
            return "Is the bloating worse after meals or all the time?"
        elif "loss of appetite" in symptom:
            return "Have you noticed any weight loss recently?"

        # General/misc
        elif "fatigue" in symptom:
            return "Do you feel tired all day or at specific times?"
        elif "sweating" in symptom:
            return "Is it night sweats or during activity?"
        elif "weight loss" in symptom:
            return "Is the weight loss intentional or unexplained?"
        elif "weight gain" in symptom:
            return "Have your eating or exercise habits changed?"
        elif "insomnia" in symptom:
            return "Do you have trouble falling asleep or staying asleep?"
        elif "sleepiness" in symptom:
            return "Do you feel sleepy even after a full night's rest?"

        # Skin & allergies
        elif "rash" in symptom:
            return "Is the rash itchy, painful, or spreading?"
        elif "itching" in symptom:
            return "Where on your body do you feel itching?"
        elif "hives" in symptom:
            return "Did the hives appear suddenly or gradually?"
        elif "acne" in symptom:
            return "Is it severe or occasional?"
        elif "eczema" in symptom:
            return "Do certain triggers make it worse?"
        elif "dry skin" in symptom:
            return "Is the dryness localized or all over your body?"

        # Eyes, Ears, Nose, Throat
        elif "blurred vision" in symptom:
            return "Is the blurring constant or comes and goes?"
        elif "red eyes" in symptom:
            return "Do you also experience discharge or itchiness?"
        elif "earache" in symptom:
            return "Do you also have hearing loss or discharge?"
        elif "ringing in ears" in symptom or "tinnitus" in symptom:
            return "Is it in one ear or both?"
        elif "loss of smell" in symptom:
            return "Was the loss sudden or gradual?"
        elif "loss of taste" in symptom:
            return "Do you still enjoy eating food?"

        # Urinary
        elif "frequent urination" in symptom:
            return "Is it painful or just more frequent?"
        elif "painful urination" in symptom:
            return "Is there any blood in your urine?"
        elif "urinary incontinence" in symptom:
            return "Does it happen during sneezing, laughing, or randomly?"

        # Reproductive
        elif "irregular periods" in symptom:
            return "How long have your periods been irregular?"
        elif "menstrual cramps" in symptom:
            return "Are the cramps more severe than usual?"
        elif "vaginal discharge" in symptom:
            return "Is it accompanied by any odor or irritation?"
        elif "erectile dysfunction" in symptom:
            return "Is it occasional or persistent?"

        # Mental health
        elif "anxiety" in symptom:
            return "Do you feel anxious in social settings or generally all the time?"
        elif "depression" in symptom:
            return "Have you lost interest in things you once enjoyed?"
        elif "mood swings" in symptom:
            return "Do your mood changes affect your daily activities?"

        # Other common
        elif "palpitations" in symptom:
            return "Do your heart palpitations occur with exertion or at rest?"
        elif "bruising" in symptom:
            return "Are the bruises painful or occur without injury?"
        elif "swelling" in symptom:
            return "Where do you experience swelling and is it persistent?"

        else:
            return random.choice (["Hmmm!. I am near the diagnosis Can you share more about your symptoms or medical history?",
                                   "I think I'm getting closer to identifying the issue. Could you provide more details about your symptoms or medical history?",
                                   "It seems like we're almost there. Can you elaborate on your symptoms or any relevant medical history?",
                                   "I'm narrowing it down. Could you share more information about your symptoms or any previous health conditions",
                                   "We're making progress! Please tell me more about your symptoms or your medical background",
                                   "I’m almost there! Can you give me more insight into your symptoms or medical history?",
                                   "I’m getting a clearer picture. Could you provide more details on your symptoms or any past medical conditions?",
                                   "We’re on the right track! Please share more about your symptoms or any significant health history.",
                                   "It feels like we're almost done! Could you give more details on your symptoms or medical history?"])

    def generate_report(self):
        """Generate PDF report of the diagnosis"""
        if not hasattr(self, 'last_diagnosis'):
            messagebox.showwarning("Warning", "No diagnosis available to generate report")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Save Diagnosis Report"
        )

        if not filename:
            return

        try:
            c = canvas.Canvas(filename, pagesize=letter)
            width, height = letter
            y = height - 50

            # Title
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, y, "AI Doctor Diagnosis Report")
            y -= 30

            # Date
            c.setFont("Helvetica", 12)
            c.drawString(50, y, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
            y -= 40

            # Conversation History
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y, "Conversation History:")
            y -= 20

            c.setFont("Helvetica", 11)
            for user, bot in self.chat_history:
                for line in [f"👤 You: {user}", f"🤖 Doctor: {bot}"]:
                    for part in line.split('\n'):
                        c.drawString(60, y, part)
                        y -= 15
                        if y < 50:
                            c.showPage()
                            y = height - 50

            # Diagnosis and Recommendation
            y -= 10
            c.setFont("Helvetica-Bold", 13)
            c.drawString(50, y, f"Diagnosis: {self.last_diagnosis}")
            y -= 20
            c.drawString(50, y, f"Recommendation: {self.last_advice}")

            # Disclaimer
            y -= 40
            c.setFont("Helvetica-Oblique", 10)
            c.drawString(50, y,
                         "Disclaimer: This AI-based report is for informational purposes only. Consult a medical professional for treatment.")

            c.save()
            messagebox.showinfo("Success", f"Report saved as {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")

    def clear_chat(self):
        """Clear the chat history"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_history = []
        self.add_bot_message("👋 Hello! Describe your symptoms to get started.")

# text_widget = tk.Text(root, font=("Arial", 20), wrap="word", width=70, height=15)
# text_widget.insert("1.0", "\n".join(f"{k}: {v}" for k, v in suggestions.items()))
# text_widget.pack(padx=10, pady=10)
if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalDiagnosisApp(root)
    root.mainloop()

    # train_acc = accuracy_score(y_train, self.best_clf.predict(X_train))
    # test_acc = accuracy_score(y_test, self.best_clf.predict(X_test))
    # print(f"Training Accuracy: {train_acc:.2f}")
    # print(f"Testing Accuracy: {test_acc:.2f}")

    #
    # start_time = time.time()
    #
    # print(f"Execution time: {time.time() - start_time:.2f} seconds")
