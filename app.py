from flask import Flask, render_template, request, jsonify 
from joblib import load

app = Flask(__name__)
model = load('decision_tree_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Form Data:", request.form)
        age_mapping = {
            '55-59': 0,
            '80 or older': 1,

            '65-69': 2,
            '75-79': 3,
            '40-44': 4,
            '70-74': 5,
            '60-64': 6,
            '50-54': 7,
            '45-49': 8,
            '18-24': 9,
            '35-39': 10,
            '30-34': 11,
            '25-29': 12
        }
        # Retrieve form data and process it
        smoking = 'smoking' in request.form  # 1 if checkbox is checked, 0 otherwise
        stroke = 1 if request.form.get('stroke', 'No') == 'Yes' else 0  # 1 if 'Yes' selected, 0 otherwise
        difficulty_walking = 1 if request.form.get('difficultyWalking', 'No') == 'Yes' else 0  # 1 if 'Yes' selected, 0 otherwise
        diabetic = 1 if request.form.get('diabetic', 'No') == 'Yes' else 0  # 1 if 'Yes' selected, 0 otherwise
        sex = 1 if request.form.get('sex', 'Male') == 'Female' else 0  # 1 for Female, 0 for Male
        kidney_disease = 1 if request.form.get('kidneyDisease', 'No') == 'Yes' else 0  # 1 if 'Yes' selected, 0 otherwise
        skin_cancer = 1 if request.form.get('skinCancer', 'No') == 'Yes' else 0  # 1 if 'Yes' selected, 0 otherwise
        physical_activity = 1 if request.form.get('physicalActivity', 'No') == 'Yes' else 0  # 1 if 'Yes' selected, 0 otherwise
        age_category = request.form['ageCategory']
        physical_health = float(request.form['physicalHealth'])
        age = age_mapping.get(age_category)
        print("Received Form Data:")
        print("Smoking:", smoking)
        print("Stroke:", stroke)
        print("Difficulty Walking:", difficulty_walking)
        print("Diabetic:", diabetic)
        print("Sex (Female=1, Male=0):", sex)
        print("Kidney Disease:", kidney_disease)
        print("Skin Cancer:", skin_cancer)
        print("Physical Activity:", physical_activity)
        print("Age Category:", age_category)
        print("Physical Health:", physical_health)
        print("Age (Mapped):", age)
        prediction = model.predict([[smoking, stroke, physical_health, difficulty_walking, sex, age, diabetic, physical_activity, kidney_disease, skin_cancer]])
        print(prediction)
        prediction = int(prediction[0])
        return jsonify({'prediction': prediction})
    except KeyError as e:
        # Handle missing form field
        return jsonify({'error': f'Missing form field: {e}'}), 400
if __name__ == '__main__':
    app.run(debug=False)
