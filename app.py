from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)


model = joblib.load('linear_regression_model.pkl')

@app.route('/')
def home():
    return render_template('webpage.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    sex = request.form['sex']
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    region = request.form['region']
    
    
    sex_male = 1 if sex == 'male' else 0
    smoker_yes = 1 if smoker == 'yes' else 0
    region_dict = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    region_encoded = region_dict.get(region, -1)
    
    
    input_data = np.array([[age, bmi, children, sex_male, smoker_yes, region_encoded]])
    
   
    predicted_charge = model.predict(input_data)[0]
    
    return render_template('prediction.html', predicted_charge=predicted_charge)

if __name__ == '__main__':
    app.run(debug=True)
