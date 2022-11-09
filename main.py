import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    total_sqft = request.form.get('total_sqft')
    # print(location, bhk, bath, total_sqft)
    inp = pd.DataFrame([[location, total_sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(inp)[0]
    return str(prediction)
    # return ""


if __name__ == "__main__":
    app.run(debug=True, port=5001)
