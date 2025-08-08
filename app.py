from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load('model/ridge_model.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            grlivarea = float(request.form['GrLivArea'])
            garagecars = float(request.form['GarageCars'])
            totalbsmt = float(request.form['TotalBsmtSF'])
            overallqual = float(request.form['OverallQual'])

            input_df = pd.DataFrame([[grlivarea, garagecars, totalbsmt, overallqual]],
                                    columns=['GrLivArea', 'GarageCars', 'TotalBsmtSF', 'OverallQual'])

            prediction = model.predict(input_df)[0]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
