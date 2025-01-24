from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Carregar o modelo KMeans
kmeans_model = joblib.load('model/kmeans_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        # Coletando os dados do formulário
        A1_Score = float(request.form['A1_Score'])
        A2_Score = float(request.form['A2_Score'])
        A3_Score = float(request.form['A3_Score'])
        A4_Score = float(request.form['A4_Score'])
        A5_Score = float(request.form['A5_Score'])
        A6_Score = float(request.form['A6_Score'])
        A7_Score = float(request.form['A7_Score'])
        A8_Score = float(request.form['A8_Score'])
        A9_Score = float(request.form['A9_Score'])
        A10_Score = float(request.form['A10_Score'])
        austim = int(request.form['austim']) 
        result = A1_Score + A2_Score + A3_Score + A4_Score + A5_Score + A6_Score + A7_Score + A8_Score + A9_Score + A10_Score
        ethnicity = int(request.form['ethnicity']) 

        data = pd.DataFrame([[A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, A7_Score, A8_Score, A9_Score, A10_Score, austim, result, ethnicity]],
                            columns=['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'austim', 'result', 'ethnicity'])

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        cluster = kmeans_model.predict(data_scaled)[0]

        if cluster == 0:
            prediction_text = f"Resultado negativo para algum grau de autismo"
        else:
            prediction_text = f"Resultado positivo para algum grau de autismo"

        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
