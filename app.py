############################################
# app.py (Flask API)
############################################
from flask import Flask, jsonify
from pipeline import run_pipeline

app = Flask(__name__)

@app.route("/run", methods=['GET'])
def run():
    df = run_pipeline()
    outliers = df[df['anomaly']==-1]

    return jsonify({
        "total_records": len(df),
        "anomalies": len(outliers),
        "sample": outliers.head(5).to_dict()
    })


@app.route("/health")
def health():
    return {"status":"ok"}


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
