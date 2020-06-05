from flask import Flask, render_template, request

from src.predict_one import predict_one

app = Flask(__name__)

# home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/demo', methods=['GET', 'POST'])
def demo():
    return render_template('demo.html')


@app.route('/demo_predict', methods=['POST'])
def my_form_post():
    text = request.form['tweet']
    df = predict_one(text)
    html_df = df.to_html(classes='table table-hover')
    return render_template('demo_predict.html', pred=df['Prediction']['Ensemble'],preds=html_df, tweet=text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)

