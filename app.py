from flask import Flask,request
from model import predict
# create the flask object
app = Flask(__name__)

@app.route('/')
def index():
    return "OK"

@app.route('/predict',methods=['GET','POST'])
def predict():
    data = load_X()
    if data == None:
        return 'Got None'
    else:
        # model.predict.predict returns a dictionary
        prediction = model.predict(data)
    return json.dumps(str(prediction))
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
