from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('dia.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("dia.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[x for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        data=1
    else :
        data=0
    
    print(output,data)
    return render_template("afterdia.html",data=data)
        

if __name__ == '__main__':
    app.run(debug=True)
