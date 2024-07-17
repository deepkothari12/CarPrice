from flask import Flask , render_template , request
import pandas as pd
import pickle as pkl
import pandas
app = Flask(__name__)

model = pkl.load(open("D:\Vs.code\.vscode\CarPrice\LinearRegressionmodel.pkl" , 'rb'))
car = pd.read_csv('D:\Vs.code\.vscode\CarPrice\Clear_car_data.csv')


@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_model = sorted(car['name'].unique())
    year = sorted(car['year'].unique() , reverse=True)
    fuel_type = sorted(car['fuel_type'].unique())

    return render_template('index.html' , companies = companies , car_model = car_model ,
                           years = year , fuel_type = fuel_type)


@app.route('/predict' , methods = ['post'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms = int(request.form.get('kms'))

    prediction = model.predict(pd.DataFrame([[car_model , company , year , kms , fuel_type ]] , columns=['name' , 'company',
                                                                                                         'year','kms_driven' , 'fuel_type']))
    
    #print(prediction[0])
    return str(prediction[0])


if(__name__ == "__main__"):
    app.run(debug=True , port=4040)