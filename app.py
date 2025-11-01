from flask import Flask,request,render_template
import numpy as np
import pickle
import pandas
import sklearn


#importing model
model=pickle.load(open('model.pkl','rb'))

#creating flask app
app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])

#prediction method
def predict():
    N=int(request.form['Nitrogen'])
    P=int(request.form['Phosporus'])
    K=int(request.form['Potassium'])
    temp=float(request.form['Temperature'])
    humidity=float(request.form['Humidity'])
    ph=float(request.form['Ph'])
    rainfall=float(request.form['Rainfall']) 

    feature_list=[N, P, K, temp, humidity, ph, rainfall]
    single_pred=np.array(feature_list).reshape(1,-1)#convert 1-d array through numpy

    predict=model.predict(single_pred)
#final predict ans store in predict model
#now we find ans in crop_dict if find

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if predict[0] in crop_dict:
        crop = crop_dict[predict[0]]
        result="{} is a best crop to be cultivated ".format(crop)
    else:
        result="Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html',result=result)
#now final result sent ti render_template



#python main
if __name__ == "__main__":
    app.run(debug=True)
