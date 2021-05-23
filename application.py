import urllib
from urllib.parse import urlencode
import json
import validators
url = 'https://neutrinoapi.net/bad-word-filter'
params = {
    'user-id': 'naren81',
    'api-key': 'M4IbR5AwsH29MhiMGiJmDDyq0d01shupkPRRP0coQEG69vb9',
    'content': 'https://en.wikipedia.org/wiki/Profanity'
}


import numpy as np
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import joblib
def _get_profane_prob(prob):
  return prob[1]
application = Flask(__name__) # initializing a flask app
app=application
@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            te = []
            #  reading the inputs given by the user
            gre_score=(request.form['gre_score'])
            te.append(gre_score)
            is_research = request.form['research']
            if(is_research=='TEXT'):
                research=1
                vectorizer = joblib.load('vectorizer.joblib')
                model = joblib.load('model.joblib')
                x = model.predict(vectorizer.transform(te))
                y = np.apply_along_axis(_get_profane_prob, 1, model.predict_proba(vectorizer.transform(te)))
                if x[0]==0:
                    prediction = 'Sentence is not abusive'
                    return render_template('results.html',prediction=prediction)
                else:
                    temp = round(100*y[0])
                    prediction = 'Sentence is abusive with score '+ str(temp) +" percent"
                    print('prediction is', prediction)
            # showing the prediction results in a UI
                    return render_template('results.html',prediction=prediction)
            if(is_research=='URL'):
                if validators.url(gre_score)==True:
                    gre_score=(request.form['gre_score'])
                    params['content']=gre_score
                    encoded_params = urlencode(params).encode('utf8')
                    response = urllib.request.urlopen(url, data = encoded_params)
                    result = json.loads(response.read())
                    if result['is-bad']==True:
                        t = result['bad-words-list']
                        st = ",".join(t)
                        prediction = "Url is abusive and it contains following abusive words "+ st
                        return render_template('results.html',prediction=prediction)
                    else:
                        prediction = 'Url is not abusive'
                        return render_template('results.html',prediction=prediction)
                else:
                    prediction = "Not Valid Url"
                    return render_template('results.html',prediction=prediction)
                    
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app
