from flask import Flask, render_template, request, redirect, url_for
from wine_classifier import get_trained_classifier, return_recommendations
from wtforms import StringField, SubmitField
from flask_wtf import FlaskForm
import resource
import joblib

def get_train_classifier():
    return joblib.load('nb-classifier.joblib')

def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_string"
classifier = get_trained_classifier()
print('done!')

class Form(FlaskForm):
    style = {'class': "place-items-center; bg-stone-800; border-2; border-gray-200; rounded; shadow;"}
    description = StringField('Description', render_kw=style)
    submit = SubmitField('Enquire')

@app.route('/', methods=['GET', 'POST'])
def idnex():
    form = Form()
    if request.method == 'POST':
        description = form.description.data
        output = return_recommendations(description, classifier)
        return render_template('classify.html', form=form, first=f'{output[0][0]}   ({output[0][1]}%)', 
                                                           second=f'{output[1][0]}   ({output[1][1]}%)', 
                                                           third=f'{output[2][0]}   ({output[2][1]}%)')
    return render_template('classify.html', form=form, first='', second='', third='')

if __name__ == '__main__':
    # limit_memory()
    app.run(debug=True)