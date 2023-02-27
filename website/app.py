from flask import Flask, render_template, request, url_for, send_from_directory
from wine_classifier import get_trained_classifier, return_recommendations
from wtforms import StringField, SubmitField
from flask_wtf import FlaskForm
import resource
import os

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_string"

# Attempting to set resouce limits.

# soft, hard = resource.getrlimit(resource.RLIMIT_AS)
# soft = int(resource.RLIM_INFINITY / 200000000)
# hard = int(resource.RLIM_INFINITY / 200000000)
# print(f'soft: {soft}')
# print(f'hard: {hard}')
# resource.setrlimit(resource.RLIMIT_AS, (-1, -1))

classifier = get_trained_classifier()
print('done!')

class Form(FlaskForm):
    style = {'class': "place-items-center border-2 border-gray-300 rounded-lg p-2 font-sans w-3/4 md:w-1/2"}
    description = StringField('Description', render_kw=style)
    submit = SubmitField('Enquire')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = Form()
    if request.method == 'POST':
        description = form.description.data
        output = return_recommendations(description, classifier)
        return render_template('classify.html', form=form, first=f'{output[0][0]}   ({output[0][1]}%)', 
                                                           second=f'{output[1][0]}   ({output[1][1]}%)', 
                                                           third=f'{output[2][0]}   ({output[2][1]}%)')
    return render_template('classify.html', form=form, first='', second='', third='')

if __name__ == '__main__':
    app.run(debug=True)