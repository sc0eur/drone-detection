from flask import Flask, render_template, stream_with_context, request, Response, flash
from werkzeug.utils import secure_filename
import os
from os.path import join
from process import generate_pic

PEOPLE_FOLDER = os.path.join('static', 'imgs')


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
app.config['DOWLOAD_FOLDER'] = "./imgs"

def stream_template(template_name, **context):
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.disable_buffering()
    return rv

# @app.route('/')
# def upload_file():
#     print(request.method)
#     return render_template('upload.html')

@app.route('/', methods = ['GET', 'POST'])
def upload_file2():
    if request.method == 'GET':
        return render_template('upload.html')
    if request.method == 'POST':
      f = request.files['file']
      # print(type(f))
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
      print(app.config['UPLOAD_FOLDER'])
      # os.rename(f.filename, "./imgs/"+f.filename)
      base = generate_pic(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
      # full_filename = os.path.join(app.config['UPLOAD_FOLDER'], base)
      return Response(stream_with_context(stream_template("index.html",join=join, imgs = base, orig = f.filename, path = app.config['UPLOAD_FOLDER'], enumerate=enumerate)))

if __name__ == '__main__':
   app.run(debug = True)
