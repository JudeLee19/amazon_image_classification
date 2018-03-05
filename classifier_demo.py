from flask import Flask, render_template, request,redirect
from collections import deque
from inference import Inference
import os
app = Flask(__name__)

UPLOAD_FOLDER = 'data/upload_file'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
DEMO_RESULT_FOLDER = 'data/upload_file/demo_result'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DEMO_RESULT_FOLDER'] = DEMO_RESULT_FOLDER

infer = Inference()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods = ['POST', 'GET'])
def index_html():
    html_str = deque([])
    with open("index.html", 'r') as f_handle:
        for line in f_handle:
            html_str.append(line)
    html_str = list(html_str)
    return "\n".join(html_str)

@app.route('/classify', methods = ['POST', 'GET'])
def classify_image():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            image_name = str(app.config['UPLOAD_FOLDER']) + '/' + str(file.filename)
            
            img_url = 'https://images-na.ssl-images-amazon.com/images/I/91MR26Sa4zL._AC_SR201,266_.jpg'
            # print(img_url)
            # class_name = infer.inference(image_name)
            class_name = infer.inference(img_url)
            # print(class_name)
            return class_name

if __name__ == '__main__':
    app.run("0.0.0.0", port=8850, debug=False)
