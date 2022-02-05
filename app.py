# backend server
from flask import Flask, after_this_request, request, send_file, send_from_directory, after_this_request
import os
import shlex
from random import randint
from mosaic_rectify import rectify

app = Flask(__name__, static_url_path='')

@app.route('/', methods=['GET'])
def home():
    return send_file('./website/index.html')



@app.route('/static/<path:path>', methods=['GET'])
def static_files(path):
    return send_from_directory('./website', path)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['jpg', 'png', 'jpeg', 'heic']

@app.route('/rectify', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        return 'No image found'
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filepath = os.path.join('/tmp', file.filename)
        file.save(filepath)

        converted_filepath = '/tmp/test_image_{}.jpg'.format(randint(0, 1e6))
        os.system('convert {} {}'.format(shlex.quote(filepath), converted_filepath))
        os.remove(filepath)

        coords = [
            (int(request.form['x1']), int(request.form['y1'])), 
            (int(request.form['x2']), int(request.form['y2'])), 
            (int(request.form['x3']), int(request.form['y3'])), 
            (int(request.form['x4']), int(request.form['y4'])) 
        ]

        @after_this_request
        def remove_temp_file(response):
            os.remove(converted_filepath)
            return response

        rectify(converted_filepath, coords)
        return send_file(converted_filepath)
    return 'Unknown image format'
    

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)

# get the image form the broser, get the file to rectifire, another file -> send back the file, delete the temporary files 
# upload image and put it into the file 


