import tensorflow as tf
from flask import Flask, render_template, request, url_for, redirect,Response,send_from_directory,flash
import cv2
import os
from werkzeug.utils import secure_filename
from os.path import join, dirname, realpath
app=Flask(__name__)
UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static/uploads/') # where uploaded files are stored
ALLOWED_EXTENSIONS = set(['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG', 'gif', 'GIF']) # models support png and gif as well

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # max upload - 10MB
app.secret_key = 'secret'

def allowed_file(filename):
    	return '.' in filename and \
		   filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/assessment', methods=['GET', 'POST'])
def upload_and_classify():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(url_for('home'))
		
		file = request.files['file']

		# if user does not select file, browser also
		# submit a empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(url_for('home'))

		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename) # used to secure a filename before storing it directly on the filesystem
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			# return redirect(url_for('uploaded_file',
			#                         filename=filename))
			filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			model_results = preprocess(filepath)

			return render_template('result.html', result=model_results, filename=filename)
	
	flash('Invalid file format - please try your upload again.')
	return redirect(url_for('home'))

@app.route('/')
def home():
    return render_template('layout.html')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/front')
def front():
    return render_template('front.html')

@app.route('/<result>',methods=['POST'])
def result(result):
    return render_template('result.html',result=result)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)



def preprocess(image):
    img=cv2.imread(image)
    img=cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA)
    img=img/255.
    img=tf.expand_dims(img,axis=0)
    c=model(img)
    return c
    

def model(img):
    listy=[]
    model_1=tf.keras.models.load_model('data1a (1).h5',compile=True)
    model_2=tf.keras.models.load_model('data2a (2).h5',compile=True)
    model_3=tf.keras.models.load_model('data3a (2).h5',compile=True)
    
    a=model_1.predict(img)
    b=model_2.predict(img)
    c=model_3.predict(img)
    
    a=a.round()
    b=b.argmax()
    c=c.argmax()
    
    if a==0:
        damage=0
    else:
        damage=1
        
    if b==0:
        loc='front'
    elif b==1:
        loc='rear'
    else:
        loc='side'
    
    if c==0:
        severe='minor'
    elif c==1:
        severe='moderate'
    else:
        severe='critical'
        
    if damage ==1:
     result = {'gate1': 'Car validation check: ', 
               'gate1_result': 1, 
		'gate2': 'Damage presence check: ',
		'gate2_result': 0,
		'gate2_message': {0: 'Are you sure that your car is damaged? Please retry your submission.',
		1: 'Hint: Try zooming in/out, using a different angle or different lighting.'},
		'location': None,
		'severity': None,
		'final': 'Damage assessment unsuccessful!'}
     return result
    else:
        result = {'gate1': 'Car validation check: ', 
        'gate1_result': 1, 
        'gate2': 'Damage presence check: ',
        'gate2_result': 1,

        'location': loc,
        'severity': severe,
        'final': 'Damage assessment complete!'}
        return result
    
        
        
    

    



    

if __name__=="__main__":
    app.run(debug=True)
