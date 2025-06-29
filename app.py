# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
from io import BytesIO
from xhtml2pdf import pisa
from datetime import datetime
from flask import redirect, request

# Flask setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

model = load_model('poultry_disease_model.h5')
CLASS_NAMES = ['Coccidiosis', 'Healthy', 'Newcastle', 'Salmonella']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash("No file part in request.")
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash("No file selected.")
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        return render_template('index.html',
                               prediction=predicted_class,
                               confidence=round(confidence, 2),
                               image_path=url_for('static', filename='uploads/' + filename),
                               filename=filename)

    else:
        flash("Invalid file type. Please upload an image file.")
        return redirect(url_for('index'))

@app.route('/download_report')
def download_report():
    prediction = request.args.get('prediction')
    confidence = request.args.get('confidence')
    image_file = request.args.get('filename')
    abs_path = os.path.join(app.root_path, 'static/uploads', image_file)

    html = render_template('report_template.html', prediction=prediction,
                           confidence=confidence, image_path=abs_path,
                           timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    pdf = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf)

    if not pisa_status.err:
        pdf.seek(0)
        return send_file(pdf, download_name="poultry_report.pdf", as_attachment=True)
    else:
        return "Error generating PDF"

@app.route('/training')
def training():
    disease_info = {
        "Coccidiosis": {
            "symptoms": "Bloody droppings, weight loss, ruffled feathers",
            "treatment": "Amprolium, Sulfa drugs",
            "management": "Maintain dry litter, use medicated feed"
        },
        "Newcastle": {
            "symptoms": "Coughing, sneezing, twisted neck",
            "treatment": "Supportive care, antibiotics for secondary infections",
            "management": "Vaccination, biosecurity measures"
        },
        "Salmonella": {
            "symptoms": "Diarrhea, weakness, reduced egg production",
            "treatment": "Antibiotics under veterinary guidance",
            "management": "Clean water/feed, rodent control"
        }
    }
    return render_template("training.html", disease_info=disease_info)

@app.route('/research')
def research():
    query = request.args.get('query', '')
    return redirect(f"https://scholar.google.com/scholar?q=Veterinary+{query}")

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
