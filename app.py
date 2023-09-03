import logging
import os
import pymysql
import bcrypt
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, session
from wtforms import Form, StringField, PasswordField, validators, SelectField, FileField
import secrets
import re
from wtforms import Form, StringField, PasswordField, validators
from functools import wraps
import base64
import dlib
import cv2
import numpy as np



class LoginForm(Form):
    username = StringField('Username', [validators.InputRequired()])
    password = PasswordField('Password', [validators.InputRequired()])


app = Flask(__name__)
secret_key = secrets.token_hex(16)

# Use the generated secret key in your Flask app
app.secret_key = secret_key

# Function to connect to the MySQL database and create tables if they don't exist
def connect_to_database():
    try:
        connection = pymysql.connect(
            host='localhost',
            user='brian',
            password='orbit',
            db='spacex',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        with connection.cursor() as cursor:
            # Create users table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) NOT NULL,
                    password VARCHAR(255) NOT NULL
                )
            ''')
            # Create registered_clients table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS registered_clients (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    user_class VARCHAR(255) NOT NULL,
                    picture_url VARCHAR(255) DEFAULT NULL,
                    hashed_password VARCHAR(255) NOT NULL
                )
            ''')
            # Commit the table creation
            connection.commit()
        return connection
    except pymysql.MySQLError as e:
        # Handle database connection errors here
        print(f"Database error: {e}")
        return None

# Create a global database connection
db_connection = connect_to_database()

# Function to hash passwords securely
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')

# Function to verify user credentials
def verify_user(username, password):
    connection = connect_to_database()
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            return user
    return None

# Function to validate registration input, including password strength
def is_valid_gate_registration(name, user_class, picture_data, password):
    # Check if the name is not empty (you can add more validation if needed)
    if not name:
        return False

    # Check if the password is at least 5 characters long
    if len(password) < 5:
        return False

    # Check if the password contains at least one special character and one numeric value
    if not (re.search(r'[!@#$%^&*()_+{}\[\]:;<>,.?~\\\-+=]', password) and re.search(r'\d', password)):
        return False

    # Add more validation rules if needed

    return True

# Function to insert gate registration data into the database
def insert_gate_registration_data(name, user_class, hashed_password, picture_url):
    connection = connect_to_database()
    with connection.cursor() as cursor:
        try:
            cursor.execute(
                "INSERT INTO registered_clients (name, user_class, hashed_password, picture_url) VALUES (%s, %s, %s, %s)",
                (name, user_class, hashed_password, picture_url)
            )
            connection.commit()
            return cursor.lastrowid  # Return the ID of the inserted record
        except pymysql.MySQLError as e:
            # Handle database insertion errors here
            print(f"Database error: {e}")
            return None

# Function to save uploaded pictures
def save_picture(picture_file):
    if not picture_file:
        return None  # Handle the case where no picture is provided

    # Specify the directory where you want to save the pictures
    upload_folder = 'path/to/your/upload/folder'
    
    # Ensure the upload folder exists, create it if necessary
    os.makedirs(upload_folder, exist_ok=True)

    # Generate a unique file name (you can use a library like UUID)
    picture_filename = 'unique_filename.png'  # Replace with your logic

    # Build the full path to save the picture
    picture_path = os.path.join(upload_folder, picture_filename)

    # Save the picture to the specified path
    picture_file.save(picture_path)

    # Return the relative or absolute URL of the saved picture
    return '/static/uploads/' + picture_filename  # Replace with your actual URL

# Define the gate registration form
class GateRegistrationForm(Form):
    name = StringField('Name', [validators.InputRequired()])
    user_class = SelectField('Class', [validators.InputRequired()], choices=[
        ('student', 'Student'),
        ('staff', 'Staff'),
        ('intern', 'Intern'),
        ('support_staff', 'Support Staff'),
        ('visitor', 'Visitor')
    ])
    picture = FileField('Profile Picture', [validators.Optional()])
    password = PasswordField('Password', [validators.InputRequired()])

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')


# Define a custom decorator function to check if the user is logged in
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if the user is logged in (you can customize this condition)
        if not is_logged_in():
            flash("Please log in to access the dashboard.")
            return redirect(url_for('login'))  # Redirect to the login page
        return f(*args, **kwargs)
    return decorated_function

# Apply the login_required decorator to the /dashboard route
@app.route('/dashboard')
@login_required
def dashboard():
    # Your dashboard logic here
    return render_template('dashboard.html')



# Route for the registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = GateRegistrationForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        user_class = form.user_class.data
        password = form.password.data
        picture_url = None

        # Check if a picture file was provided and save it
        if 'picture' in request.files:
            picture_file = request.files['picture']
            if allowed_file(picture_file.filename):
                picture_url = save_picture(picture_file)
            else:
                flash("Invalid file type for profile picture. Please use JPG or PNG.")
                return render_template('register.html', form=form)

        # Hash the password
        hashed_password = hash_password(password)

        # Insert registration data into the database
        user_id = insert_gate_registration_data(name, user_class, hashed_password, picture_url)

        if user_id:
            flash("Registration successful! You can now log in.")
            return redirect(url_for('login'))
        else:
            flash("Registration failed. Please try again later.")

    return render_template('register.html', form=form)

def get_user_type(username):
    connection = connect_to_database()
    with connection.cursor() as cursor:
        # Check if the username exists in the 'registered_clients' table
        cursor.execute("SELECT * FROM registered_clients WHERE name = %s", (username,))
        result = cursor.fetchone()

        if result:
            # Username found in 'registered_clients', return the user type
            return 'registered_clients'
        else:
            # Username not found in 'registered_clients', return 'users' as default
            return 'users'
        
        
# Function to get the user's ID from the database based on username
def get_user_id(username):
    connection = connect_to_database()
    with connection.cursor() as cursor:
        cursor.execute("SELECT id FROM registered_clients WHERE name = %s", (username,))
        result = cursor.fetchone()
        if result:
            return result['id']
    return None  # Return None if the username is not found or an error occurs




@app.route('/login', methods=['GET', 'POST'])
def login():
    if is_logged_in():
        return redirect(url_for('dashboard'))

    form = LoginForm(request.form)
    if request.method == 'POST' and form.validate():
        username = form.username.data
        password = form.password.data

        # Check if the username exists in the 'users' table
        user = verify_user(username, password)

        if user:
            # User found in 'users' table, set session variables and redirect to dashboard
            session['user_id'] = user['id']
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            # Username not found in 'users', try 'registered_clients'
            user_type = get_user_type(username)

            if user_type == 'registered_clients':
                # Attempt to authenticate against 'registered_clients' table
                user_id = get_user_id(username)

                if user_id:
                    # Fetch the user's name from 'registered_clients'
                    connection = connect_to_database()
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT name FROM registered_clients WHERE id = %s", (user_id,))
                        result = cursor.fetchone()
                        if result:
                            user_name = result['name']
                            # Set session variables and redirect to user_dashboard
                            session['user_id'] = user_id
                            session['username'] = user_name
                            return redirect(url_for('user_dashboard'))

            flash("Login failed. Please check your credentials.")

    return render_template('login.html', form=form)
from flask import session, send_file


@app.route('/user_dashboard')
def user_dashboard():
    # Check if the user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    # Get the user ID from the session
    user_id = session['user_id']

    # Use the user ID to query user data from the 'registered_clients' table
    connection = connect_to_database()
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM registered_clients WHERE id = %s", (user_id,))
        user_data = cursor.fetchone()

    user_name = "User"  # Default values in case user_data is not found
    user_user_class = "Unknown"
    image_path = None  # Initialize image_path as None

    if user_data:
        user_name = user_data['name']
        user_user_class = user_data['user_class']

        # Decode the image data (assuming 'picture_data' is the column containing the image)
        image_data = user_data['picture_data']
        if image_data:
            # Decode the image data and pass it directly to the template
            user_image = base64.b64encode(image_data).decode('utf-8')
        else:
            # Handle the case where user data is not found or image is not available
            user_image = None

    # Render the 'user_dashboard.html' template and pass user data and image path
    return render_template('user_dashboard.html', user_name=user_name, user_user_class=user_user_class, user_image=user_image)

@app.route('/logout')
def logout():
    # Implement your logout logic he# Route for the login page
    session.clear()  # Clear the user session
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))  # Redirect to the login page after logout


from flask import request, redirect, url_for

@app.route('/verify_id', methods=['POST'])
def verify_id():
    # Check if a file was uploaded
    if 'face' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['face']

    # Check if the file is empty
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        # Process the uploaded facial data here (e.g., verification)

        # Once processed, you can redirect to a success or result page
        flash('Facial data uploaded and verified successfully')
        return redirect(url_for('success_page'))  # Replace 'success_page' with your desired route

    # If the file doesn't have an allowed extension
    flash('Invalid file type. Please upload a valid image file.')
    return redirect(request.url)


# Function to check if an uploaded file is allowed (e.g., file type)
def allowed_file(filename):
    # Implement your allowed file type logic here
    # For example, you can check if the filename ends with a specific extension (e.g., .jpg, .png)
    allowed_extensions = {'jpg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


# Define a route for the gate_register page
@app.route('/gate_register', methods=['GET'])
def gate_register():
    # Render the registration form template
    return render_template('gate_register.html')

# Define a route to handle the registration form submission
@app.route('/register_user', methods=['POST'])
def register_user():
    # Get the form data from the request
    name = request.form.get('name')
    user_class = request.form.get('user_class')
    password = request.form.get('password')
    picture = request.files.get('picture')

    # Check if the username already exists
    if is_user_already_registered(name):
        flash("Username already exists. Please choose a different one.")
        return redirect(url_for('gate_register'))

    # Check if the password and confirm_password match (you can add more validation)
    confirm_password = request.form.get('confirm_password')
    if password != confirm_password:
        flash("Passwords do not match. Please try again.")
        return redirect(url_for('gate_register'))

    # Check if the user input is valid (you can add more validation)
    if not is_valid_gate_registration(name, user_class, picture, password):
        flash("Invalid registration data. Please check your input.")
        return redirect(url_for('gate_register'))

    # Hash the password
    hashed_password = hash_password(password)

    # Save the profile picture (if provided)
    picture_url = save_picture(picture)

    # Insert the user data into the registered_clients table
    insert_gate_registration_data(name, user_class, hashed_password, picture_url)

    flash("Registration successful! You can now log in.")
    return redirect(url_for('login'))  # Redirect to the login page after successful registration
# Function to check if a user is logged in
def is_logged_in():
    return 'user_id' in session

# Function to check if a user is already registered
def is_user_already_registered(name):
    connection = connect_to_database()
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM registered_clients WHERE name = %s", (name,))
        existing_user = cursor.fetchone()
        return existing_user is not None
    
    
# Load dlib's pre-trained models
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")




def generate_templates(image_base64):
    # Decode base64 image to bytes and convert to an image array
    image_bytes = base64.b64decode(image_base64)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sample_templates = []
    
    # Detect faces in the image
    faces = dlib.get_frontal_face_detector()(rgb_image)
    
    for face_rect in faces:
        landmarks = shape_predictor(rgb_image, face_rect)
        shape = dlib.get_face_chip(rgb_image, landmarks)
        descriptor = np.array(face_recognizer.compute_face_descriptor(shape))
        sample_templates.append(descriptor)
    
    return sample_templates

def compare_templates(template1, template2, threshold=0.49):
    # Compute the similarity between two templates
    similarity = np.linalg.norm(template1 - template2)
    return similarity > threshold

@app.route('/verify_id', methods=['POST'])
@login_required
def verify_id():
    result = {'result': 'error', 'message': ''}
    
    # Check if a file was uploaded
    if 'face' not in request.files:
        result['message'] = 'No file part'
        return jsonify(result)

    file = request.files['face']

    # Check if the file is empty
    if file.filename == '':
        result['message'] = 'No selected file'
        return jsonify(result)

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        # Process the uploaded facial data and compare with the sample
        sample_image_base64 = session.get('sample_image_base64')
        if not sample_image_base64:
            result['message'] = 'Sample image not found in session'
            return jsonify(result)

        sample_templates = generate_templates(sample_image_base64)
        uploaded_image = cv2.imdecode(np.frombuffer(file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        uploaded_templates = generate_templates(uploaded_image)

        authenticated = compare_templates(sample_templates[0], uploaded_templates[0])
           

        if authenticated:
            result['result'] = 'granted'
            result['message'] = 'Authentication successful'
        else:
            result['message'] = 'Authentication failed'

        # Log the authentication result
        username = session.get('username')  # Replace with your logic to get the username
        if username:
            log_message = f"User '{username}' authenticated: {result['result']}"
            logging.info(log_message)
        else:
            logging.warning("Username not found in session")

    else:
        result['message'] = 'Invalid file type. Please upload a valid image file.'

    return jsonify(result)

# Function to check if an uploaded file is allowed (e.g., file type)
def allowed_file(filename):
    # Implement your allowed file type logic here
    # For example, you can check if the filename ends with a specific extension (e.g., .jpg, .png)
    allowed_extensions = {'jpg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
