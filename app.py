# ================= IMPORT =================
from flask import Flask, render_template, request, jsonify, session, redirect
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sqlite3
import uuid
import json
import cv2
from werkzeug.security import generate_password_hash, check_password_hash
import base64

print("DB PATH:", os.path.abspath("database.db"))

# ================= APP =================
app = Flask(__name__)
app.secret_key = "supersecret"

# ================= MODEL =================
model = tf.keras.models.load_model("model/skin_model.h5")

with open("model/classes.json", "r") as f:
    class_indices = json.load(f)

classes = {v: k for k, v in class_indices.items()}

with open("model/disease_info.json") as f:
    disease_info = json.load(f)

# ================= PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= DB =================
conn = sqlite3.connect(os.path.join(os.getcwd(), "database.db"), check_same_thread=False)
c = conn.cursor()

# 🔥 ADD THIS HERE
c.execute('''
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT,
    result TEXT,
    confidence REAL
)
''')
conn.commit()

# USERS
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password TEXT,
    dob TEXT,
    gender TEXT
)
''')

# HISTORY
c.execute('''
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    disease TEXT,
    confidence REAL,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hidden INTEGER DEFAULT 0
)
''')

# ensure hidden column (safe)
try:
    c.execute("ALTER TABLE history ADD COLUMN hidden INTEGER DEFAULT 0")
except:
    pass

conn.commit()

# ================= GLOBAL =================
last_data = {}

# ================= HEATMAP =================
def generate_heatmap(img_array, model):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        return None

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap

# ================= HOME =================
@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template("index.html")

# ================= LOGIN =================
@app.route('/login', methods=['GET','POST'])
def login():

    if request.method == "POST":

        email = request.form['email']
        password = request.form['password']

        # 🔥 user fetch yahi hona chahiye
        c.execute("SELECT * FROM users WHERE email=?", (email,))
        user = c.fetchone()

        # 🔥 yahi check hoga
        if user and check_password_hash(user[3], password):

            session['user_id'] = user[0]
            session['user_email'] = user[2]
            session['name'] = user[1]
            session['dob'] = user[4]
            session['gender'] = user[5]

            # 🔥 ADMIN CHECK
            if email.strip().lower() == "admin@gmail.com":
                session['is_admin'] = True
                return redirect('/admin')
            else:
                session['is_admin'] = False
                return redirect('/')

        return render_template("login.html", error="Invalid login")

    return render_template("login.html")

# ================= SIGNUP =================
@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        dob = request.form['dob']
        gender = request.form['gender']

        c.execute("INSERT INTO users (name,email,password,dob,gender) VALUES (?,?,?,?,?)",
                  (name,email,password,dob,gender))
        conn.commit()

        return redirect('/login')

    return render_template("signup.html")

# ================= LOGOUT =================
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# ================= PREDICT (WEBCAM + UPLOAD SUPPORT) =================
@app.route('/predict', methods=['POST'])
def predict():
    global last_data

    if 'file' not in request.files:
        return jsonify({"error": "No file"})

    file = request.files['file']

    if file.filename == "":
        return jsonify({"error": "Empty file"})

    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img = Image.open(filepath).convert("RGB").resize((224,224))
    arr = np.array(img)/255.0
    arr = arr.reshape(1,224,224,3)

    pred = model.predict(arr)
    idx = np.argmax(pred)
    result = classes[idx]
    confidence = round(float(np.max(pred)*100),2)

    info = disease_info.get(result, {"desc":"No info","treatment":"Consult doctor"})

    heatmap_url = None
    try:
        heatmap = generate_heatmap(arr, model)
        if heatmap is not None:
            heatmap = cv2.resize(heatmap, (224,224))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            original = cv2.imread(filepath)
            original = cv2.resize(original, (224,224))

            superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

            heatmap_filename = "heatmap_" + filename
            heatmap_path = os.path.join(UPLOAD_FOLDER, heatmap_filename)

            cv2.imwrite(heatmap_path, superimposed)
            heatmap_url = f"/static/uploads/{heatmap_filename}"

    except Exception as e:
        print("Heatmap error:", e)

    # save history (same logic)
    user_id = session.get('user_id') or 1

    c.execute('''
    INSERT INTO history (user_id, disease, confidence, hidden)
    VALUES (?, ?, ?, 0)
    ''', (user_id, result, confidence))
    
    # 🔥🔥 NEW CODE ADD (ADMIN PANEL KE LIYE)
    c.execute('''
    INSERT INTO reports (email, result, confidence)
    VALUES (?, ?, ?)
    ''', (session.get('user_email'), result, confidence))
    
    print("🔥 INSERT RUNNING")
    print("EMAIL:", session.get('user_email'))
    print("RESULT:", result)
    print("CONF:", confidence)

    conn.commit()

    last_data = {
        "result": result,
        "confidence": confidence,
        "desc": info["desc"],
        "treatment": info["treatment"],
        "image": f"/static/uploads/{filename}",
        "heatmap": heatmap_url
    }

    return jsonify(last_data)

# ================= REPORT =================
@app.route('/report')
def report():
    global last_data

    if not last_data:
        return "No report available"

    return render_template("report.html",
        name=session.get('name'),
        dob=session.get('dob'),
        gender=session.get('gender'),
        data=last_data
    )

# ================= DELETE =================
@app.route('/delete_history/<int:id>', methods=['POST'])
def delete_history(id):
    user_id = session.get('user_id')
    user_email = session.get('user_email')

    if user_email == "admin@gmail.com":
        c.execute("DELETE FROM history WHERE id=?", (id,))
    else:
        c.execute("UPDATE history SET hidden=1 WHERE id=? AND user_id=?", (id, user_id))

    conn.commit()
    return redirect('/history')

# ================= HISTORY =================
@app.route('/history')
def history():
    user_id = session.get('user_id') or 1
    user_email = session.get('user_email')

    if user_email == "admin@gmail.com":
        c.execute("SELECT * FROM history ORDER BY date DESC")
    else:
        c.execute("SELECT * FROM history WHERE user_id=? AND hidden=0 ORDER BY date DESC", (user_id,))

    data = c.fetchall()

    return render_template('history.html', history=data)

# =================WEBCAM===================
@app.route('/webcam')
def webcam():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template("webcam.html")

@app.route('/admin')
def admin():

    if not session.get('is_admin'):
        return "Access Denied ❌"

    # total users
    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]

    # total predictions (history use kar)
    c.execute("SELECT COUNT(*) FROM history")
    total_reports = c.fetchone()[0]

    predictions = total_reports

    # data table
    c.execute('''
    SELECT users.email, history.disease, history.confidence, history.id
    FROM history
    JOIN users ON users.id = history.user_id
    ''')

    data = c.fetchall()
    
   # 🔥 DAILY USAGE DATA (ADD THIS)
    c.execute('''
    SELECT DATE('now'), COUNT(*)
    FROM history
    ''')

    daily_data = c.fetchall()

    return render_template("admin.html",
        data=data,
        total_users=total_users,
        total_reports=total_reports,
        daily_activity=total_reports,
        predictions=predictions,
        daily_data=daily_data,
   )
    
@app.route('/admin/chart-data')
def chart_data():

    c.execute("""
    SELECT DATE(login_time), COUNT(*)
    FROM login_history
    GROUP BY DATE(login_time)
    """)

    data = c.fetchall()

    labels = [row[0] for row in data]
    values = [row[1] for row in data]

    return jsonify({"labels": labels, "values": values})    

@app.route('/admin/users')
def admin_users():

    if not session.get('is_admin'):
        return "Access Denied"

    c.execute("SELECT id, name, email, dob, gender FROM users")
    users = c.fetchall()

    return render_template("admin_users.html", users=users)


@app.route('/admin/delete/<int:id>')
def delete_user(id):

    c.execute("DELETE FROM users WHERE id=?", (id,))
    conn.commit()

    return redirect('/admin/users')

@app.route('/test-insert')
def test_insert():
    c.execute("INSERT INTO reports (email, result, confidence) VALUES ('test@gmail.com','Acne',99)")
    conn.commit()
    return "Inserted Successfully"

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
    
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)    
    
