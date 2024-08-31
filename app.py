import streamlit as st
from PIL import Image
import cv2
import numpy as np
import mysql.connector
import os
import face_recognition
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
import pandas as pd

# Streamlit configuration
st.set_page_config(page_title="Face Attendance System", layout="wide", initial_sidebar_state="expanded")
st.title("🎯 Face Attendance System")

# Sidebar for navigation
with st.sidebar:
    selected_option = option_menu(
        "Main Menu", ["Home", "Add Student", "Train Model", "Take Attendance", "Attendance Report"],
        icons=["house", "person-plus-fill", "robot", "camera-video", "file-earmark-text"],
        menu_icon="cast", default_index=0, orientation="vertical"
    )

# Database connection function
def connect_db():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="",
            database="attendance"
        )
        if connection.is_connected():
            return connection
    except mysql.connector.Error as err:
        st.error(f"Database connection failed: {err}")
        return None

# Function to add student details and generate dataset
def add_student(name, age, department):
    if not name or not age.isdigit() or not department:
        st.warning("Please enter valid details in all fields.")
        return

    mydb = connect_db()
    if not mydb:
        return

    try:
        mycursor = mydb.cursor()
        mycursor.execute("SELECT COUNT(*) FROM students")
        student_id = mycursor.fetchone()[0] + 1

        sql = "INSERT INTO students (id, Name, Age, Department) VALUES (%s, %s, %s, %s)"
        val = (student_id, name, age, department)
        mycursor.execute(sql, val)
        mydb.commit()

        # Generate dataset
        cap = cv2.VideoCapture(0)
        dataset = []
        face_count = 0

        with st.spinner("Capturing student data... Please wait."):
            while len(dataset) < 200:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to open webcam!")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                if face_locations:
                    face_count += len(face_locations)
                    top, right, bottom, left = face_locations[0]
                    face = frame[top:bottom, left:right]
                    face = cv2.resize(face, (200, 200))
                    dataset.append(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
                    cv2.imshow("Cropped Face", face)
                    cv2.putText(face, f"Face Count: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        for i, face in enumerate(dataset):
            file_name_path = f"{data_dir}/student.{student_id}.{i+1}.jpg"
            cv2.imwrite(file_name_path, face)

        st.success(f"Student data captured successfully!")

    except mysql.connector.Error as err:
        st.error(f"Database error: {err}")
    finally:
        if mydb and mydb.is_connected():
            mydb.close()

# Function to train classifier
def train_classifier():
    data_dir = "data"
    if not os.path.exists(data_dir):
        st.warning("Data directory does not exist.")
        return

    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]

    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        student_id = int(os.path.split(image)[1].split(".")[1])
        faces.append(imageNp)
        ids.append(student_id)

    ids = np.array(ids)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    st.success("Training completed!")

# Function to take attendance
def take_attendance():
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    last_attendance_time = {}
    attendance_status = {}

    st.subheader("Attendance Register")
    status_table = st.empty()

    with st.spinner("Starting webcam..."):
        attendance_complete = False
        while not attendance_complete:
            ret, img = cap.read()
            if not ret:
                st.error("Failed to access webcam!")
                break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            try:
                face_locations = face_recognition.face_locations(img_rgb)
            except Exception as e:
                st.error(f"Error during face detection: {e}")
                break

            if not face_locations:
                continue

            attendance_complete = True

            for (top, right, bottom, left) in face_locations:
                face = img[top:bottom, left:right]
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                id, pred = clf.predict(face_gray)
                confidence = int(100 * (1 - pred / 300))

                mydb = connect_db()
                if not mydb:
                    st.error("Database connection failed!")
                    return

                try:
                    mycursor = mydb.cursor()
                    mycursor.execute("SELECT Name, Department FROM students WHERE id = %s", (id,))
                    s = mycursor.fetchone()
                    name, department = s if s else ("Unknown", "Unknown")

                    current_time = datetime.now()
                    if id in last_attendance_time:
                        time_difference = current_time - last_attendance_time[id]
                        if time_difference < timedelta(minutes=1):
                            attendance_complete = False
                            continue

                    if confidence > 74:
                        attendance_sql = "INSERT INTO attendance (id, Name, Date, Time, Department) VALUES (%s, %s, CURDATE(), CURTIME(), %s)"
                        attendance_val = (id, name, department)
                        mycursor.execute(attendance_sql, attendance_val)
                        mydb.commit()

                        last_attendance_time[id] = current_time
                        attendance_status[name] = "Present"
                    else:
                        attendance_status[name] = "Absent"
                        attendance_complete = False

                except mysql.connector.Error as err:
                    st.error(f"Database error: {err}")
                    attendance_status["Unknown"] = "Absent"
                finally:
                    if mydb and mydb.is_connected():
                        mydb.close()

                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0) if confidence > 74 else (0, 0, 255), 2)
                cv2.putText(img, name if confidence > 74 else "UNKNOWN", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if confidence > 74 else (0, 0, 255), 2)

            status_table.write(pd.DataFrame(list(attendance_status.items()), columns=["Student Name", "Status"]))

            stframe.image(img_rgb, channels="RGB", use_column_width=True)

    cap.release()
    stframe.empty()
    st.success("Attendance capture complete!")

# Function to view attendance report
def view_attendance_report():
    mydb = connect_db()
    if not mydb:
        return

    try:
        mycursor = mydb.cursor()
        mycursor.execute("""
            SELECT attendance.id, attendance.Name, attendance.Date, attendance.Time, students.Department 
            FROM attendance
            JOIN students ON attendance.id = students.id
        """)
        records = mycursor.fetchall()

        if records:
            st.subheader("Attendance Report")
            df = pd.DataFrame(records, columns=["ID", "Name", "Date", "Time", "Department"])
            st.write(df)
        else:
            st.info("No attendance records found.")

    except mysql.connector.Error as err:
        st.error(f"Database error: {err}")
    finally:
        if mydb and mydb.is_connected():
            mydb.close()

# Navigation logic
if selected_option == "Home":
    st.subheader("Welcome to the Face Attendance System!")
    st.write("Use the sidebar to navigate through different functionalities.")

elif selected_option == "Add Student":
    st.subheader("Add Student Details and Capture Data")
    name = st.text_input("Enter Name:")
    age = st.text_input("Enter Age:")
    department = st.text_input("Enter Department:")
    if st.button("Add Student"):
        add_student(name, age, department)

elif selected_option == "Train Model":
    st.subheader("Train Classifier")
    if st.button("Start Training"):
        train_classifier()

elif selected_option == "Take Attendance":
    st.subheader("Real-Time Attendance")
    take_attendance()

elif selected_option == "Attendance Report":
    view_attendance_report()
