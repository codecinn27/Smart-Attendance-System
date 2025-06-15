import sqlite3
import shutil
import os

def create_database():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    # Drop table if exists so we can recreate it with the right schema
    cursor.execute("DROP TABLE IF EXISTS students")
    cursor.execute("DROP TABLE IF EXISTS classes")
    cursor.execute("DROP TABLE IF EXISTS enrollments")
    cursor.execute("DROP TABLE IF EXISTS attendance")
    
    # Create students table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        student_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER NOT NULL
    )
    """)

    # Create classes table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS classes (
        class_id INTEGER PRIMARY KEY AUTOINCREMENT,
        class_name TEXT NOT NULL UNIQUE,
        teacher TEXT NOT NULL
    )
    """)

    # Create enrollments table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS enrollments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        class_id INTEGER NOT NULL,
        FOREIGN KEY(student_id) REFERENCES students(student_id),
        FOREIGN KEY(class_id) REFERENCES classes(class_id),
        UNIQUE(student_id, class_id)
    )
    """)

    # Create attendance table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        enrollment_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        status TEXT,
        clock_in_time TEXT,
        clock_out_time TEXT,
        FOREIGN KEY(enrollment_id) REFERENCES enrollments(id),
        UNIQUE(enrollment_id, date)
    )
    """)


    conn.commit()
    conn.close()
    print("Database and tables created successfully!")
    
def clear_folders():
    dataset_path = './static/dataset'
    trained_path = './static/trained'

    # Delete all contents inside ./static/dataset (keep dataset folder)
    if os.path.exists(dataset_path):
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

    # Delete the entire ./static/trained folder (and all contents)
    if os.path.exists(trained_path):
        shutil.rmtree(trained_path)
        

if __name__ == "__main__":
    create_database()
    clear_folders()
