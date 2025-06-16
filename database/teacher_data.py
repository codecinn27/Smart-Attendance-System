import sqlite3

def insert_teacher_data():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    # Clear existing data
    cursor.execute("DELETE FROM classes")

    # List of classes and teachers
    classes = [
        ('Computer Vision And Pattern Recognition', 'Dr Nik Mohd Zarifie'),
        ('High Performance Computing', 'Dr Sani Irwan'),
        ('Engineering Economy', 'Dr Al Amin'),
    ]

    # Insert new data
    for class_name, teacher in classes:
        cursor.execute("INSERT INTO classes (class_name, teacher) VALUES (?, ?)", (class_name, teacher))

    conn.commit()
    conn.close()
    print("✅ Teacher data inserted successfully!")

if __name__ == "__main__":
    insert_teacher_data()
