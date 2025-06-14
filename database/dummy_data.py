import sqlite3
from datetime import date

conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()

# Insert dummy students
students = [
    ('Alice', 20),
    ('Bob', 21),
    ('Charlie', 19),
    ('Diana', 22),
]

for name, age in students:
    cursor.execute("INSERT INTO students (name, age) VALUES (?, ?)", (name, age))

# Delete existing classes to avoid UNIQUE constraint issues
cursor.execute("DELETE FROM classes")
classes = [
    ('Maths', 'Mr. Smith'),
    ('English', 'Ms. Johnson'),
    ('Science', 'Dr. Williams'),
]

for class_name, teacher in classes:
    cursor.execute("INSERT INTO classes (class_name, teacher) VALUES (?, ?)", (class_name, teacher))

conn.commit()

# Fetch students and classes
cursor.execute("SELECT student_id, name FROM students")
students_db = cursor.fetchall()

cursor.execute("SELECT class_id, class_name FROM classes")
classes_db = cursor.fetchall()

# Enrollments
enrollments = [
    ('Alice', 'Science'),
    ('Bob', 'Maths'),
    ('Charlie', 'English'),
    ('Diana', 'Science'),
    ('Alice', 'Maths'),
]

for student_name, class_name in enrollments:
    student_id = next(sid for sid, name in students_db if name == student_name)
    class_id = next(cid for cid, cname in classes_db if cname == class_name)
    cursor.execute("INSERT INTO enrollments (student_id, class_id) VALUES (?, ?)", (student_id, class_id))

conn.commit()

# Attendance records with clock_in and clock_out times
attendance_records = [
    # (student_name, class_name, date, present (1/0), clock_in_time, clock_out_time)
    ('Alice', 'Science', '2025-06-16', 1, '08:05:00', '12:00:00'),
    ('Bob', 'Maths', '2025-06-16', 0, None, None),
    ('Charlie', 'English', '2025-06-16', 1, '08:10:30', '11:50:00'),
    ('Diana', 'Science', '2025-06-16', 1, '08:00:00', '12:10:00'),

    ('Alice', 'Maths', '2025-06-17', 1, '08:03:00', '11:55:00'),
    ('Bob', 'Maths', '2025-06-17', 1, '08:07:45', '12:05:00'),
]

for student_name, class_name, att_date, present, clock_in_time, clock_out_time in attendance_records:
    student_id = next(sid for sid, name in students_db if name == student_name)
    class_id = next(cid for cid, cname in classes_db if cname == class_name)

    cursor.execute(
        "SELECT id FROM enrollments WHERE student_id = ? AND class_id = ?",
        (student_id, class_id)
    )
    enrollment_row = cursor.fetchone()
    if enrollment_row is None:
        print(f"No enrollment found for {student_name} in {class_name}")
        continue
    enrollment_id = enrollment_row[0]

    cursor.execute(
        "INSERT INTO attendance (enrollment_id, date, status, clock_in_time, clock_out_time) VALUES (?, ?, ?, ?, ?)",
        (enrollment_id, att_date, 'present' if present else 'absent', clock_in_time, clock_out_time)
    )

conn.commit()
conn.close()

print("Dummy data inserted successfully!")
