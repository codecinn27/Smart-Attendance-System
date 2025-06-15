import sqlite3


conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()

cursor.execute("DELETE FROM classes")
classes = [
    ('Computer Vision And Pattern Recognition', 'Dr Nik Mohd Zarifie'),
    ('High Performance Computing', 'Dr Sani Irwan'),
    ('Engineering Economy', 'Dr Al Amin'),
]

for class_name, teacher in classes:
    cursor.execute("INSERT INTO classes (class_name, teacher) VALUES (?, ?)", (class_name, teacher))

conn.commit()

print("Teacher Data inserted successfully!")