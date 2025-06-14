import sqlite3

def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            age INTEGER
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS enrollments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            enrolled_class TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    ''')
    conn.commit()
    conn.close()
    
def fetch_and_print(cursor, table_name):
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    print(f"\n--- {table_name.upper()} ---")
    for row in rows:
        print(row)

def display_enrollments_with_names(cursor):
    query = """
    SELECT e.id, s.name AS student_name, c.class_name
    FROM enrollments e
    JOIN students s ON e.student_id = s.student_id
    JOIN classes c ON e.class_id = c.class_id
    ORDER BY s.name, c.class_name
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    print("\n--- ENROLLMENTS (Student Name & Class Name) ---")
    for enrollment_id, student_name, class_name in rows:
        print(f"Enrollment ID: {enrollment_id} | Student: {student_name} | Class: {class_name}")

def display_attendance_with_names(cursor):
    query = """
    SELECT a.id, s.name AS student_name, c.class_name, a.date, a.status, a.clock_in_time, a.clock_out_time
    FROM attendance a
    JOIN enrollments e ON a.enrollment_id = e.id
    JOIN students s ON e.student_id = s.student_id
    JOIN classes c ON e.class_id = c.class_id
    ORDER BY a.date, s.name, c.class_name
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    print("\n--- ATTENDANCE (Student Name, Class Name, Date, Status, Clock-In, Clock-Out) ---")
    for attendance_id, student_name, class_name, date_, status, clock_in, clock_out in rows:
        clock_in_display = clock_in if clock_in else "N/A"
        clock_out_display = clock_out if clock_out else "N/A"
        print(f"Attendance ID: {attendance_id} | Student: {student_name} | Class: {class_name} | Date: {date_} | Status: {status} | Clock-In: {clock_in_display} | Clock-Out: {clock_out_display}")

def main():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    tables = ['students', 'classes', 'enrollments', 'attendance']
    for table in tables:
        fetch_and_print(cursor, table)

    display_enrollments_with_names(cursor)
    display_attendance_with_names(cursor)

    conn.close()

if __name__ == "__main__":
    main()
