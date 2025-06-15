import sqlite3
from datetime import datetime

def get_class_id_by_value(class_name):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT class_id FROM classes WHERE class_name = ?", (class_name,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def get_student_id_by_name(name):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT student_id FROM students WHERE name = ?", (name,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def get_enrollment_id(student_id, class_id):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM enrollments WHERE student_id = ? AND class_id = ?", (student_id, class_id))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def mark_attendance(name, class_name):
    student_id = get_student_id_by_name(name)
    class_id = get_class_id_by_value(class_name)
    if student_id is None or class_id is None:
        print(f"[WARN] Unknown student or class: {name}, {class_name}")
        return

    enrollment_id = get_enrollment_id(student_id, class_id)
    if enrollment_id is None:
        print(f"[WARN] Student not enrolled in class: {name}, {class_name}")
        return

    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    try:
        cursor.execute("""
            INSERT INTO attendance (enrollment_id, date, status, clock_in_time)
            VALUES (?, ?, 'Present', ?)
        """, (enrollment_id, today, time_now))
        conn.commit()
        print(f"[INFO] Attendance marked for {name} in {class_name} at {time_now}")
    except sqlite3.IntegrityError:
        print(f"[INFO] Already marked for today: {name} in {class_name}")
    finally:
        conn.close()


def save_attendance_to_db(student_id, class_id):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    # Get enrollment ID
    cursor.execute("""
        SELECT id FROM enrollments
        WHERE student_id = ? AND class_id = ?
    """, (student_id, class_id))
    result = cursor.fetchone()
    if not result:
        conn.close()
        print("[DB] Student not enrolled in this class.")
        return

    enrollment_id = result[0]
    date_today = datetime.now().strftime('%Y-%m-%d')
    clock_in_time = datetime.now().strftime('%H:%M:%S')

    # Check if attendance already exists
    cursor.execute("""
        SELECT id FROM attendance WHERE enrollment_id = ? AND date = ?
    """, (enrollment_id, date_today))
    if cursor.fetchone():
        conn.close()
        return  # Already marked today

    cursor.execute("""
        INSERT INTO attendance (enrollment_id, date, status, clock_in_time)
        VALUES (?, ?, ?, ?)
    """, (enrollment_id, date_today, 'Present', clock_in_time))

    conn.commit()
    conn.close()
    print(f"[DB] Attendance saved for student {student_id} in class {class_id}")