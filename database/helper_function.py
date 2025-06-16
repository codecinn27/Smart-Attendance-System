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
        #print("[DB] Student not enrolled in this class.")
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
        return  False# Already marked today

    cursor.execute("""
        INSERT INTO attendance (enrollment_id, date, status, clock_in_time)
        VALUES (?, ?, ?, ?)
    """, (enrollment_id, date_today, 'Present', clock_in_time))

    conn.commit()
    conn.close()
    print(f"[DB] Attendance saved for student {student_id} in class {class_id}")
    return True # successfully saved

def has_attendance_today(student_id, class_id, date_str):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # Step 1: Get enrollment_id for this student in the class
    cursor.execute("""
        SELECT id FROM enrollments
        WHERE student_id = ? AND class_id = ?
    """, (student_id, class_id))
    result = cursor.fetchone()

    if not result:
        conn.close()
        return False  # Student not enrolled in this class

    enrollment_id = result[0]

    # Step 2: Check attendance table for a record today
    cursor.execute("""
        SELECT id FROM attendance
        WHERE enrollment_id = ? AND date = ?
    """, (enrollment_id, date_str))
    attendance_exists = cursor.fetchone() is not None

    conn.close()
    return attendance_exists



def save_attendance_to_db_10mins(student_id, class_id):
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
    now = datetime.strptime(clock_in_time, '%H:%M:%S')

    # Check last recorded time for today
    cursor.execute("""
        SELECT clock_in_time FROM attendance
        WHERE enrollment_id = ? AND date = ?
        ORDER BY clock_in_time DESC LIMIT 1
    """, (enrollment_id, date_today))
    
    last_row = cursor.fetchone()

    if last_row:
        last_time = datetime.strptime(last_row[0], '%H:%M:%S')
        time_diff = (now - last_time).total_seconds()

        if time_diff < 600:  # less than 10 minutes
            print(f"[DB] Skipped: Last attendance was {int(time_diff)}s ago (<10 mins).")
            conn.close()
            return

    # Insert new attendance record
    cursor.execute("""
        INSERT INTO attendance (enrollment_id, date, status, clock_in_time)
        VALUES (?, ?, ?, ?)
    """, (enrollment_id, date_today, 'Present', clock_in_time))

    conn.commit()
    conn.close()
    print(f"[DB] ✅ Attendance saved for student {student_id} in class {class_id} at {clock_in_time}")
