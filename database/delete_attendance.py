import sqlite3

def delete_all_attendance():
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()

        # Delete all rows from attendance table
        cursor.execute("DELETE FROM attendance")
        conn.commit()

        print("[INFO] ✅ All attendance records deleted successfully.")

    except Exception as e:
        print(f"[ERROR] Failed to delete records: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    delete_all_attendance()
