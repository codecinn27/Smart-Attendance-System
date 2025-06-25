from .table_creation import create_database, clear_folders
from .delete_attendance import delete_all_attendance
from .teacher_data import insert_teacher_data

def clearDatabase():
    try:
        delete_all_attendance()
        create_database()
        clear_folders()
        insert_teacher_data()
        return {"status": "success", "message": "Database reset successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}