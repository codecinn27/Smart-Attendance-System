from table_creation import create_database, clear_folders
from delete_attendance import delete_all_attendance
from teacher_data import insert_teacher_data

if __name__ == "__main__":
    delete_all_attendance()
    create_database()
    clear_folders()
    insert_teacher_data()