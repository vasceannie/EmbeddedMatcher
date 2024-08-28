import psycopg2
from psycopg2 import OperationalError

def create_connection():
    try:
        connection = psycopg2.connect(
            user="overlord",
            password="password",
            host="192.168.0.90",
            port="5432",
            database="ars_spend_class",
            connect_timeout=10
        )
        print("Connection to PostgreSQL DB successful")
        connection.close()
    except OperationalError as e:
        print(f"The error '{e}' occurred")

if __name__ == "__main__":
    create_connection()