import time
import psycopg2
import os

host = os.getenv("DB_HOST", "db")
port = int(os.getenv("DB_PORT", 5432))

for i in range(20):
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user="admin",
            password="admin",
            dbname="logs"
        )
        conn.close()
        print("db ready")
        break
    except Exception as e:
        print("waiting for db...", i, e)
        time.sleep(2)
else:
    raise RuntimeError("db not ready")
