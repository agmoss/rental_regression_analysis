import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()


class Connection:
    @staticmethod
    def db_config():
        """Setup"""

        password = os.getenv("PASSWORD")
        user = os.getenv("SERVERUSER")
        db = os.getenv("DB")
        host = os.getenv("HOST")

        return host, user, password, db

    @staticmethod
    def connect():
        """ Connect to MySQL database """

        while True:

            try:
                host, user, password, db = Connection.db_config()

                conn = mysql.connector.connect(
                    host=host, database=db, user=user, password=password
                )

            except Exception:
                raise

            else:
                return conn
