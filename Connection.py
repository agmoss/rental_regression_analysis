import mysql.connector

class Connection:

    @staticmethod
    def db_config():
        """Setup"""

        import json

        with open('config.json', 'r') as f:
            config = json.load(f)

        host = config['DATABASE_CONFIG']['host']
        user = config['DATABASE_CONFIG']['user']
        password = config['DATABASE_CONFIG']['password']
        db = config['DATABASE_CONFIG']['dbname']

        return host, user, password, db

    @staticmethod
    def connect():
        """ Connect to MySQL database """

        while True:

            try:
                host, user, password, db = Connection.db_config()

                conn = mysql.connector.connect(host=host,
                                            database=db,
                                            user=user,
                                            password=password)

            except Exception:
                raise

            else:
                return conn