"""
Database access for the app.

Which database is used is decided by environment variables:

* MARIADB_URL (or MYSQL_URL)      -> MariaDB / MySQL, e.g. the company VPS
      mysql://USER:PASSWORD@HOST:3306/vendor_cards
      add ?ssl=required to encrypt the connection
* DATABASE_URL / POSTGRES_URL     -> Postgres (e.g. Neon on Vercel)
* nothing set                     -> the SQLite file vendors.db on this PC

The rest of the app writes SQLite-style SQL ("?" placeholders); this module
translates it for Postgres and MariaDB, so all three work with the same code.
"""

import os
import re
import sqlite3
import threading
from urllib.parse import parse_qs, unquote, urlparse

DB_PATH = None          # SQLite file, set by app.py

KIND = "sqlite"         # "sqlite" | "postgres" | "mysql"
URL = ""
IS_ONLINE = False       # a shared database server (not the local file)
IS_POSTGRES = False     # kept for older code


def configure(url=None):
    """Choose the database. Without a URL the environment decides."""
    global KIND, URL, IS_ONLINE, IS_POSTGRES
    if url is None:
        url = (os.getenv("MARIADB_URL") or os.getenv("MYSQL_URL")
               or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or "")
    url = url.strip()
    if url.startswith(("mysql://", "mariadb://", "mysql+pymysql://")):
        KIND = "mysql"
    elif url.startswith(("postgres://", "postgresql://")):
        KIND = "postgres"
    else:
        KIND, url = "sqlite", ""
    URL = url
    IS_ONLINE = KIND != "sqlite"
    IS_POSTGRES = IS_ONLINE          # "shared online database" for the app
    _local.__dict__.clear()


def label():
    return {"sqlite": "SQLite file", "postgres": "Online database (Postgres)",
            "mysql": "Online database (MariaDB)"}[KIND]


class Row(tuple):
    """A row that works like sqlite3.Row: row[0], row["name"], dict(row)."""

    def __new__(cls, values, names):
        row = tuple.__new__(cls, values)
        row._names = names
        return row

    def __getitem__(self, key):
        if isinstance(key, str):
            return tuple.__getitem__(self, self._names.index(key))
        return tuple.__getitem__(self, key)

    def keys(self):
        return list(self._names)


_local = threading.local()


# ------------------------------------------------------------ connections

def _pg_connection():
    import psycopg

    def row_factory(cursor):
        names = [d.name for d in (cursor.description or [])]
        return lambda values: Row(values, names)

    conn = getattr(_local, "conn", None)
    if conn is None or conn.closed or conn.broken:
        conn = psycopg.connect(URL, row_factory=row_factory, autocommit=False, connect_timeout=15)
        _local.conn = conn
    return conn


def _mysql_settings():
    parsed = urlparse(URL)
    query = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
    settings = {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 3306,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "database": unquote(parsed.path.lstrip("/")),
        "charset": "utf8mb4",
        "connect_timeout": 15,
        "autocommit": False,
        # times are stored in UTC, like on the other databases
        "init_command": "SET time_zone = '+00:00'",
    }
    ssl_mode = (query.get("ssl") or query.get("ssl-mode") or query.get("sslmode") or "").lower()
    if ssl_mode in ("1", "true", "required", "require", "verify"):
        import ssl
        context = ssl.create_default_context()
        if ssl_mode != "verify":        # encrypt, but accept the server's own certificate
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        settings["ssl"] = context
    return settings


def _mysql_connection():
    import pymysql
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.ping(reconnect=True)
            return conn
        except Exception:
            conn = None
    conn = pymysql.connect(**_mysql_settings())
    _local.conn = conn
    return conn


# -------------------------------------------------------- SQL translation

def _safe_alter(sql):
    """Several servers may start at once: adding a column twice must not fail."""
    return re.sub(r"\bADD COLUMN\s+(?!IF NOT EXISTS)", "ADD COLUMN IF NOT EXISTS ", sql, flags=re.I)


def _translate_pg(sql):
    sql = _safe_alter(sql)
    sql = sql.replace("%", "%%").replace("?", "%s")
    sql = re.sub(r"\bLIKE\b", "ILIKE", sql)                       # case-insensitive search
    sql = re.sub(r"\bINTEGER PRIMARY KEY AUTOINCREMENT\b", "SERIAL PRIMARY KEY", sql)
    sql = re.sub(r"\bBLOB\b", "BYTEA", sql)
    return sql


def _translate_mysql(sql):
    sql = _safe_alter(sql)
    sql = sql.replace("%", "%%").replace("?", "%s")
    sql = re.sub(r"\bINTEGER PRIMARY KEY AUTOINCREMENT\b", "INT AUTO_INCREMENT PRIMARY KEY", sql)
    sql = re.sub(r"\b(\w+)\s+TEXT PRIMARY KEY\b", r"\1 VARCHAR(255) PRIMARY KEY", sql)
    sql = re.sub(r"\bBLOB\b", "LONGBLOB", sql)
    sql = re.sub(r"\bTIMESTAMP\b", "DATETIME", sql)               # no automatic updates
    # upsert: ON CONFLICT(x) DO UPDATE SET a = excluded.a  ->  ON DUPLICATE KEY UPDATE a = VALUES(a)
    sql = re.sub(r"ON CONFLICT\s*\([^)]*\)\s*DO UPDATE SET", "ON DUPLICATE KEY UPDATE", sql, flags=re.I)
    sql = re.sub(r"\bexcluded\.(\w+)", r"VALUES(\1)", sql)
    # "key" and "value" are reserved words in MariaDB (settings table)
    if re.search(r"\bsettings\b", sql):
        sql = re.sub(r"(?<![`\w])(key|value)(?![`\w])", r"`\1`", sql)
    # new tables: proper Unicode, case-insensitive text
    if re.match(r"\s*CREATE TABLE", sql, re.I):
        sql = sql.rstrip().rstrip(";") + " DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"
    return sql


# --------------------------------------------------------------- wrapper

class Cursor:
    def __init__(self, cursor, lastrowid=None, names=None):
        self._cursor = cursor
        self.lastrowid = lastrowid
        self._names = names

    def _wrap(self, row):
        if row is None or self._names is None:
            return row
        return Row(row, self._names)

    def fetchone(self):
        try:
            return self._wrap(self._cursor.fetchone())
        except Exception:
            return None

    def fetchall(self):
        try:
            return [self._wrap(r) for r in self._cursor.fetchall()]
        except Exception:
            return []

    def __iter__(self):
        return iter(self.fetchall())


class Connection:
    """Small wrapper with the parts of the sqlite3 API the app uses."""

    # sqlite3 code sets this; rows are always dict-like here
    row_factory = None

    def __init__(self):
        if KIND == "postgres":
            self._conn = _pg_connection()
        elif KIND == "mysql":
            self._conn = _mysql_connection()
        else:
            self._conn = sqlite3.connect(DB_PATH)
            self._conn.row_factory = sqlite3.Row

    def execute(self, sql, params=()):
        if KIND == "sqlite":
            cursor = self._conn.execute(sql, params)
            return Cursor(cursor, cursor.lastrowid)

        if KIND == "mysql":
            cursor = self._conn.cursor()
            cursor.execute(_translate_mysql(sql), tuple(params))
            names = [d[0] for d in cursor.description] if cursor.description else None
            return Cursor(cursor, cursor.lastrowid, names)

        sql = _translate_pg(sql)
        returning = False
        if re.match(r"\s*INSERT\s+INTO\s+vendors\b", sql, re.I) and "RETURNING" not in sql.upper():
            sql = sql.rstrip().rstrip(";") + " RETURNING id"
            returning = True
        cursor = self._conn.cursor()
        cursor.execute(sql, params)
        lastrowid = cursor.fetchone()[0] if returning else None
        return Cursor(cursor, lastrowid)

    def columns(self, table):
        """Column names of a table."""
        if KIND == "postgres":
            rows = self.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = ?", (table,))
            return {r[0] for r in rows}
        if KIND == "mysql":
            rows = self.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = DATABASE() AND table_name = ?", (table,))
            return {r[0] for r in rows}
        return {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}

    def commit(self):
        self._conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            self._conn.rollback()
        else:
            self._conn.commit()
        if KIND == "sqlite":
            self._conn.close()
        return False


def connect():
    return Connection()


def backup_bytes():
    """A copy of the database for "Download backup" (SQLite only)."""
    if KIND != "sqlite":
        return None
    with sqlite3.connect(DB_PATH) as conn:
        return conn.serialize()


configure()
