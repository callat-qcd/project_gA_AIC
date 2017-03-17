# store analysis data to sqlite3 db
import sqlite3

def login(p):
    conn = sqlite3.connect(p['dbname'])
    c = conn.cursor()
    return c,conn

def id_name_nbs_result_table(c,conn,p):
    cmd = "CREATE TABLE IF NOT EXISTS %s (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, nbs INTEGER NOT NULL, result JSON NOT NULL, CONSTRAINT unique_%s UNIQUE (name,nbs));" %(p['tblname'],p['tblname'])
    c.execute(cmd)
    return c,conn

def id_name_nbs_result_insert(c,conn,p,name,nbs,result):
    cmd = "INSERT OR REPLACE INTO %s (name, nbs, result) VALUES ('%s', %s, '%s');" %(p['tblname'],name,nbs,result)
    c.execute(cmd)
    conn.commit()
    return c,conn

