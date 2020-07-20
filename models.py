  
# models.py
import datetime
from peewee import *

db = SqliteDatabase("/data/database.db", pragmas=[('journal_mode', 'wal')])

class ClassifyEntity(Model):
    smiles = TextField(unique=True, index=True)
    classification_json = TextField()

    class Meta:
        database = db

#Creating the Tables
db.create_tables([ClassifyEntity], safe=True)
