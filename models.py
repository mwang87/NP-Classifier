  
# models.py
import datetime
from peewee import *
from app import db

class ClassifyEntity(Model):
    smiles = TextField(unique=True, index=True)
    classification_json = TextField()

    class Meta:
        database = db

#Creating the Tables
db.create_tables([ClassifyEntity], safe=True)
