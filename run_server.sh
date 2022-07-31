#!/bin/bash

source activate rdkit

#python ./app.py
gunicorn -w 8 --threads=6 --worker-class=gthread -b 0.0.0.0:5000 --timeout 20 app:server --access-logfile /app/logs/access.log --max-requests 1000
