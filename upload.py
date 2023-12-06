import os
import time
import threading

address =                 [
        "3.136.85.89",
        "3.135.17.227",
        "18.191.32.127",
        "18.118.34.89",
        "3.133.96.214",
        "3.20.222.173",
        "18.222.128.237",
        "18.217.207.59",
        "18.224.153.161",
        "52.14.95.221",
        "3.135.9.78",
        "3.16.57.106",
        "18.217.35.59",
        "3.141.26.102",
        "3.135.224.36", 
        "18.188.196.181"
    ]

all_threads = []
for curr in address:
    os.system(f"rsync -Pav -e \"ssh -i reboot-fixer.pem\" ../dorylus ubuntu@{curr}:~/")
    time.sleep(30)