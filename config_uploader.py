import os
import time

address =           [
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
        "3.135.224.36"
    ]

for curr in address:
    os.system(f"scp -i \"reboot-fixer.pem\" collab.config ubuntu@{curr}:/home/ubuntu/collab.config")
    time.sleep(20)