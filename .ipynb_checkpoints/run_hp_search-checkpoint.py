import os
import subprocess
import sys

my_device = int(sys.argv[1])
prefix = sys.argv[2]

for conf in os.scandir('hp_search/conf/'):
    if conf.name.startswith(prefix):
        num = int(conf.name.split('.')[1])
        device = num%3 + 1
        if device == my_device:
            print (conf.path)
            print ('device:', device)
            result = subprocess.run(['python', 'main.py', '--config', conf.path, '--device', str(device)], capture_output=True)
            with open(os.path.join('hp_search/logs', '.'.join(conf.name.split('.')[:-1])+'.log'), 'wb') as of:
                of.write(result.stdout)