import os
import subprocess
import sys

my_device = int(sys.argv[1])
if len(sys.argv)>2:
    prefix = sys.argv[2]
else: 
    prefix = ''

for conf in os.scandir('hp_search/conf2/'):
    if conf.name.startswith(prefix):
        log_path = os.path.join('hp_search/logs2', '.'.join(conf.name.split('.')[:-1])+'.log')
        if not os.path.exists(log_path):
            num = int(conf.name.split('.')[1])
            device = num%4
            if device == my_device:
                print (conf.path)
                print ('device:', device)
                result = subprocess.run(['python', 'main.py', '--config', conf.path, '--device', str(device)], capture_output=True)
                with open(log_path, 'wb') as of:
                    of.write(result.stdout)