import os
import subprocess
import sys

my_device = int(sys.argv[1])
if len(sys.argv)>2:
    prefix = sys.argv[2]
else: 
    prefix = ''

for conf in os.scandir('final_setup/ooo_conf'):
    if conf.name.startswith(prefix):
        model_base_name = '.'.join(conf.name.split('.')[:-1])
        log_path = os.path.join('final_setup/ooo_logs', model_base_name+'.log')
        dset_path = os.path.join('final_setup/ooo_models', model_base_name+'.dset')
        if not os.path.exists(dset_path):
            seed_num = model_base_name.split('.')[-1].split('_')[0]
            if (int(seed_num)<55):
                print (conf.path)
                print ('device:', my_device)
                print('seed:', seed_num)
                result = subprocess.run(['python', 'main.'+seed_num+'.py', 
                                         '--config', conf.path, '--device', str(my_device)], 
                                        capture_output=True)
                with open(log_path, 'wb') as of:
                    of.write(result.stdout)
                if len(result.stderr)>0:
                    print(result.stderr)