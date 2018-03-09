import importlib
import os,sys
import os
from os import makedirs
from shutil import rmtree

def load_config(path='.'):
    path=os.path.abspath(path)
    if os.path.isdir(path):
        dir_path=path
        config_name="config"
    else:
        dir_path=os.path.dirname(path)
        config_name=os.path.basename(path).split('.')[0]

    sys.path.insert(0, dir_path)
    mo =importlib.import_module(config_name) 
    print "load config from {}/{}".format(dir_path, config_name)
    return mo.config

def check_dir(dir_path, ask_for_del=False):
    if os.path.exists(dir_path):
        y=''
        if ask_for_del:
            y=raw_input('new empty {}? y/n:'.format(dir_path))
        if y.strip()=='y' or not ask_for_del:
            rmtree(dir_path)
        else:
            print('use a clean summary_dir')
            quit()
    makedirs(dir_path)
