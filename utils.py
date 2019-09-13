import os
import shutil
import logging

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S', level=logging.INFO)

def get_bool(boolean_string):
    if boolean_string not in {'False', 'True'}:
        raise ValueError('{0} is not a valid boolean string'.format(boolean_string))
    return boolean_string == 'True'

def create_dir_if_required(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def copy_file(full_file_name, copy_to_dir):
    create_dir_if_required(copy_to_dir)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, copy_to_dir)    
    else:
        logging.error("file {0} does not exist.".format(full_file_name))


