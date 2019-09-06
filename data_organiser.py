import numpy as np
import csv
import argparse
import copy
import itertools

import os
import shutil
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S', level=logging.INFO)

def create_dir_if_required(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def copy_file(full_file_name, train_dir):
    create_dir_if_required(train_dir)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, train_dir)    
    else:
        logging.error("file {0} does not exist.".format(full_file_name))


def main(args=None):
    """devides data into training, evaluation and test sets"""

    def check_list_empty(list):
        if len(list) != 0:
            logging.error("list is not emptry yet: {0}".format(list))

    def get_subject_id(file_name):
        return file_name.split('_')[1]

    all_list = ['1001','1002','1003','1004','1005','1006','1007','1008','1010','1011','1012','1013','1014','1015','1016','1017','1018','1019','1020','1021','1022','1023','1024','1025','1026','1027','1028','1029','1030','1031','1032','1033','1035','1036','1037','1039','1040','1041','1043','1044','1045','1046','1047','1048','1050','1051','1052','1053','1054','1055','1056','1057','1059','1060','1061','1063','1064','1067','1068','1071','1072','1073','1075','1076','1077','1079','1080','1081','1082','1083','1084','1085','1086','1087','1088','1089','1090','1091','1092','1093','1094','1095','1096','1097','1098','1099','1100','1101','1102','1103','1104','1105','1107','1108','1109','1110','1111','1112','1113','1115','1116']
    train_list = ['1001','1002','1003','1006','1007','1008','1012','1013','1014','1017','1018','1019','1022','1023','1024','1027','1028','1029','1032','1033','1035','1039','1040','1041','1045','1046','1047','1051','1052','1053','1056','1057','1059','1063','1064','1067','1072','1073','1077','1079','1080','1083','1084','1085','1088','1089','1090','1093','1094','1095','1098','1099','1100','1103','1104','1105','1109','1110','1111','1115','1116']
    eval_list = ['1004','1010','1015','1020','1025','1030','1036','1043','1048','1054','1060','1068','1075','1081','1086','1091','1096','1101','1107','1112']
    test_list = ['1005','1011','1016','1021','1026','1031','1037','1044','1050','1055','1061','1071','1076','1082','1087','1092','1097','1102','1108','1113']

    train_dir = os.path.join(args.des_dir,'train') 
    eval_dir = os.path.join(args.des_dir,'eval') 
    test_dir = os.path.join(args.des_dir,'test')  
    
    src_files = os.listdir(args.src_dir)
    for file_name in src_files:
        subject_id = get_subject_id(file_name)
        full_file_name = os.path.join(args.src_dir, file_name)
        logging.info("processing subject {0}, copying file {1}".format(subject_id, full_file_name))
        if subject_id in train_list:
            copy_file(full_file_name, train_dir)
            train_list.remove(subject_id)
            all_list.remove(subject_id)
        elif subject_id in eval_list:
            copy_file(full_file_name, eval_dir)
            eval_list.remove(subject_id)
            all_list.remove(subject_id)
        elif subject_id in test_list:
            copy_file(full_file_name, test_dir)
            test_list.remove(subject_id)
            all_list.remove(subject_id)
    
    check_list_empty(train_list)
    check_list_empty(eval_list)
    check_list_empty(test_list)
    check_list_empty(all_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='devides data into training, evaluation and test sets')
    parser.add_argument('--src_dir', type=str, default=r'C:\H\PhD\ORIBA\Model\FileGen\OREBA\smo_1', nargs='?', help='Directory to search for data.')
    parser.add_argument('--des_dir', type=str, default=r'Y:\input\OREBA\64_std_uni\smo_1', nargs='?', help='Directory to copy train, eval and test sets.')
    args = parser.parse_args()
    main(args)
