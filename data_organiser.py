import argparse
import os
import shutil
import logging
import tensorflow as tf
from utils import *
from oreba_utils import *

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S', level=logging.INFO)

class DataOrganiser:
    def organise(params):
        """devides data into training, validation and test sets"""

        def copy_file_and_log(full_file_name, copy_to_dir):
            if os.path.isfile(os.path.join(copy_to_dir, file_name)):
                logging.info("file {0} already exists".format(os.path.join(copy_to_dir, file_name)))
            else:
                copy_file(full_file_name, copy_to_dir)

        all_list = ['1001','1002','1003','1004','1005','1006','1007','1008','1010','1011','1012','1013','1014','1015','1016','1017','1018','1019','1020','1021','1022','1023','1024','1025','1026','1027','1028','1029','1030','1031','1032','1033','1035','1036','1037','1039','1040','1041','1043','1044','1045','1046','1047','1048','1050','1051','1052','1053','1054','1055','1056','1057','1059','1060','1061','1063','1064','1067','1068','1071','1072','1073','1075','1076','1077','1079','1080','1081','1082','1083','1084','1085','1086','1087','1088','1089','1090','1091','1092','1093','1094','1095','1096','1097','1098','1099','1100','1101','1102','1103','1104','1105','1107','1108','1109','1110','1111','1112','1113','1115','1116']
        train_list = ['1001','1002','1003','1006','1007','1008','1012','1013','1014','1017','1018','1019','1022','1023','1024','1027','1028','1029','1032','1033','1035','1039','1040','1041','1045','1046','1047','1051','1052','1053','1056','1057','1059','1063','1064','1067','1072','1073','1077','1079','1080','1083','1084','1085','1088','1089','1090','1093','1094','1095','1098','1099','1100','1103','1104','1105','1109','1110','1111','1115','1116']
        val_list = ['1004','1010','1015','1020','1025','1030','1036','1043','1048','1054','1060','1068','1075','1081','1086','1091','1096','1101','1107','1112']
        test_list = ['1005','1011','1016','1021','1026','1031','1037','1044','1050','1055','1061','1071','1076','1082','1087','1092','1097','1102','1108','1113']

        train_dir = os.path.join(params.des_dir,'train') 
        val_dir = os.path.join(params.des_dir,'eval') 
        val_sub_dir = os.path.join(params.des_dir,'eval_sub') 
        test_dir = os.path.join(params.des_dir,'test')  
        test_sub_dir = os.path.join(params.des_dir,'test_sub')  
    
        src_files = os.listdir(params.src_dir)
        for file_name in src_files:
            subject_id = get_subject_id(file_name)
            full_file_name = os.path.join(params.src_dir, file_name)
            logging.info("processing subject {0}, copying file {1}".format(subject_id, full_file_name))
            if subject_id in train_list:
                copy_to_dir = train_dir
                copy_to_sub_dir = ''
            elif subject_id in val_list:
                copy_to_dir = val_dir
                copy_to_sub_dir = os.path.join(val_sub_dir, subject_id) if params.make_subfolders_val else ''
            elif subject_id in test_list:
                copy_to_dir = test_dir
                copy_to_sub_dir = os.path.join(test_sub_dir, subject_id) if params.make_subfolders_test else ''
            else:
                raise RuntimeError("subject {0} is not in any of the lists")

            copy_file_and_log(full_file_name, copy_to_dir)
            if copy_to_sub_dir != '':
                copy_file_and_log(full_file_name, copy_to_sub_dir)


def main(args=None):
    params = tf.contrib.training.HParams(
        src_dir = args.src_dir,
        des_dir = args.des_dir,
        make_subfolders_val = get_bool(args.make_subfolders_val),
        make_subfolders_test = get_bool(args.make_subfolders_test))

    DataOrganiser.organise(params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='devides data into training, validation and test sets')
    parser.add_argument('--src_dir', type=str, default='', nargs='?', help='Directory to search for data.')
    parser.add_argument('--des_dir', type=str, default='', nargs='?', help='Directory to copy train, val and test sets.')
    parser.add_argument('--make_subfolders_val', type=str, default='False' , nargs='?', help='Create sub forlder per each file in validation set if true.')
    parser.add_argument('--make_subfolders_test', type=str, default='False' , nargs='?', help='Create sub forlder per each file in test set if true.')
    args = parser.parse_args()
    main(args)
