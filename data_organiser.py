import csv
import argparse
import os
import shutil
from absl import logging
from utils import *

#logging.set_verbosity('info')

#ALL_LIST = ['1001','1002','1003','1004','1005','1006','1007','1008','1010',
#    '1011','1012','1013','1014','1015','1016','1017','1018','1019','1020',
#    '1021','1022','1023','1024','1025','1026','1027','1028','1029','1030',
#    '1031','1032','1033','1035','1036','1037','1039','1040','1041','1043',
#    '1044','1045','1046','1047','1048','1050','1051','1052','1053','1054',
#    '1055','1056','1057','1059','1060','1061','1063','1064','1067','1068',
#    '1071','1072','1073','1075','1076','1077','1079','1080','1081','1082',
#    '1083','1084','1085','1086','1087','1088','1089','1090','1091','1092',
#    '1093','1094','1095','1096','1097','1098','1099','1100','1101','1102',
#    '1103','1104','1105','1107','1108','1109','1110','1111','1112','1113',
#    '1115','1116']
#TRAIN_LIST = ['1001','1002','1003','1006','1007','1008','1012','1013','1014',
#    '1017','1018','1019','1022','1023','1024','1027','1028','1029','1032',
#    '1033','1035','1039','1040','1041','1045','1046','1047','1051','1052',
#    '1053','1056','1057','1059','1063','1064','1067','1072','1073','1077',
#    '1079','1080','1083','1084','1085','1088','1089','1090','1093','1094',
#    '1095','1098','1099','1100','1103','1104','1105','1109','1110','1111',
#    '1115','1116']
#VAL_LIST = ['1004','1010','1015','1020','1025','1030','1036','1043','1048',
#    '1054','1060','1068','1075','1081','1086','1091','1096','1101','1107',
#    '1112']
#TEST_LIST = ['1005','1011','1016','1021','1026','1031','1037','1044','1050',
#    '1055','1061','1071','1076','1082','1087','1092','1097','1102','1108',
#    '1113']

class DataOrganiser:
    def get_splits(self, split_filenamepath):
        ALL_LIST = []
        TRAIN_LIST = []
        VAL_LIST = []
        TEST_LIST = []
        split_file_info = csv.reader(open(split_filenamepath, 'r'), delimiter=',')
        next(split_file_info, None)
        for row in split_file_info:
            if row[1].strip().upper() == 'TRAIN':
                TRAIN_LIST.append(row[0])
            elif row[1].strip().upper() == 'VALID':
                VAL_LIST.append(row[0])
            elif row[1].strip().upper() == 'TEST':
                TEST_LIST.append(row[0])
            else:
                raise RuntimeError("split {0} is not recognised".format(row[1]));
            ALL_LIST.append(row[0])
        return ALL_LIST,TRAIN_LIST,VAL_LIST,TEST_LIST    

    def organise(self, src_dir, des_dir, make_subfolders_val, make_subfolders_test, split_filenamepath):
        """divides data into training, validation and test sets"""



        def get_subject_id(file_name):
            #return file_name.split('_')[1]
            return file_name.split('_')[1] + '_' + file_name.split('_')[2]

        def copy_file_and_log(full_file_name, copy_to_dir):
            if os.path.isfile(os.path.join(copy_to_dir, file_name)):
                logging.info("file {0} already exists".format(os.path.join(copy_to_dir, file_name)))
            else:
                copy_file(full_file_name, copy_to_dir)

        ALL_LIST,TRAIN_LIST,VAL_LIST,TEST_LIST = self.get_splits(split_filenamepath)

        train_dir = os.path.join(des_dir, 'train')
        val_dir = os.path.join(des_dir, 'eval')
        val_sub_dir = os.path.join(des_dir, 'eval_sub')
        test_dir = os.path.join(des_dir, 'test')
        test_sub_dir = os.path.join(des_dir, 'test_sub')

        src_files = os.listdir(src_dir)
        for file_name in src_files:
            subject_id = get_subject_id(file_name)
            full_file_name = os.path.join(src_dir, file_name)
            logging.info("processing subject {0}, copying file {1}".format(subject_id, full_file_name))
            if subject_id in TRAIN_LIST:
                copy_to_dir = train_dir
                copy_to_sub_dir = ''
            elif subject_id in VAL_LIST:
                copy_to_dir = val_dir
                copy_to_sub_dir = os.path.join(val_sub_dir, subject_id) if make_subfolders_val else ''
            elif subject_id in TEST_LIST:
                copy_to_dir = test_dir
                copy_to_sub_dir = os.path.join(test_sub_dir, subject_id) if make_subfolders_test else ''
            else:
                raise RuntimeError("subject {0} is not in any of the lists".format(subject_id, full_file_name))

            copy_file_and_log(full_file_name, copy_to_dir)
            if copy_to_sub_dir != '':
                copy_file_and_log(full_file_name, copy_to_sub_dir)

def main(args=None):
    DataOrganiser.organise(src_dir=args.src_dir, des_dir=args.des_dir,
        make_subfolders_val=get_bool(args.make_subfolders_val),
        make_subfolders_test=get_bool(args.make_subfolders_test))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='devides data into training, validation and test sets')
    parser.add_argument('--src_dir', type=str, default='', nargs='?', help='Directory to search for data.')
    parser.add_argument('--des_dir', type=str, default='', nargs='?', help='Directory to copy train, val and test sets.')
    parser.add_argument('--make_subfolders_val', type=str, default='False' , nargs='?', help='Create sub forlder per each file in validation set if true.')
    parser.add_argument('--make_subfolders_test', type=str, default='False' , nargs='?', help='Create sub forlder per each file in test set if true.')
    parser.add_argument('--split_filenamepath', type=str, default='' , nargs='?', help='Full name and path of the file that contains the splits')
    args = parser.parse_args()
    main(args)
