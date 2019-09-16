import numpy as np
from reader import OrebaReader
import csv
import argparse
import os
import copy
import itertools
import logging

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S', level=logging.INFO)

def main(args=None):
    """check if the hand mentioned the most in label3 (hand) matches with the participant's dominant hand"""
    def check_list_empty(list, list_name):
        if len(list) != 0:
            raise RuntimeError('list {1} is not emptry yet: {0}'.format(list, list_name))

    all_list = ['1001','1002','1003','1004','1005','1006','1007','1008','1010','1011','1012','1013','1014','1015','1016','1017','1018','1019','1020','1021','1022','1023','1024','1025','1026','1027','1028','1029','1030','1031','1032','1033','1035','1036','1037','1039','1040','1041','1043','1044','1045','1046','1047','1048','1050','1051','1052','1053','1054','1055','1056','1057','1059','1060','1061','1063','1064','1067','1068','1071','1072','1073','1074','1075','1076','1077','1079','1080','1081','1082','1083','1084','1085','1086','1087','1088','1089','1090','1091','1092','1093','1094','1095','1096','1097','1098','1099','1100','1101','1102','1103','1104','1105','1107','1108','1109','1110','1111','1112','1113','1115','1116']
    right_list = ['1001','1003','1004','1005','1006','1007','1011','1012','1013','1014','1015','1016','1017','1018','1019','1020','1022','1023','1024','1025','1026','1027','1028','1029','1030','1031','1032','1033','1035','1036','1040','1043','1044','1045','1046','1047','1050','1051','1052','1053','1054','1055','1056','1057','1059','1060','1061','1063','1064','1067','1068','1071','1072','1073','1074','1075','1076','1077','1079','1080','1081','1082','1083','1084','1086','1087','1088','1089','1090','1091','1092','1093','1094','1095','1096','1097','1098','1099','1100','1102','1103','1104','1105','1107','1108','1109','1111','1113','1115','1116']
    left_list = ['1002','1008','1010','1021','1037','1039','1041','1048','1085','1101','1110','1112']

    subject_ids = [x for x in next(os.walk(args.src_dir))[1]]
    reader = OrebaReader()
    exp_file = open(args.result_file_name,"w")

    header = "id,dominant hand,most used eat,most used drink,most used all,\
    eat right,eat left,eat both,\
    drink right,drink left,drink both,\
    all right,all left,all both,\
    eat result,drink result,all result"
    exp_file.write(header)
    exp_file.write('\n')

    for subject_id in subject_ids:
        all_list.remove(subject_id)
        if subject_id == '1074':
            right_list.remove(subject_id)
            continue
        logging.info("Working on subject {}".format(subject_id))
        # Read acc and gyro
        timestamps, left_acc, left_gyro, right_acc, right_gyro = \
            reader.read_inert(args.src_dir, subject_id)
        # Make hands uniform by flipping left to right if needed
        dominant_hand = reader.read_dominant(args.src_dir, subject_id)
        if dominant_hand.lower() == 'right':
            right_list.remove(subject_id)
        elif dominant_hand.lower() == 'left':
            left_list.remove(subject_id)
        else:
            raise RuntimeError('{0} not in the list of right and left'.format(dominant_hand))
 
        # Read annotations
        annotations = reader.read_annotations(args.src_dir, subject_id)
        label_1, label_2, label_3, label_4 = reader.get_labels(annotations, timestamps)

        label_3_right_all=0
        label_3_left_all=0
        label_3_idle_all=0
        label_3_both_all=0

        label_3_right_eat=0
        label_3_left_eat=0
        label_3_idle_eat=0
        label_3_both_eat=0

        label_3_right_drink=0
        label_3_left_drink=0
        label_3_idle_drink=0
        label_3_both_drink=0
            
        for i in range(len(label_3)):
            if label_3[i].lower() == "right":
                label_3_right_all += 1
            elif label_3[i].lower() == "left":
                label_3_left_all += 1
            elif label_3[i].lower() == "idle":
                label_3_idle_all += 1
            elif label_3[i].lower() == "both":
                label_3_both_all += 1
            else:
                print(label_3[i])

        for i in range(len(label_3)):
            if label_2[i].lower() == "eat":
                if label_3[i].lower() == "right":
                    label_3_right_eat += 1
                elif label_3[i].lower() == "left":
                    label_3_left_eat += 1
                elif label_3[i].lower() == "idle":
                    label_3_idle_eat += 1
                elif label_3[i].lower() == "both":
                    label_3_both_eat += 1
                else:
                    print(label_3[i])

            if label_2[i].lower() == "drink":
                if label_3[i].lower() == "right":
                    label_3_right_drink += 1
                elif label_3[i].lower() == "left":
                    label_3_left_drink += 1
                elif label_3[i].lower() == "idle":
                    label_3_idle_drink += 1
                elif label_3[i].lower() == "both":
                    label_3_both_drink += 1
                else:
                    print(label_3[i])

        if label_3_right_all > label_3_left_all:
            most_used_all = "right" 
        elif label_3_right_all < label_3_left_all:
            most_used_all = "left" 
        elif label_3_right_all == 0:
            most_used_all = "none" 
        else:
            most_used_all = "both" 

        if label_3_right_eat > label_3_left_eat:
            most_used_eat = "right" 
        elif label_3_right_eat < label_3_left_eat:
            most_used_eat = "left" 
        elif label_3_right_eat == 0:
            most_used_eat = "none" 
        else:
            most_used_eat = "both" 

        if label_3_right_drink > label_3_left_drink:
            most_used_drink = "right" 
        elif label_3_right_drink < label_3_left_drink:
            most_used_drink = "left" 
        elif label_3_right_drink == 0:
            most_used_drink = "none" 
        else:
            most_used_drink = "both" 

        result_eat = "OK" if dominant_hand == most_used_eat else "wrong"
        result_drink = "OK" if dominant_hand == most_used_drink else "wrong"
        result_all = "OK" if dominant_hand == most_used_all else "wrong"

        line = subject_id + ',' + dominant_hand + ',' + most_used_eat + ',' + most_used_drink + ',' + most_used_all + ',' + \
            str(label_3_right_eat) + ',' + str(label_3_left_eat) + ',' + str(label_3_both_eat) + ',' + \
            str(label_3_right_drink) + ',' + str(label_3_left_drink) + ',' + str(label_3_both_drink) + ',' + \
            str(label_3_right_all) + ',' + str(label_3_left_all) + ',' + str(label_3_both_all) + ',' + \
            result_eat + ',' + result_drink + ',' + result_all

        exp_file.writelines(line)
        exp_file.write('\n')

    check_list_empty(all_list,'all subjects')
    check_list_empty(right_list,'right-handed')
    check_list_empty(left_list,'left-handed')

    exp_file.close()
    reader.done()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='check if the hand mentioned the most in label3 (hand) matches with the participants dominant hand')
    parser.add_argument('--src_dir', type=str, default=r'\\uncle.newcastle.edu.au\entities\research\oreba\OREBA\Phase 1\Synchronised', nargs='?', help='Directory to search for data.')
    parser.add_argument('--result_file_name', type=str, default=r'C:\H\PhD\ORIBA\Model\Code\data\verify_dom_hand.csv', nargs='?', help='The resut file name.')
    args = parser.parse_args()
    main(args)
