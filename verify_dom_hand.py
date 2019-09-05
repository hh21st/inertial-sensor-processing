import numpy as np
from reader import OrebaReader
import csv
import argparse
import os
import copy
import itertools

def main(args=None):
    """check if the hand mentioned the most in label3 (hand) matches with the participant's dominant hand"""
    subject_ids = [x for x in next(os.walk(args.src_dir))[1]]
    reader = OrebaReader()
    exp_file = open(args.result_file_name,"w")
    for subject_id in subject_ids:
        if subject_id == "1074":
            continue
        # Read acc and gyro
        timestamps, left_acc, left_gyro, right_acc, right_gyro = \
            reader.read_inert(args.src_dir, subject_id)
        # Make hands uniform by flipping left to right if needed
        dominant_hand = reader.read_dominant(args.src_dir, subject_id)
        # Read annotations
        annotations = reader.read_annotations(args.src_dir, subject_id)
        label_1, label_2, label_3, label_4 = reader.get_labels(annotations, timestamps)
        label_3_right=0
        label_3_left=0
        label_3_idle=0
        label_3_both=0

        label_3_right_d=0
        label_3_left_d=0
        label_3_idle_d=0
        label_3_both_d=0
            
        for i in range(len(label_3)):
            if label_2[i] == "Eat":
                if label_3[i] == "Right":
                    label_3_right += 1
                elif label_3[i] == "Left":
                    label_3_left += 1
                elif label_3[i] == "Idle":
                    label_3_idle += 1
                elif label_3[i] == "Both":
                    label_3_both += 1
                else:
                    print(label_3[i])

            if label_2[i] == "Drink":
                if label_3[i] == "Right":
                    label_3_right_d += 1
                elif label_3[i] == "Left":
                    label_3_left_d += 1
                elif label_3[i] == "Idle":
                    label_3_idle_d += 1
                elif label_3[i] == "Both":
                    label_3_both_d += 1
                else:
                    print(label_3[i])

        if label_3_right > label_3_left:
            dom_count = "right" 
        elif label_3_right < label_3_left:
            dom_count = "left" 
        elif label_3_right == 0:
            dom_count = "none" 
        else:
            dom_count = "both" 

        if label_3_right_d > label_3_left_d:
            dom_count_d = "right" 
        elif label_3_right_d < label_3_left_d:
            dom_count_d = "left" 
        elif label_3_right_d == 0:
            dom_count_d = "none" 
        else:
            dom_count_d = "both" 

        result =   "<  OK!  >        " if dominant_hand == dom_count else "< WRONG >        "
        result_d = "        <Dr==DOM>" if dominant_hand == dom_count_d else "        <DR!!DOM>"

        report = "subject:{0}: {7}, dom_hand:{1}, lablel_3:{2}, right:{3}, left:{4}, both:{5}, idle:{6}\n".\
            format(subject_id,dominant_hand,dom_count,label_3_right,label_3_left,label_3_both,label_3_idle, result)
        print(report)
        exp_file.writelines(report)
        report_d = "subject:{0}: {7}, dom_hand:{1}, lablel_3:{2}, right:{3}, left:{4}, both:{5}, idle:{6}\n".\
            format(subject_id,dominant_hand,dom_count_d,label_3_right_d,label_3_left_d,label_3_both_d,label_3_idle_d, result_d)
        print(report_d)        
        exp_file.writelines(report_d)
    exp_file.close()
    reader.done()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process inertial sensor data')
    parser.add_argument('--src_dir', type=str, default='', nargs='?', help='Directory to search for data.')
    parser.add_argument('--result_file_name', type=str, default='verify_dom_hand', nargs='?', help='The resut file name.')
    args = parser.parse_args()
    main(args)
