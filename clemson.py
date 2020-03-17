"""Clemson dataset"""

import csv
import os
import datetime as dt
import logging
import glob
import xlrd
import xml.etree.cElementTree as etree
import tensorflow as tf
import numpy as np

FREQUENCY = 15
ACC_SENSITIVITY = 660.0
GYRO_SENSITIVITY = 2.5
DEFAULT_LABEL = "Idle"
FLIP_ACC = [-1., 1., 1.]
FLIP_GYRO = [1., -1., -1.]
TIME_FACTOR = 1000000

TRAIN_IDS = []
VALID_IDS = []
TEST_IDS = []

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class Dataset():

    def __init__(self, src_dir, exp_dir, dom_hand_spec, label_spec,
        label_spec_inherit, exp_uniform, exp_format):
        self.src_dir = src_dir
        self.exp_dir = exp_dir
        self.dom_hand_spec = dom_hand_spec
        self.label_spec = label_spec
        self.label_spec_inherit = label_spec_inherit
        self.exp_uniform = exp_uniform
        self.exp_format = exp_format
        # Class names
        self.names_1, self.names_2, self.names_3, self.names_4 = \
            self.__class_names()

    def __class_names(self):
        """Get class names from label master file"""
        label_spec_path = os.path.join(self.src_dir, self.label_spec)
        assert os.path.isfile(label_spec_path), "Couldn't find label master file"
        names_1 = []; names_2 = []; names_3 = []; names_4 = []
        tree = etree.parse(label_spec_path)
        categories = tree.getroot()
        for tag in categories[0]:
            names_1.append(tag.attrib['name'])
        for tag in categories[1]:
            names_2.append(tag.attrib['name'])
        for tag in categories[2]:
            names_3.append(tag.attrib['name'])
        for tag in categories[3]:
            names_4.append(tag.attrib['name'])
        return names_1, names_2, names_3, names_4

    def ids(self):
        data_dir = os.path.join(self.src_dir, "all-data")
        subject_ids = [x for x in next(os.walk(data_dir))[1]]
        ids = []
        for subject_id in subject_ids:
            subject_dir = os.path.join(data_dir, subject_id)
            session_ids = [x for x in next(os.walk(subject_dir))[1]]
            for session_id in session_ids:
                ids.append((subject_id, session_id))
        return ids

    def check(self, id):
        # Path of gesture annotations
        gesture_dir = os.path.join(self.src_dir, "all-gt-gestures", id[0],
            id[1], "gesture_union.txt")
        if not os.path.isfile(gesture_dir):
            logging.warn("No gesture annotations found. Skipping {}_{}.".format(
                id[0], id[1]))
            return False
        # Path of bite annotations
        bite_dir = os.path.join(self.src_dir, "all-gt-bites", id[0],
            id[1], "gt_union.txt")
        if not os.path.isfile(bite_dir):
            logging.warn("No bite annotations found. Skipping {}_{}.".format(
                id[0], id[1]))
            return False
        return True

    def data(self, _, id):
        logging.info("Reading raw data from txt")
        # Read acc and gyro
        dir = os.path.join(self.src_dir, "all-data", id[0], id[1])
        files = glob.glob(os.path.join(dir, "*.txt"))
        assert files, "No raw data found for {} {}".format(id[0], id[1])
        acc = []
        gyro = []
        with open(files[0]) as dest_f:
            for row in csv.reader(dest_f, delimiter='\t'):
                acc_x = (float(row[0]) - 1.65) * 1000.0 / ACC_SENSITIVITY
                acc_y = (float(row[1]) - 1.65) * 1000.0 / ACC_SENSITIVITY
                acc_z = (float(row[2]) - 1.65) * 1000.0 / ACC_SENSITIVITY
                acc.append([acc_x, acc_y, acc_z])
                # Todo estimate zero rate output mean instead of simply using 1.25!
                gyro_x = (float(row[3])-1.25) * 1000.0 / GYRO_SENSITIVITY
                gyro_y = (float(row[4])-1.25) * 1000.0 / GYRO_SENSITIVITY
                gyro_z = (float(row[5])-1.25) * 1000.0 / GYRO_SENSITIVITY
                gyro.append([gyro_x, gyro_y, gyro_z])
        dt = TIME_FACTOR // FREQUENCY # In microseconds
        timestamps = range(0, len(acc)*dt, dt)
        return timestamps, {"hand": (acc, gyro)}

    def dominant(self, id):
        """Read handedness, which is the hand sensor was placed on"""
        file_path = os.path.join(self.src_dir, "demographics.xlsx")
        workbook = xlrd.open_workbook(file_path)
        sheet = workbook.sheet_by_index(0)
        for rowx in range(sheet.nrows):
            cols = sheet.row_values(rowx)
            if cols[0].lower() == id[0]:
                return cols[4].lower()
        return None

    def labels(self, _, id, timestamps):
        def _index_to_ms(index):
            dt = TIME_FACTOR // FREQUENCY
            return index * dt
        # Read gesture ground truth
        gesture_dir = os.path.join(self.src_dir, "all-gt-gestures", id[0],
            id[1], "gesture_union.txt")
        label_1, start_time, end_time = [], [], []
        with open(gesture_dir) as dest_f:
            for row in csv.reader(dest_f, delimiter='\t'):
                if row[0].lower() in self.names_1:
                    label_1.append(row[0].lower())
                    start_time.append(_index_to_ms(int(row[1])))
                    end_time.append(_index_to_ms(int(row[2])))
        # Read bite ground truth by matching with gestures
        bite_dir = os.path.join(self.src_dir, "all-gt-bites", id[0],
            id[1], "gt_union.txt")
        num = len(timestamps)
        labels_1 = np.empty(num, dtype='U25'); labels_1.fill(DEFAULT_LABEL)
        labels_2 = np.empty(num, dtype='U25'); labels_2.fill(DEFAULT_LABEL)
        labels_3 = np.empty(num, dtype='U25'); labels_3.fill(DEFAULT_LABEL)
        labels_4 = np.empty(num, dtype='U25'); labels_4.fill(DEFAULT_LABEL)
        for l1, start, end in zip(label_1, start_time, end_time):
            start_frame = np.argmax(np.array(timestamps) >= start)
            end_frame = np.argmax(np.array(timestamps) > end)
            match_found = False
            with open(bite_dir) as dest_f:
                for row in csv.reader(dest_f, delimiter='\t'):
                    time = _index_to_ms(int(row[1]))
                    if time >= start and time <= end:
                        if row[2].lower() in self.names_2:
                            l2 = row[2].lower()
                        if row[3].lower() in self.names_3:
                            l3 = row[3].lower()
                        if row[4].lower() in self.names_4:
                            l4 = row[4].lower()
                        match_found = True
                        break
            if not match_found:
                l2 = "NA"; l3 = "NA"; l4 = "NA"
            labels_1[start_frame:end_frame] = l1
            if l2 in self.names_2:
                labels_2[start_frame:end_frame] = l2
            if l3 in self.names_3:
                labels_3[start_frame:end_frame] = l3
            if l4 in self.names_4:
                labels_4[start_frame:end_frame] = l4

        return (labels_1, labels_2, labels_3, labels_4)

    def write(self, path, id, timestamps, data, dominant_hand, labels):
        frame_ids = range(0, len(timestamps))
        id = '_'.join(id)
        def _format_time(t):
            return (dt.datetime.min + dt.timedelta(microseconds=t)).time().strftime('%H:%M:%S.%f')
        timestamps = [_format_time(t) for t in timestamps]
        acc = data["hand"][0]
        gyro = data["hand"][1]
        if self.exp_format == 'csv':
            with open(path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(["id", "frame_id", "timestamp", "acc_x", "acc_y",
                    "acc_z", "gyro_x", "gyro_y", "gyro_z", "hand",
                    "label_1", "label_2", "label_3", "label_4"])
                for i in range(0, len(timestamps)):
                    writer.writerow([id, frame_ids[i], timestamps[i],
                        acc[i][0], acc[i][1], acc[i][2], gyro[i][0], gyro[i][1],
                        gyro[i][2], dominant_hand, labels[0][i], labels[1][i],
                        labels[2][i], labels[3][i]])
        elif self.exp_format == 'tfrecord':
            with tf.io.TFRecordWriter(path) as tfrecord_writer:
                for i in range(0, len(timestamps)):
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'example/subject_id': _bytes_feature(id.encode()),
                        'example/frame_id': _int64_feature(frame_ids[i]),
                        'example/timestamp': _bytes_feature(timestamps[i].encode()),
                        'example/acc': _floats_feature(acc[i].ravel()),
                        'example/gyro': _floats_feature(gyro[i].ravel()),
                        'example/label_1': _bytes_feature(labels[0][i].encode()),
                        'example/label_2': _bytes_feature(labels[1][i].encode()),
                        'example/label_3': _bytes_feature(labels[2][i].encode()),
                        'example/label_4': _bytes_feature(labels[3][i].encode())
                    }))
                    tfrecord_writer.write(example.SerializeToString())

    def done(self):
        logging.info("Done")

    def get_flip_signs(self):
        return FLIP_ACC, FLIP_GYRO

    def get_frequency(self):
        return FREQUENCY

    def get_time_factor(self):
        return TIME_FACTOR

    def get_train_ids(self):
        return TRAIN_IDS

    def get_valid_ids(self):
        return VALID_IDS

    def get_test_ids(self):
        return TEST_IDS
