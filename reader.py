import csv
import os
import math
import jpype
import jpype.imports
import numpy as np
import datetime as dt
import xml.etree.cElementTree as etree
import glob
import h5py
import pickle
import xlrd

OREBA_FREQUENCY = 64
OREBA_DEFAULT_LABEL = "Idle"
CLEMSON_FREQUENCY = 15
CLEMSON_ACC_SENSITIVITY = 660.0
CLEMSON_GYRO_SENSITIVITY = 2.5
CLEMSON_DEFAULT_LABEL = "Idle"
FIC_DEFAULT_LABEL = "Idle"

class OrebaReader:
    def __init__(self):
        jpype.addClassPath('org.unisens.jar')
        jpype.addClassPath('org.unisens.ri.jar')
        jpype.startJVM()

    def read_inert(self, src_dir, subject_id):
        def _parse_j2py(jpype_x, jpype_y, jpype_z):
            """Convert from double[][] to list"""
            return list(zip([list(i)[0] for i in jpype_x],
                            [list(i)[0] for i in jpype_y],
                            [list(i)[0] for i in jpype_z]))
        dir = os.path.join(src_dir, subject_id, "data_sensor")
        from org.unisens import UnisensFactoryBuilder
        jUnisensFactory = UnisensFactoryBuilder.createFactory()
        jUnisens = jUnisensFactory.createUnisens(dir)
        jLeftAccXEntry = jUnisens.getEntry('left_accx.bin')
        jLeftAccYEntry = jUnisens.getEntry('left_accy.bin')
        jLeftAccZEntry = jUnisens.getEntry('left_accz.bin')
        jLeftGyroXEntry = jUnisens.getEntry('left_gyrox.bin')
        jLeftGyroYEntry = jUnisens.getEntry('left_gyroy.bin')
        jLeftGyroZEntry = jUnisens.getEntry('left_gyroz.bin')
        jRightAccXEntry = jUnisens.getEntry('right_accx.bin')
        jRightAccYEntry = jUnisens.getEntry('right_accy.bin')
        jRightAccZEntry = jUnisens.getEntry('right_accz.bin')
        jRightGyroXEntry = jUnisens.getEntry('right_gyrox.bin')
        jRightGyroYEntry = jUnisens.getEntry('right_gyroy.bin')
        jRightGyroZEntry = jUnisens.getEntry('right_gyroz.bin')
        count = jLeftAccXEntry.getCount()
        sample_rate = jLeftAccXEntry.getSampleRate()
        left_acc = _parse_j2py(jLeftAccXEntry.readScaled(count),
                               jLeftAccYEntry.readScaled(count),
                               jLeftAccZEntry.readScaled(count))
        left_gyro = _parse_j2py(jLeftGyroXEntry.readScaled(count),
                                jLeftGyroYEntry.readScaled(count),
                                jLeftGyroZEntry.readScaled(count))
        right_acc = _parse_j2py(jRightAccXEntry.readScaled(count),
                                jRightAccYEntry.readScaled(count),
                                jRightAccZEntry.readScaled(count))
        right_gyro = _parse_j2py(jRightGyroXEntry.readScaled(count),
                                 jRightGyroYEntry.readScaled(count),
                                 jRightGyroZEntry.readScaled(count))
        dt = 1000000 // OREBA_FREQUENCY
        timestamps = range(0, count*dt, dt)
        return timestamps, left_acc, left_gyro, right_acc, right_gyro

    def read_annotations(self, src_dir, subject_id):
        def _time_to_ms(time):
            t = dt.datetime.strptime(time, '%M:%S.%f')
            return t.minute * 60 * 1000 * 1000 + t.second * 1000 * 1000 \
                + t.microsecond
        path = os.path.join(src_dir, subject_id, subject_id + "_annotations.csv")
        assert os.path.isfile(path), "Couldn't find annotations file"
        start_time, end_time = [], []
        label_1, label_2, label_3, label_4 = [], [], [], []
        with open(path) as dest_f:
            next(dest_f)
            for row in csv.reader(dest_f, delimiter=','):
                start_time.append(_time_to_ms(row[0]))
                end_time.append(_time_to_ms(row[1]))
                label_1.append(row[4])
                label_2.append(row[5])
                label_3.append(row[6])
                label_4.append(row[7])
        return [start_time, end_time, label_1, label_2, label_3, label_4]

    def read_dominant(self, src_dir, subject_id):
        file_full_name = os.path.join(src_dir, 'dom_hand_info.csv')
        dom_hand_info = csv.reader(open(file_full_name, 'r'), delimiter=',')
        next(dom_hand_info, None)
        for row in dom_hand_info:
            if subject_id == row[0]:
                return row[1].strip().lower()
        return 'not found'

    def get_labels(self, annotations, timestamps):
        """Infer labels from annotations and timestamps"""
        num = len(timestamps)
        labels_1 = np.empty(num, dtype='U25'); labels_1.fill(OREBA_DEFAULT_LABEL)
        labels_2 = np.empty(num, dtype='U25'); labels_2.fill(OREBA_DEFAULT_LABEL)
        labels_3 = np.empty(num, dtype='U25'); labels_3.fill(OREBA_DEFAULT_LABEL)
        labels_4 = np.empty(num, dtype='U25'); labels_4.fill(OREBA_DEFAULT_LABEL)
        for start_time, end_time, label_1, label_2, label_3, label_4 in zip(*annotations):
            start_frame = np.argmax(np.array(timestamps) >= start_time)
            end_frame = np.argmax(np.array(timestamps) > end_time)
            labels_1[start_frame:end_frame] = label_1
            labels_2[start_frame:end_frame] = label_2
            labels_3[start_frame:end_frame] = label_3
            labels_4[start_frame:end_frame] = label_4
        return list(labels_1), list(labels_2), list(labels_3), list(labels_4)

    def done(self):
        jpype.shutdownJVM()

class ClemsonReader:
    def read_inert(self, data_dir, subject_id, session):
        dir = os.path.join(data_dir, subject_id, session)
        files = glob.glob(os.path.join(dir, "*.txt"))
        assert files, "No raw data found for {} {}".format(subject_id, session)
        acc = []
        gyro = []
        with open(files[0]) as dest_f:
            for row in csv.reader(dest_f, delimiter='\t'):
                acc_x = (float(row[0]) - 1.65) * 1000.0 / CLEMSON_ACC_SENSITIVITY
                acc_y = (float(row[1]) - 1.65) * 1000.0 / CLEMSON_ACC_SENSITIVITY
                acc_z = (float(row[2]) - 1.65) * 1000.0 / CLEMSON_ACC_SENSITIVITY
                acc.append([acc_x, acc_y, acc_z])
                # Todo estimate zero rate output mean instead of simply using 1.25!
                gyro_x = (float(row[3])-1.25) * 1000.0 / CLEMSON_GYRO_SENSITIVITY
                gyro_y = (float(row[4])-1.25) * 1000.0 / CLEMSON_GYRO_SENSITIVITY
                gyro_z = (float(row[5])-1.25) * 1000.0 / CLEMSON_GYRO_SENSITIVITY
                gyro.append([gyro_x, gyro_y, gyro_z])
        dt = 1000000 // CLEMSON_FREQUENCY
        timestamps = range(0, len(acc)*dt, dt)
        return timestamps, acc, gyro

    def read_annotations(self, gesture_dir, bite_dir):
        def _index_to_ms(index):
            dt = 1000000 // CLEMSON_FREQUENCY
            return index * dt
        # Read gesture ground truth
        label_1, start_time, end_time = [], [], []
        with open(gesture_dir) as dest_f:
            for row in csv.reader(dest_f, delimiter='\t'):
                label_1.append(row[0])
                start_time.append(_index_to_ms(int(row[1])))
                end_time.append(_index_to_ms(int(row[2])))
        # Read bite ground truth by matching with gestures
        label_2, label_3, label_4, label_5 = [], [], [], []
        for i in range(0, len(label_1)):
            if label_1[i] == 'bite' or label_1[i] == 'drink':
                match_found = False
                with open(bite_dir) as dest_f:
                    for row in csv.reader(dest_f, delimiter='\t'):
                        time = _index_to_ms(int(row[1]))
                        if time >= start_time[i] and time <= end_time[i]:
                            label_2.append(row[2])
                            label_3.append(row[3])
                            label_4.append(row[4])
                            label_5.append(row[5])
                            match_found = True
                            break
                if not match_found:
                    label_2.append("NA")
                    label_3.append("NA")
                    label_4.append("NA")
                    label_5.append("NA")
            else:
                label_2.append("NA")
                label_3.append("NA")
                label_4.append("NA")
                label_5.append("NA")
        return [start_time, end_time, label_1, label_2, label_3, label_4, label_5]

    def read_hand(self, src_dir, subject_id):
        """Read handedness, which is the hand sensor was placed on"""
        file_path = os.path.join(src_dir, "demographics.xlsx")
        workbook = xlrd.open_workbook(file_path)
        sheet = workbook.sheet_by_index(0)
        for rowx in range(sheet.nrows):
            cols = sheet.row_values(rowx)
            if cols[0].lower() == subject_id:
                return cols[4].lower()
        return "NA"

    def get_labels(self, annotations, timestamps):
        """Infer labels from annotations and timestamps"""
        num = len(timestamps)
        labels_1 = np.empty(num, dtype='U25'); labels_1.fill(CLEMSON_DEFAULT_LABEL)
        labels_2 = np.empty(num, dtype='U25'); labels_2.fill(CLEMSON_DEFAULT_LABEL)
        labels_3 = np.empty(num, dtype='U25'); labels_3.fill(CLEMSON_DEFAULT_LABEL)
        labels_4 = np.empty(num, dtype='U25'); labels_4.fill(CLEMSON_DEFAULT_LABEL)
        labels_5 = np.empty(num, dtype='U25'); labels_5.fill(CLEMSON_DEFAULT_LABEL)
        for start_time, end_time, label_1, label_2, label_3, label_4, label_5 \
            in zip(*annotations):
            start_frame = np.argmax(np.array(timestamps) >= start_time)
            end_frame = np.argmax(np.array(timestamps) > end_time)
            labels_1[start_frame:end_frame] = label_1
            labels_2[start_frame:end_frame] = label_2
            labels_3[start_frame:end_frame] = label_3
            labels_4[start_frame:end_frame] = label_4
            labels_5[start_frame:end_frame] = label_5
        return list(labels_1), list(labels_2), list(labels_3), list(labels_4), list(labels_5)

class FICReader:
    def read_pickle(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    def get_labels(self, annotations, timestamps):
        """Infer labels from annotations and timestamps"""
        num = len(timestamps)
        labels_1 = np.empty(num, dtype='U25'); labels_1.fill(FIC_DEFAULT_LABEL)
        for start_time, end_time in annotations:
            start_frame = np.argmax(np.array(timestamps) >= start_time)
            end_frame = np.argmax(np.array(timestamps) > end_time)
            labels_1[start_frame:end_frame] = "Intake"
        return list(labels_1)
