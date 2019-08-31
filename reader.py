import csv
import os
import math
import jpype
import jpype.imports
import numpy as np
import datetime as dt

OREBA_FREQUENCY = 64

class UnisensReader:
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
        # TODO read dom_hand information
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

    def done(self):
        jpype.shutdownJVM()


class OrebaReader:
    def __init__(self, filename):
        self.filename = filename

    def read_inert(self):
        acc = []
        gyro = []
        with open(self.filename) as dest_f:
            reader = csv.reader(dest_f, delimiter=',')
            next(reader, None)  # skip the headers
            for row in reader:
                acc_x = float(row[2]) * OREBA_ACC_MULTIPLIER
                acc_y = float(row[3]) * OREBA_ACC_MULTIPLIER
                acc_z = float(row[4]) * OREBA_ACC_MULTIPLIER
                acc.append([acc_x, acc_y, acc_z])
                # Gyro data is in degrees per second, so convert to radians
                gyro_x = math.radians(float(row[5]) * OREBA_GYRO_MULTIPLIER)
                gyro_y = math.radians(float(row[6]) * OREBA_GYRO_MULTIPLIER)
                gyro_z = math.radians(float(row[7]) * OREBA_GYRO_MULTIPLIER)
                gyro.append([gyro_x, gyro_y, gyro_z])

        return acc, gyro
