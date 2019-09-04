import csv
from datetime import timedelta, time, datetime

class OrebaWriter:
    def __init__(self, path):
        self.path = path

    def write_dev(self, subject_id, timestamps, left_acc, left_gyro,
        right_acc, right_gyro, dominant_hand, label_1, label_2, label_3,
        label_4, exp_uniform):
        frame_ids = range(0, len(timestamps))
        def _format_time(t):
            return (datetime.min + timedelta(microseconds=t)).time().strftime('%H:%M:%S.%f')
        timestamps = [_format_time(t) for t in timestamps]
        with open(self.path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            if exp_uniform == 'True':
                writer.writerow(["id", "frame_id", "timestamp",
                    "dom_acc_x", "dom_acc_y", "dom_acc_z",
                    "dom_gyro_x", "dom_gyro_y", "dom_gyro_z",
                    "ndom_acc_x", "ndom_acc_y", "ndom_acc_z",
                    "ndom_gyro_x", "ndom_gyro_y", "ndom_gyro_z",
                    "dom_hand", "label_1", "label_2", "label_3", "label_4"])
                for i in range(0, len(timestamps)):
                    writer.writerow([subject_id, frame_ids[i], timestamps[i],
                        right_acc[i][0], right_acc[i][1], right_acc[i][2],
                        right_gyro[i][0], right_gyro[i][1], right_gyro[i][2],
                        left_acc[i][0], left_acc[i][1], left_acc[i][2],
                        left_gyro[i][0], left_gyro[i][1], left_gyro[i][2],
                        dominant_hand, label_1[i], label_2[i], label_3[i],
                        label_4[i]])
            else:
                writer.writerow(["id", "frame_id", "timestamp",
                    "left_acc_x", "left_acc_y", "left_acc_z",
                    "left_gyro_x", "left_gyro_y", "left_gyro_z",
                    "right_acc_x", "right_acc_y", "right_acc_z",
                    "right_gyro_x", "right_gyro_y", "right_gyro_z",
                    "dominant_hand", "label_1", "label_2", "label_3", "label_4"])
                for i in range(0, len(timestamps)):
                    writer.writerow([subject_id, frame_ids[i], timestamps[i],
                        left_acc[i][0], left_acc[i][1], left_acc[i][2],
                        left_gyro[i][0], left_gyro[i][1], left_gyro[i][2],
                        right_acc[i][0], right_acc[i][1], right_acc[i][2],
                        right_gyro[i][0], right_gyro[i][1], right_gyro[i][2],
                        dominant_hand, label_1[i], label_2[i], label_3[i],
                        label_4[i]])

    # TODO Decide how to present data for publication (hand flipping etc.)
    def write_pub(self, subject_id, timestamps, left_acc, left_acc_0, left_gyro,
        left_gyro_0, right_acc, right_acc_0, right_gyro, right_gyro_0,
        dominant_hand, label_1, label_2, label_3, label_4):
        frame_ids = range(0, len(timestamps))
        def _format_time(t):
            return (datetime.min + timedelta(microseconds=t)).time().strftime('%H:%M:%S.%f')
        timestamps = [_format_time(t) for t in timestamps]
        with open(self.path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["subject_id", "frame_id", "timestamp",
                "left_acc_x", "left_acc_y", "left_acc_z",
                "left_acc_x_0", "left_acc_y_0", "left_acc_z_0",
                "left_gyro_x", "left_gyro_y", "left_gyro_z",
                "left_gyro_x_0", "left_gyro_y_0", "left_gyro_z_0",
                "right_acc_x", "right_acc_y", "right_acc_z",
                "right_acc_x_0", "right_acc_y_0", "right_acc_z_0",
                "right_gyro_x", "right_gyro_y", "right_gyro_z",
                "right_gyro_x_0", "right_gyro_y_0", "right_gyro_z_0",
                "dominant_hand", "label_1", "label_2", "label_3", "label_4"])
            for i in range(0, len(timestamps)):
                writer.writerow([subject_ids, frame_ids[i], timestamps[i],
                    left_acc[i][0], left_acc[i][1], left_acc[i][2],
                    left_acc_0[i][0], left_acc_0[i][1], left_acc_0[i][2],
                    left_gyro[i][0], left_gyro[i][1], left_gyro[i][2],
                    left_gyro_0[i][0], left_gyro_0[i][1], left_gyro_0[i][2],
                    right_acc[i][0], right_acc[i][1], right_acc[i][2],
                    right_acc_0[i][0], right_acc_0[i][1], right_acc_0[i][2],
                    right_gyro[i][0], right_gyro[i][1], right_gyro[i][2],
                    right_gyro_0[i][0], right_gyro_0[i][1], right_gyro_0[i][2],
                    dominant_hand, label_1[i], label_2[i], label_3[i],
                    label_4[i]])

class ClemsonWriter:
    def __init__(self, path):
        self.path = path

    def write_pub(self, subject_id, session_id, timestamps, acc, acc_0, gyro,
        gyro_0, hand, label_1, label_2, label_3, label_4, label_5):
        frame_ids = range(0, len(timestamps))
        id = subject_id + "_" + session_id
        def _format_time(t):
            return (datetime.min + timedelta(microseconds=t)).time().strftime('%H:%M:%S.%f')
        timestamps = [_format_time(t) for t in timestamps]
        with open(self.path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["id", "frame_id", "timestamp", "acc_x", "acc_y",
                "acc_z", "acc_x_0", "acc_y_0", "acc_z_0", "gyro_x", "gyro_y",
                "gyro_z", "gyro_x_0", "gyro_y_0", "gyro_z_0", "hand",
                "label_1", "label_2", "label_3", "label_4", "label_5"])
            for i in range(0, len(timestamps)):
                writer.writerow([id, frame_ids[i], timestamps[i],
                    acc[i][0], acc[i][1], acc[i][2], acc_0[i][0], acc_0[i][1],
                    acc_0[i][2], gyro[i][0], gyro[i][1], gyro[i][2],
                    gyro_0[i][0], gyro_0[i][1], gyro_0[i][2], hand, label_1[i],
                    label_2[i], label_3[i], label_4[i], label_5[i]])

    def write_dev(self, subject_id, session_id, timestamps, acc, gyro, hand,
        label_1, label_2, label_3, label_4, label_5):
        frame_ids = range(0, len(timestamps))
        id = subject_id + "_" + session_id
        def _format_time(t):
            return (datetime.min + timedelta(microseconds=t)).time().strftime('%H:%M:%S.%f')
        timestamps = [_format_time(t) for t in timestamps]
        with open(self.path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["id", "frame_id", "timestamp", "acc_x", "acc_y",
                "acc_z", "gyro_x", "gyro_y", "gyro_z", "hand",
                "label_1", "label_2", "label_3", "label_4", "label_5"])
            for i in range(0, len(timestamps)):
                writer.writerow([id, frame_ids[i], timestamps[i],
                    acc[i][0], acc[i][1], acc[i][2], gyro[i][0], gyro[i][1],
                    gyro[i][2], hand, label_1[i], label_2[i], label_3[i],
                    label_4[i], label_5[i]])

class FICWriter:
    def __init__(self, path):
        self.path = path

    def write_pub(self, subject_id, session_id, timestamps, acc, acc_0, gyro,
        gyro_0, label_1, units):
        frame_ids = range(0, len(timestamps))
        id = str(subject_id) + "_" + str(session_id)
        def _format_time(t, units):
            if units == 'nanos':
                t *= 1000000
            return (datetime.min + timedelta(milliseconds=t)).time().strftime('%H:%M:%S.%f')
        timestamps = [_format_time(t, units) for t in timestamps]
        with open(self.path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["id", "frame_id", "timestamp", "acc_x", "acc_y",
                "acc_z", "acc_x_0", "acc_y_0", "acc_z_0", "gyro_x", "gyro_y",
                "gyro_z", "gyro_x_0", "gyro_y_0", "gyro_z_0", "label_1"])
            for i in range(0, len(timestamps)):
                writer.writerow([id, frame_ids[i], timestamps[i],
                    acc[i][0], acc[i][1], acc[i][2], acc_0[i][0], acc_0[i][1],
                    acc_0[i][2], gyro[i][0], gyro[i][1], gyro[i][2],
                    gyro_0[i][0], gyro_0[i][1], gyro_0[i][2], label_1[i]])

    def write_dev(self, subject_id, session_id, timestamps, acc, gyro,
        label_1, units):
        frame_ids = range(0, len(timestamps))
        id = str(subject_id) + "_" + str(session_id)
        def _format_time(t, units):
            if units == 'nanos':
                t /= 1000000
            return (datetime.min + timedelta(milliseconds=t)).time().strftime('%H:%M:%S.%f')
        timestamps = [_format_time(t, units) for t in timestamps]
        with open(self.path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["id", "frame_id", "timestamp", "acc_x", "acc_y",
                "acc_z", "gyro_x", "gyro_y", "gyro_z", "label_1"])
            for i in range(0, len(timestamps)):
                writer.writerow([id, frame_ids[i], timestamps[i],
                    acc[i][0], acc[i][1], acc[i][2], gyro[i][0], gyro[i][1],
                    gyro[i][2], label_1[i]])
