import csv
from datetime import timedelta, time, datetime
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

class OrebaWriter:
    def __init__(self, path):
        self.path = path

    def write(self, subject_id, timestamps, left_acc, left_gyro,
        right_acc, right_gyro, dominant_hand, label_1, label_2, label_3,
        label_4, exp_uniform, exp_format):
        frame_ids = range(0, len(timestamps))
        def _format_time(t):
            return (datetime.min + timedelta(microseconds=t)).time().strftime('%H:%M:%S.%f')
        timestamps = [_format_time(t) for t in timestamps]
        if exp_format == 'csv':
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
        elif exp_format == 'tfrecord':
            with tf.io.TFRecordWriter(self.path) as tfrecord_writer:
                for i in range(0, len(timestamps)):
                    if exp_uniform == 'True':
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'example/subject_id': _bytes_feature(subject_id.encode()),
                            'example/frame_id': _int64_feature(frame_ids[i]),
                            'example/timestamp': _bytes_feature(timestamps[i].encode()),
                            'example/dom_acc': _floats_feature(right_acc[i].ravel()),
                            'example/dom_gyro': _floats_feature(right_gyro[i].ravel()),
                            'example/ndom_acc': _floats_feature(left_acc[i].ravel()),
                            'example/ndom_gyro': _floats_feature(left_gyro[i].ravel()),
                            'example/dominant_hand': _bytes_feature(dominant_hand.encode()),
                            'example/label_1': _bytes_feature(label_1[i].encode()),
                            'example/label_2': _bytes_feature(label_2[i].encode()),
                            'example/label_3': _bytes_feature(label_3[i].encode()),
                            'example/label_4': _bytes_feature(label_4[i].encode())
                        }))
                    else:
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'example/subject_id': _bytes_feature(subject_id.encode()),
                            'example/frame_id': _int64_feature(frame_ids[i]),
                            'example/timestamp': _bytes_feature(timestamps[i].encode()),
                            'example/left_acc': _floats_feature(left_acc[i].ravel()),
                            'example/left_gyro': _floats_feature(left_gyro[i].ravel()),
                            'example/right_acc': _floats_feature(right_acc[i].ravel()),
                            'example/right_gyro': _floats_feature(right_gyro[i].ravel()),
                            'example/dominant_hand': _bytes_feature(dominant_hand.encode()),
                            'example/label_1': _bytes_feature(label_1[i].encode()),
                            'example/label_2': _bytes_feature(label_2[i].encode()),
                            'example/label_3': _bytes_feature(label_3[i].encode()),
                            'example/label_4': _bytes_feature(label_4[i].encode())
                        }))
                    tfrecord_writer.write(example.SerializeToString())


    def write_summary(self, ids, n_gestures, t_gestures, t_total):
        with open(self.path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["id", "n_gestures", "t_gestures", "t_total"])
            for i in range(0, len(ids)):
                writer.writerow([ids[i], n_gestures[i], t_gestures[i], t_total[i]])

class ClemsonWriter:
    def __init__(self, path):
        self.path = path

    def write(self, subject_id, session_id, timestamps, acc, gyro, hand,
        label_1, label_2, label_3, label_4, label_5, exp_format):
        frame_ids = range(0, len(timestamps))
        id = subject_id + "_" + session_id
        def _format_time(t):
            return (datetime.min + timedelta(microseconds=t)).time().strftime('%H:%M:%S.%f')
        timestamps = [_format_time(t) for t in timestamps]
        if exp_format == 'csv':
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
        elif exp_format == 'tfrecord':
            with tf.io.TFRecordWriter(self.path) as tfrecord_writer:
                for i in range(0, len(timestamps)):
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'example/subject_id': _bytes_feature(id.encode()),
                        'example/frame_id': _int64_feature(frame_ids[i]),
                        'example/timestamp': _bytes_feature(timestamps[i].encode()),
                        'example/acc': _floats_feature(acc[i].ravel()),
                        'example/gyro': _floats_feature(gyro[i].ravel()),
                        'example/label_1': _bytes_feature(label_1[i].encode()),
                        'example/label_2': _bytes_feature(label_2[i].encode()),
                        'example/label_3': _bytes_feature(label_3[i].encode()),
                        'example/label_4': _bytes_feature(label_4[i].encode()),
                        'example/label_5': _bytes_feature(label_5[i].encode())
                    }))
                    tfrecord_writer.write(example.SerializeToString())

    def write_summary(self, ids, n_gestures, t_gestures, t_total):
        with open(self.path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["id", "n_gestures", "t_gestures", "t_total"])
            for i in range(0, len(ids)):
                writer.writerow([ids[i], n_gestures[i], t_gestures[i], t_total[i]])

class FICWriter:
    def __init__(self, path):
        self.path = path

    def write(self, subject_id, session_id, timestamps, acc, gyro, label_1, exp_format):
        frame_ids = list(range(0, len(timestamps)))
        id = str(subject_id) + "_" + str(session_id)
        def _format_time(t):
            return (datetime.min + timedelta(seconds=t)).time().strftime('%H:%M:%S.%f')
        timestamps = [_format_time(t) for t in timestamps]
        if exp_format == 'csv':
            with open(self.path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(["id", "frame_id", "timestamp", "acc_x", "acc_y",
                    "acc_z", "gyro_x", "gyro_y", "gyro_z", "label_1"])
                for i in range(0, len(timestamps)):
                    writer.writerow([id, frame_ids[i], timestamps[i],
                        acc[i][0], acc[i][1], acc[i][2], gyro[i][0], gyro[i][1],
                        gyro[i][2], label_1[i]])
        elif exp_format == 'tfrecord':
            with tf.io.TFRecordWriter(self.path) as tfrecord_writer:
                for i in range(0, len(timestamps)):
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'example/subject_id': _bytes_feature(id.encode()),
                        'example/frame_id': _int64_feature(frame_ids[i]),
                        'example/timestamp': _bytes_feature(timestamps[i].encode()),
                        'example/acc': _floats_feature(acc[i].ravel()),
                        'example/gyro': _floats_feature(gyro[i].ravel()),
                        'example/label_1': _bytes_feature(label_1[i].encode())
                    }))
                    tfrecord_writer.write(example.SerializeToString())

    def write_summary(self, ids, n_gestures, t_gestures, t_total):
        with open(self.path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["id", "n_gestures", "t_gestures", "t_total"])
            for i in range(0, len(ids)):
                writer.writerow([ids[i], n_gestures[i], t_gestures[i], t_total[i]])
