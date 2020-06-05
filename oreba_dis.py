"""OREBA dataset"""

import numpy as np
import datetime as dt
import csv
import os
import logging
import jpype.imports
import xml.etree.cElementTree as etree
import tensorflow as tf

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',
  datefmt='%H:%M:%S', level=logging.INFO)

FREQUENCY = 64
FLIP_ACC = [1., -1., 1.]
FLIP_GYRO = [-1., 1., -1.]
TIME_FACTOR = 1000000
DEFAULT_LABEL = "Idle"

TRAIN_IDS = ['1001','1002','1003','1006','1007','1008','1012','1013','1014',
  '1017','1018','1019','1022','1023','1024','1027','1028','1029','1032',
  '1033','1035','1039','1040','1041','1045','1046','1047','1051','1052',
  '1053','1056','1057','1059','1063','1064','1067','1072','1073','1077',
  '1079','1080','1083','1084','1085','1088','1089','1090','1093','1094',
  '1095','1098','1099','1100','1103','1104','1105','1109','1110','1111',
  '1115','1116']
VALID_IDS = ['1004','1010','1015','1020','1025','1030','1036','1043','1048',
  '1054','1060','1068','1075','1081','1086','1091','1096','1101','1107',
  '1112']
TEST_IDS = ['1005','1011','1016','1021','1026','1031','1037','1044','1050',
  '1055','1061','1071','1076','1082','1087','1092','1097','1102','1108',
  '1113']

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
    # Init jpype
    jpype.addClassPath('unisens/org.unisens.jar')
    jpype.addClassPath('unisens/org.unisens.ri.jar')
    jpype.startJVM()
    # Class names
    self.names_1, self.names_2, self.names_3, self.names_4 = \
      self.__class_names()

  def __class_names(self):
    """Get class names from label master file"""
    label_spec_path = os.path.join(self.src_dir, self.label_spec)
    assert os.path.isfile(label_spec_path), "Couldn't find label master file at {}".format(label_spec_path)
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
    ids = [x for x in next(os.walk(self.src_dir))[1]]
    return ids

  def check(self, id):
    return id != "1074_1"

  def data(self, _, id):
    logging.info("Reading raw data from Unisens")
    def _parse_j2py(jpype_x, jpype_y, jpype_z):
      """Convert from double[][] to list"""
      return list(zip([list(i)[0] for i in jpype_x],
              [list(i)[0] for i in jpype_y],
              [list(i)[0] for i in jpype_z]))
    dir = os.path.join(self.src_dir, id, "data_sensor")
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
    dt = TIME_FACTOR // FREQUENCY # In microseconds
    timestamps = range(0, count*dt, dt)
    return timestamps, {"left":  (left_acc, left_gyro), \
              "right": (right_acc, right_gyro)}

  def dominant(self, id):
    file_full_name = os.path.join(self.src_dir, self.dom_hand_spec)
    dom_hand_info = csv.reader(open(file_full_name, 'r'), delimiter=',')
    next(dom_hand_info, None)
    for row in dom_hand_info:
      if id == row[0]:
        return row[1].strip().lower()
    return "NA"

  def labels(self, _, id, timestamps):
    def _time_to_ms(time):
      t = dt.datetime.strptime(time, '%M:%S.%f')
      return t.minute * 60 * 1000 * 1000 + t.second * 1000 * 1000 \
        + t.microsecond
    path = os.path.join(self.src_dir, id, id + "_annotations.csv")
    assert os.path.isfile(path), "Couldn't find annotations file"
    # Read from file and infer labels from annotations and timestamps
    num = len(timestamps)
    labels_1 = np.empty(num, dtype='U25'); labels_1.fill(DEFAULT_LABEL)
    labels_2 = np.empty(num, dtype='U25'); labels_2.fill(DEFAULT_LABEL)
    labels_3 = np.empty(num, dtype='U25'); labels_3.fill(DEFAULT_LABEL)
    labels_4 = np.empty(num, dtype='U25'); labels_4.fill(DEFAULT_LABEL)
    with open(path) as dest_f:
      next(dest_f)
      for row in csv.reader(dest_f, delimiter=','):
        start_time = _time_to_ms(row[0])
        end_time = _time_to_ms(row[1])
        start_frame = np.argmax(np.array(timestamps) >= start_time)
        end_frame = np.argmax(np.array(timestamps) > end_time)
        if row[4] in self.names_1:
          labels_1[start_frame:end_frame] = row[4]
        elif self.label_spec_inherit:
          continue
        if row[5] in self.names_2:
          labels_2[start_frame:end_frame] = row[5]
        if row[6] in self.names_3:
          labels_3[start_frame:end_frame] = row[6]
        if row[7] in self.names_4:
          labels_4[start_frame:end_frame] = row[7]
    return (labels_1, labels_2, labels_3, labels_4)

  def write(self, path, id, timestamps, data, dominant_hand, labels):
    frame_ids = range(0, len(timestamps))
    def _format_time(t):
      return (dt.datetime.min + dt.timedelta(microseconds=t)).time().strftime('%H:%M:%S.%f')
    timestamps = [_format_time(t) for t in timestamps]
    right_acc = np.asarray(data["right"][0])
    right_gyro = np.asarray(data["right"][1])
    left_acc = np.asarray(data["left"][0])
    left_gyro = np.asarray(data["left"][1])
    assert len(timestamps) == len(right_acc), \
      "Number timestamps and acc readings must be equal"
    assert len(timestamps) == len(left_acc), \
      "Number timestamps and acc readings must be equal"
    assert len(timestamps) == len(right_gyro), \
      "Number timestamps and acc readings must be equal"
    assert len(timestamps) == len(left_gyro), \
      "Number timestamps and acc readings must be equal"
    if self.exp_format == 'csv':
      with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if self.exp_uniform:
          writer.writerow(["id", "frame_id", "timestamp",
            "dom_acc_x", "dom_acc_y", "dom_acc_z",
            "dom_gyro_x", "dom_gyro_y", "dom_gyro_z",
            "ndom_acc_x", "ndom_acc_y", "ndom_acc_z",
            "ndom_gyro_x", "ndom_gyro_y", "ndom_gyro_z",
            "dom_hand", "label_1", "label_2", "label_3", "label_4"])
          for i in range(0, len(timestamps)):
            writer.writerow([id, frame_ids[i], timestamps[i],
              right_acc[i][0], right_acc[i][1], right_acc[i][2],
              right_gyro[i][0], right_gyro[i][1], right_gyro[i][2],
              left_acc[i][0], left_acc[i][1], left_acc[i][2],
              left_gyro[i][0], left_gyro[i][1], left_gyro[i][2],
              dominant_hand, labels[0][i], labels[1][i],
              labels[2][i], labels[3][i]])
        else:
          writer.writerow(["id", "frame_id", "timestamp",
            "left_acc_x", "left_acc_y", "left_acc_z",
            "left_gyro_x", "left_gyro_y", "left_gyro_z",
            "right_acc_x", "right_acc_y", "right_acc_z",
            "right_gyro_x", "right_gyro_y", "right_gyro_z",
            "dominant_hand", "label_1", "label_2", "label_3", "label_4"])
          for i in range(0, len(timestamps)):
            writer.writerow([id, frame_ids[i], timestamps[i],
              left_acc[i][0], left_acc[i][1], left_acc[i][2],
              left_gyro[i][0], left_gyro[i][1], left_gyro[i][2],
              right_acc[i][0], right_acc[i][1], right_acc[i][2],
              right_gyro[i][0], right_gyro[i][1], right_gyro[i][2],
              dominant_hand, labels[0][i], labels[1][i],
              labels[2][i], labels[3][i]])
    elif self.exp_format == 'tfrecord':
      with tf.io.TFRecordWriter(path) as tfrecord_writer:
        for i in range(0, len(timestamps)):
          if self.exp_uniform:
            example = tf.train.Example(features=tf.train.Features(feature={
              'example/subject_id': _bytes_feature(id.encode()),
              'example/frame_id': _int64_feature(frame_ids[i]),
              'example/timestamp': _bytes_feature(timestamps[i].encode()),
              'example/dom_acc': _floats_feature(right_acc[i].ravel()),
              'example/dom_gyro': _floats_feature(right_gyro[i].ravel()),
              'example/ndom_acc': _floats_feature(left_acc[i].ravel()),
              'example/ndom_gyro': _floats_feature(left_gyro[i].ravel()),
              'example/dominant_hand': _bytes_feature(dominant_hand.encode()),
              'example/label_1': _bytes_feature(labels[0][i].encode()),
              'example/label_2': _bytes_feature(labels[1][i].encode()),
              'example/label_3': _bytes_feature(labels[2][i].encode()),
              'example/label_4': _bytes_feature(labels[3][i].encode())
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
              'example/label_1': _bytes_feature(labels[0][i].encode()),
              'example/label_2': _bytes_feature(labels[1][i].encode()),
              'example/label_3': _bytes_feature(labels[2][i].encode()),
              'example/label_4': _bytes_feature(labels[3][i].encode())
            }))
          tfrecord_writer.write(example.SerializeToString())

  def done(self):
    jpype.shutdownJVM()
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
