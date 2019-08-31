import csv
from datetime import timedelta, time, datetime

class TwoHandsWriter:
    def __init__(self, path):
        self.path = path

    def write(self, subject_id, timestamps, left_acc, left_acc_0, left_gyro,
        right_acc, right_acc_0, right_gyro, label_1, label_2, label_3, label_4):
        # dominant_hand, annot_1, annot_2, annot_3, annot_4
        frame_ids = range(0, len(timestamps))
        subject_ids = [subject_id] * len(timestamps)
        def _format_time(t):
            return (datetime.min + timedelta(microseconds=t)).time().strftime('%H:%M:%S.%f')
        timestamps = [_format_time(t) for t in timestamps]
        # Write filtered acc to csv
        with open(self.path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["subject_id", "frame_id", "timestamp",
                "left_acc_x", "left_acc_y", "left_acc_z", "left_acc_0_x",
                "left_acc_0_y", "left_acc_0_z", "left_gyro_x", "left_gyro_y",
                "left_gyro_x", "right_acc_x", "right_acc_y", "right_acc_z",
                "right_acc_0_x", "right_acc_0_y", "right_acc_0_z",
                "right_gyro_x", "right_gyro_y", "right_gyro_x", "label_1",
                "label_2", "label_3", "label_4"])
            for i in range(0, len(timestamps)):
                writer.writerow([subject_ids[i], frame_ids[i], timestamps[i],
                    left_acc[i][0], left_acc[i][1], left_acc[i][2],
                    left_acc_0[i][0], left_acc_0[i][1], left_acc_0[i][2],
                    left_gyro[i][0], left_gyro[i][1], left_gyro[i][2],
                    right_acc[i][0], right_acc[i][1], right_acc[i][2],
                    right_acc_0[i][0], right_acc_0[i][1], right_acc_0[i][2],
                    right_gyro[i][0], right_gyro[i][1], right_gyro[i][2],
                    label_1[i], label_2[i], label_3[i], label_4[i]])
