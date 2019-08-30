import csv
import math

OREBA_GYRO_MULTIPLIER = 0.07000000066757203
OREBA_ACC_MULTIPLIER = 0.000487999978078126

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
