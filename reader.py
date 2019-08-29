import csv
import numpy as np

OREBA_GYRO_MULTIPLIER = 0.07000000066757203

class OrebaReader:
    def __init__(self, filename):
        self.filename = filename

    def read_gyro(self):
        gyro = []
        with open(self.filename) as dest_f:
            reader = csv.reader(dest_f, delimiter=',')
            next(reader, None)  # skip the headers
            for row in reader:
                # Convert from degrees per second to radians per second
                gyro_x = float(row[5]) * OREBA_GYRO_MULTIPLIER / 180.0 * np.pi
                gyro_y = float(row[6]) * OREBA_GYRO_MULTIPLIER / 180.0 * np.pi
                gyro_z = float(row[7]) * OREBA_GYRO_MULTIPLIER / 180.0 * np.pi
                gyro.append([gyro_x, gyro_y, gyro_z])
        return gyro
