import numpy as np
import quaternion
from visualization import PygameViewer
from fusion import MadgwickFusion
from reader import OrebaReader, ClemsonReader, FICReader
from writer import OrebaWriter, ClemsonWriter, FICWriter
import csv
import argparse
import os
import logging
from scipy import signal
from math import radians
import itertools

OREBA_FREQUENCY = 64
CLEMSON_FREQUENCY = 15
FIC_FREQUENCY = 64
UPDATE_RATE = 16
FACTOR_MILLIS = 1000
FACTOR_NANOS = 1000000000
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S', level=logging.INFO)

class Node:
    """A node is an edge of the cuboid"""
    def __init__(self, coords, color):
        self.x = coords[0]
        self.y = coords[1]
        self.z = coords[2]
        self.color = color

class Face:
    """A face of the cuboid is defined using the indices of four nodes"""
    def __init__(self, nodeIdxs, color):
        self.nodeIdxs = nodeIdxs
        self.color = color

class Cuboid:
    """The cuboid"""
    def __init__(self, quaternion):
        self.nodes = []
        self.faces = []
        self.q = quaternion

    def set_nodes(self, nodes):
        self.nodes = nodes

    def set_faces(self, faces):
        self.faces = faces

    def set_quaternion(self, q):
        self.q = q

    def rotate_quaternion(self, w, dt):
        self.q = dt/2 * self.q * np.quaternion(0, w[0], w[1], w[2]) + self.q

    def rotate_point(self, point):
        return quaternion.rotate_vectors(self.q, point)

    def convert_to_computer_frame(self, point):
        computerFrameChangeMatrix = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
        return np.matmul(computerFrameChangeMatrix, point)

    def get_euler_attitude(self):
        def _rad2deg(rad):
            return rad / np.pi * 180
        m = quaternion.as_rotation_matrix(self.q)
        test = -m[2, 0]
        if test > 0.99999:
            yaw = 0
            pitch = np.pi / 2
            roll = np.arctan2(m[0, 1], m[0, 2])
        elif test < -0.99999:
            yaw = 0
            pitch = -np.pi / 2
            roll = np.arctan2(-m[0, 1], -m[0, 2])
        else:
            yaw = np.arctan2(m[1, 0], m[0, 0])
            pitch = np.arcsin(-m[2, 0])
            roll = np.arctan2(m[2, 1], m[2, 2])
        yaw = _rad2deg(yaw)
        pitch = _rad2deg(pitch)
        roll = _rad2deg(roll)
        return yaw, pitch, roll

def initialize_cuboid(quaternion):
    """Initialize cuboid with nodes and faces"""

    # The cuboid with initial quaternion
    cuboid = Cuboid(quaternion)

    # Define nodes
    nodes = []
    nodes.append(Node([-1.5, -1, -0.1], [255, 255, 255]))
    nodes.append(Node([-1.5, -1, 0.1], [255, 255, 255]))
    nodes.append(Node([-1.5, 1, -0.1], [255, 255, 255]))
    nodes.append(Node([-1.5, 1, 0.1], [255, 255, 255]))
    nodes.append(Node([1.5, -1, -0.1], [255, 255, 255]))
    nodes.append(Node([1.5, -1, 0.1], [255, 255, 255]))
    nodes.append(Node([1.5, 1, -0.1], [255, 255, 255]))
    nodes.append(Node([1.5, 1, 0.1], [255, 255, 255]))
    cuboid.set_nodes(nodes)

    # Define faces
    faces = []
    faces.append(Face([0, 2, 6, 4], [255, 0, 255]))
    faces.append(Face([0, 1, 3, 2], [255, 0, 0]))
    faces.append(Face([1, 3, 7, 5], [0, 255, 0]))
    faces.append(Face([4, 5, 7, 6], [0, 0, 255]))
    faces.append(Face([2, 3, 7, 6], [0, 255, 255]))
    faces.append(Face([0, 1, 5, 4], [255, 255, 0]))
    cuboid.set_faces(faces)

    return cuboid

def estimate_initial_quaternion(acc, gyro, data_freq, seconds=5, inc_deg=30):
    """Estimate the initital quaternion from evenly distributed possibilities"""
    # Uniformly sampled possible initial quaternions
    qs = [np.quaternion(1, radians(x), radians(y), radians(z)) for x, y, z in list(itertools.product( \
            range(0, 180, inc_deg), range(0, 180, inc_deg), range(0, 180, inc_deg)))]
    loss = []
    for q in qs:
        # Remove gravity for the first seconds
        num = seconds * data_freq
        q = np.quaternion(q)
        # We are assuming that there is not much force except gravity
        acc_0 = remove_gravity(acc[0:num], gyro[0:num], q, data_freq, 0, False)
        # If gravity is successfully removed, absolute values are low
        loss.append(np.sum(np.absolute(acc_0)))
    return qs[np.argmin(loss)]

def remove_gravity(acc, gyro, init_q, data_freq, update_freq, vis):
    """Remove gravity for one hand."""
    # Initialize
    cuboid = initialize_cuboid(init_q)
    madgwick = MadgwickFusion(cuboid.q, data_freq)
    # Initialize visualization
    pv = None
    if vis == 'True':
        pv = PygameViewer(640, 480, cuboid, data_freq)
    # Process
    acc_0 = []
    i = 0
    for acc_t, gyro_t in zip(acc, gyro):
        # Sensor fusion update
        madgwick.update(acc_t, gyro_t)
        cuboid.set_quaternion(madgwick.q)
        # Remove gravity from acceleration
        acc_t0 = quaternion.rotate_vectors(madgwick.q, np.array(acc_t))
        acc_t0 -= np.array([0, 0, 1])
        acc_t0 = quaternion.rotate_vectors(madgwick.q.inverse(), acc_t0)
        acc_0.append(acc_t0.tolist())
        # Update screen according to update rate
        if vis == 'True':
            if i % (data_freq//update_freq) == 0:
                if not pv.update():
                    break
        i += 1
    return acc_0

def standardize(x):
    np_x = np.array(x)
    np_x -= np.mean(np_x, axis=0)
    np_x /= np.std(np_x)
    return list(np_x)

def main(args=None):
    """Main"""
    if args.reader == 'Oreba':
        # For OREBA data
        # Read subjects
        subject_ids = [x for x in next(os.walk(args.src_dir))[1]]
        reader = OrebaReader()
        for subject_id in subject_ids:
            logging.info("Working on subject {}".format(subject_id))
            if args.exp_dir == args.src_dir:
                exp_path = os.path.join(args.exp_dir, subject_id, subject_id + "_inert.csv")
            else:
                if not os.path.exists(args.exp_dir):
                    os.makedirs(args.exp_dir)
                exp_path = os.path.join(args.exp_dir, subject_id + ".csv")
            if os.path.isfile(exp_path):
                logging.info("Dataset file already exists. Skipping {0}.".format(subject_id))
                continue
            # Read acc and gyro
            logging.info("Reading raw data from Unisens")
            timestamps, left_acc, left_gyro, right_acc, right_gyro = \
                reader.read_inert(args.src_dir, subject_id)
            # Estimate initial quaternion
            logging.info("Estimating initial quaternion")
            left_init_q = estimate_initial_quaternion(left_acc, left_gyro, OREBA_FREQUENCY)
            right_init_q = estimate_initial_quaternion(right_acc, right_gyro, OREBA_FREQUENCY)
            # Remove gravity from acceleration vector
            logging.info("Removing gravity")
            left_acc_0 = remove_gravity(left_acc, left_gyro, left_init_q, OREBA_FREQUENCY, 16, args.vis)
            right_acc_0 = remove_gravity(right_acc, right_gyro, right_init_q, OREBA_FREQUENCY, 16, args.vis)
            # Standardize
            left_acc_0 = standardize(left_acc_0)
            left_gyro_0 = standardize(left_gyro)
            right_acc_0 = standardize(right_acc_0)
            right_gyro_0 = standardize(right_gyro)
            # Read annotations
            annotations = reader.read_annotations(args.src_dir, subject_id)
            label_1, label_2, label_3, label_4 = reader.get_labels(annotations, timestamps)
            dominant_hand = reader.read_dominant(args.src_dir, subject_id)
            # Write csv
            writer = OrebaWriter(exp_path)
            writer.write(subject_id, timestamps, left_acc, left_acc_0,
                left_gyro, left_gyro_0, right_acc, right_acc_0, right_gyro,
                right_gyro_0, dominant_hand, label_1, label_2, label_3, label_4)
        reader.done()

    elif args.reader == 'Clemson':
        # For Clemson Cafeteria data
        # Read subjects
        data_dir = os.path.join(args.src_dir, "all-data")
        subject_ids = [x for x in next(os.walk(data_dir))[1]]
        reader = ClemsonReader()
        for subject_id in subject_ids:
            subject_dir = os.path.join(data_dir, subject_id)
            sessions = [x for x in next(os.walk(subject_dir))[1]]
            for session in sessions:
                logging.info("Working on subject {}, session {}".format(subject_id, session))
                # Make sure export dir exists
                if not os.path.exists(args.exp_dir):
                    os.makedirs(args.exp_dir)
                exp_path = os.path.join(args.exp_dir, subject_id + "_" + session + ".csv")
                # Skip if export file already exists
                if os.path.isfile(exp_path):
                    logging.info("Dataset file already exists. Skipping {}_{}.".format(subject_id, session))
                    continue
                # Path of gesture annotations
                gesture_dir = os.path.join(args.src_dir, "all-gt-gestures", subject_id,
                    session, "gesture_union.txt")
                if not os.path.isfile(gesture_dir):
                    logging.warn("No gesture annotations found. Skipping {}_{}.".format(
                        subject_id, session))
                    continue
                # Path of bite annotations
                bite_dir = os.path.join(args.src_dir, "all-gt-bites", subject_id,
                    session, "gt_union.txt")
                if not os.path.isfile(bite_dir):
                    logging.warn("No bite annotations found. Skipping {}_{}.".format(
                        subject_id, session))
                # Read acc and gyro
                timestamps, acc, gyro = reader.read_inert(data_dir, subject_id, session)
                # Estimate initial quaternion
                logging.info("Estimating initial quaternion")
                init_q = estimate_initial_quaternion(acc, gyro, CLEMSON_FREQUENCY)
                # Remove gravity from acceleration vector
                logging.info("Removing gravity")
                acc_0 = remove_gravity(acc, gyro, init_q, CLEMSON_FREQUENCY, 15, args.vis)
                # Standardize
                acc_0 = standardize(acc_0)
                gyro_0 = standardize(gyro)
                # Read annotations
                annotations = reader.read_annotations(gesture_dir, bite_dir)
                label_1, label_2, label_3, label_4, label_5 = reader.get_labels(annotations, timestamps)
                # Write csv
                writer = ClemsonWriter(exp_path)
                writer.write(subject_id, session, timestamps, acc, acc_0, gyro,
                    gyro_0, label_1, label_2, label_3, label_4, label_5)

    elif args.reader == "FIC":
        # For Food Intake Cycle (FIC) dataset
        # Make sure HDF5 file exists
        pickle_path = os.path.join(args.src_dir, "fic_pickle.pkl")
        if not os.path.isfile(pickle_path):
            raise RunimeError('Pickle file not found')
        reader = FICReader()
        data = reader.read_pickle(pickle_path)
        for i in range(0, len(data['subject_id'])):
            subject_id = data['subject_id'][i]
            session_id = data['session_id'][i]
            metadata = data['metadata'][i]
            logging.info("Working on subject {}, session {}".format(subject_id, session_id))
            # Make sure export dir exists
            if not os.path.exists(args.exp_dir):
                os.makedirs(args.exp_dir)
            exp_path = os.path.join(args.exp_dir, str(subject_id) + "_" + str(session_id) + ".csv")
            # Skip if export file already exists
            if os.path.isfile(exp_path):
                logging.info("Dataset file already exists. Skipping {}_{}.".format(subject_id, session_id))
                continue
            # Read acc and gyro
            acc = data['raw_signals'][i]['accelerometer']
            gyro = data['raw_signals'][i]['gyroscope']
            # Determine start_time as first time available for both sensors
            start_time = np.max([acc[0,0], gyro[0,0]])
            end_time = np.min([acc[acc.shape[0]-1,0], gyro[gyro.shape[0]-1,0]])
            total_time = end_time - start_time
            # Number of samples after resampling
            if metadata['timestamps_raw_units'] == 'millis':
                factor = FACTOR_MILLIS
                calc_factor = 1000000
            else:
                factor = FACTOR_NANOS
                calc_factor = 1
            num = int(total_time / (1 / FIC_FREQUENCY * factor))
            # Trim and resample acc
            acc = acc[np.where(acc==start_time)[0][0]:np.where(acc==end_time)[0][0],1:4]
            acc_res = signal.resample(acc, num)
            # Trim and resample acc
            gyro = gyro[np.where(gyro==start_time)[0][0]:np.where(gyro==end_time)[0][0],1:4]
            gyro_res = signal.resample(gyro, num)
            # Derive evenly spaced timestamps
            dt = factor / FIC_FREQUENCY
            timestamps = np.arange(0, num*dt*calc_factor, int(dt*calc_factor))
            timestamps = np.array(timestamps / calc_factor)
            # Estimate initial quaternion
            logging.info("Estimating initial quaternion")
            init_q = estimate_initial_quaternion(acc, gyro, FIC_FREQUENCY)
            # Remove gravity from acceleration vector
            logging.info("Removing gravity")
            acc_0 = remove_gravity(acc_res, gyro_res, init_q, FIC_FREQUENCY, 16, args.vis)
            # Standardize
            acc_0 = standardize(acc_0)
            gyro_0 = standardize(gyro_res)
            # Read annotations
            annotations = data['bite_gt'][i]
            label_1 = reader.get_labels(annotations, timestamps)
            # Write csv
            writer = FICWriter(exp_path)
            writer.write(subject_id, session_id, timestamps, acc_res, acc_0,
                gyro_res, gyro_0, label_1)

    else:
        raise RunimeError('No valid reader selected')
    logging.info("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process inertial sensor data')
    parser.add_argument('--src_dir', type=str, default='', nargs='?', help='Directory to search for data.')
    parser.add_argument('--exp_dir', type=str, default='export', nargs='?', help='Directory for data export.')
    parser.add_argument('--vis', choices=('True','False'), default='False', nargs='?')
    parser.add_argument('--reader', choices=('Oreba', 'Clemson', 'FIC'), default='Oreba', nargs='?')
    args = parser.parse_args()
    main(args)
