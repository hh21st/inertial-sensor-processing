import numpy as np
import quaternion
from visualization import PygameViewer
from fusion import MadgwickFusion
from reader import UnisensReader
from writer import TwoHandsWriter
import csv
import argparse
import os
import logging

LOOP_RATE = 64
UPDATE_RATE = 16
DEFAULT_LABEL = "Idle"
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
    def __init__(self):
        self.nodes = []
        self.faces = []
        self.q = np.quaternion(1, 0, 0, 0) # Initial pose estimate

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

def initialize_cuboid():
    """Initialize cuboid with nodes and faces"""

    # The cuboid
    cuboid = Cuboid()

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

def remove_gravity(acc, gyro, vis):
    """Remove gravity for one hand."""
    logging.info("Removing gravity")
    # Initialize
    cuboid = initialize_cuboid()
    madgwick = MadgwickFusion(cuboid.q, LOOP_RATE)
    # Initialize visualization
    pv = None
    if vis == 'True':
        pv = PygameViewer(640, 480, cuboid, UPDATE_RATE)
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
            if i % (LOOP_RATE//UPDATE_RATE) == 0:
                if not pv.update():
                    break
        i += 1
    return acc_0

def get_labels(annotations, timestamps):
    """Infer labels from annotations and timestamps"""
    num = len(timestamps)
    labels_1 = np.empty(num, dtype='U25'); labels_1.fill(DEFAULT_LABEL)
    labels_2 = np.empty(num, dtype='U25'); labels_2.fill(DEFAULT_LABEL)
    labels_3 = np.empty(num, dtype='U25'); labels_3.fill(DEFAULT_LABEL)
    labels_4 = np.empty(num, dtype='U25'); labels_4.fill(DEFAULT_LABEL)
    for start_time, end_time, label_1, label_2, label_3, label_4 in zip(*annotations):
        start_frame = np.argmax(np.array(timestamps) >= start_time)
        end_frame = np.argmax(np.array(timestamps) > end_time)
        labels_1[start_frame:end_frame] = label_1
        labels_2[start_frame:end_frame] = label_2
        labels_3[start_frame:end_frame] = label_3
        labels_4[start_frame:end_frame] = label_4
    return list(labels_1), list(labels_2), list(labels_3), list(labels_4)

def main(args=None):
    """Main"""
    # For Unisens data
    if args.reader == 'Unisens':
        # Read subjects
        subject_ids = [x for x in next(os.walk(args.src_dir))[1]]
        reader = UnisensReader()
        for subject_id in subject_ids:
            logging.info("Working on subject {}".format(subject_id))
            # Read acc and gyro
            logging.info("Reading raw data from Unisens")
            timestamps, left_acc, left_gyro, right_acc, right_gyro = \
                reader.read_inert(args.src_dir, subject_id)
            # Remove gravity from acceleration vector
            left_acc_0 = remove_gravity(left_acc, left_gyro, args.vis)
            right_acc_0 = remove_gravity(right_acc, right_gyro, args.vis)
            # Read annotations
            annotations = reader.read_annotations(args.src_dir, subject_id)
            label_1, label_2, label_3, label_4 = get_labels(annotations, timestamps)
            dominant_hand = reader.read_dominant(args.src_dir, subject_id)
            # Write csv

            if args.exp_dir == args.src_dir:
                exp_path = os.path.join(args.exp_dir, subject_id, subject_id + "_inert.csv")
            else:
                if not os.path.exists(args.exp_dir):
                    os.makedirs(args.exp_dir)
                exp_path = os.path.join(args.exp_dir, subject_id + ".csv")
            writer = TwoHandsWriter(exp_path)
            writer.write(subject_id, timestamps, left_acc, left_acc_0,
                left_gyro, right_acc, right_acc_0, right_gyro, dominant_hand,
                label_1, label_2, label_3, label_4)
        reader.done()

    else:
        raise RunimeError('No valid reader selected')

    # TODO: Get dominant hand

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process inertial sensor data')
    parser.add_argument('--src_dir', type=str, default='', nargs='?', help='Directory to search for data.')
    parser.add_argument('--exp_dir', type=str, default='export', nargs='?', help='Directory for data export.')
    parser.add_argument('--vis', choices=('True','False'), default='False', nargs='?')
    parser.add_argument('--reader', choices=('Unisens', 'Clemson', 'FIC'), default='Unisens', nargs='?')
    args = parser.parse_args()
    main(args)
