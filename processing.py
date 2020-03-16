import numpy as np
import quaternion
from visualization import PygameViewer
from fusion import MadgwickFusion
from reader import OrebaReader, ClemsonReader, FICReader, ExperimentReader
from writer import OrebaWriter, ClemsonWriter, FICWriter
import csv
import argparse
import os
import copy
import logging
from scipy import signal
from math import radians, degrees
import itertools
from data_organiser import DataOrganiser
from utils import *
import matplotlib.pyplot as plt

OREBA_FREQUENCY = 64
CLEMSON_FREQUENCY = 15
FIC_FREQUENCY = 64
UPDATE_RATE = 16
FACTOR_MILLIS = 1000
FACTOR_NANOS = 1000000000
GRAVITY = 9.80665
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S', level=logging.INFO)

def estimate_initial_quaternion(acc, gyro, data_freq, seconds=5, inc_deg=20):
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
        acc_0 = gravity_removal(acc[0:num], gyro[0:num], q, data_freq, 0, False)
        # If gravity is successfully removed, absolute values are low
        loss.append(np.sum(np.absolute(acc_0)))
    return qs[np.argmin(loss)]

def gravity_removal(acc, gyro, init_q, data_freq, update_freq, vis):
    """Remove gravity for one hand."""
    # Initialize
    madgwick = MadgwickFusion(init_q, data_freq)
    # Initialize visualization
    pv = None
    if vis == 'True':
        pv = PygameViewer(640, 480, init_q, data_freq)
    # Process
    acc_0 = []
    i = 0
    for acc_t, gyro_t in zip(acc, gyro):
        # Sensor fusion update
        madgwick.update(acc_t, gyro_t)
        if vis == 'True':
            pv.set_quaternion(madgwick.q)
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

def standardization(x):
    np_x = np.array(x)
    np_x -= np.mean(np_x, axis=0)
    np_x /= np.std(np_x, axis=0)
    return list(np_x)

def smoothing(x, mode, size, order):
    if size < 3 or size <= order:
        RuntimeError('Smoothing size {0} is too small.'.format(size))
        #return x;
    #if size <= order:
    #    order = size - 1
    if mode == 'savgol_filter':
        return signal.savgol_filter(x, size, order, axis=0)
    elif mode == 'medfilt':
        return signal.medfilt(x, size)
    else:
        raise RuntimeError('Smoothing mode {0} is not supported.'.format(mode))

def preprocess(acc, gyro, sampling_rate, smoothing_mode, smoothing_window_size,
    smoothing_order, use_vis, use_gravity_removal, use_smoothing,
    use_standardization):
    """Preprocess the data"""
    # 1. Remove gravity if enabled
    if use_gravity_removal:
        # 1.1 Estimate initial quaternion
        logging.info("Estimating initial quaternion")
        init_q = estimate_initial_quaternion(acc, gyro, sampling_rate)
        # 1.2 Remove gravity
        logging.info("Removing gravity")
        acc = gravity_removal(acc, gyro, init_q, sampling_rate, 16, use_vis)
    # 2. Apply smoothing if enabled
    if use_smoothing:
        def _up_to_odd_integer(f):
            return int(np.ceil(f) // 2 * 2 + 1)
        acc = smoothing(acc, mode=smoothing_mode,
            size=_up_to_odd_integer(smo_window_size), order=smoothing_order)
        gyro = smoothing(gyro, mode=smoothing_mode,
            size=_up_to_odd_integer(smo_window_size), order=smoothing_order)
    # 3, Apply standardization if enabled
    if use_standardization:
        acc = standardization(acc)
        gyro = standardization(gyro)

    return acc, gyro

def decimate(acc, gyro, timestamps, target_rate, original_rate):
    """Decimate the data"""
    if original_rate % target_rate == 0:
        factor = original_rate // target_rate
        if factor == 1:
            return timestamps, acc, gyro
        acc = signal.decimate(acc, factor, axis=0)
        gyro = signal.decimate(gyro, factor, axis=0)
        timestamps = timestamps[::factor]
        return timestamps, acc, gyro
    else:
        raise RuntimeError('Cannot decimate for this target rate')

def resample(acc, gyro, target_rate, units, total_time):
    """Resample data using target frequency"""
    # Number of samples after resampling
    if units == 'millis':
        factor = FACTOR_MILLIS
        calc_factor = 1000000
    else:
        factor = FACTOR_NANOS
        calc_factor = 1
    num = int(total_time / (1 / target_rate * factor))
    # Resample
    acc = signal.resample(acc, num)
    gyro = signal.resample(gyro, num)
    # Derive evenly spaced timestamps
    dt = factor / target_rate
    timestamps = np.arange(0, num*dt*calc_factor, int(dt*calc_factor))
    timestamps = np.array(timestamps / calc_factor)

    return timestamps, acc, gyro

def flip(acc, gyro):
    """Flip left to right hand or right hand to left hand - position 1"""
    acc = np.multiply(acc, [1, -1, 1])
    gyro = np.multiply(gyro, [-1, 1, -1])
    return acc, gyro

def flip2(acc, gyro):
    """Flip left to right hand or right hand to left hand - position 2"""
    acc = np.multiply(acc, [-1, 1, 1])
    gyro = np.multiply(gyro, [1, -1, -1])
    return acc, gyro

def main(args=None):
    """Preprocess data for the selected database."""

    if args.database == 'OREBA':
        # For OREBA data
        # Read subjects
        subject_ids = [x for x in next(os.walk(args.src_dir))[1]]
        reader = OrebaReader()
        for subject_id in subject_ids:
            # skip over faulty data file (1074_1)
            if subject_id == "1074_1":
                continue
            logging.info("Working on subject {}".format(subject_id))
            if args.exp_mode == 'dev':
                pp_s = "" +                                         \
                    ("_grm") if args.use_gravity_removal else "" +  \
                    ("_smo") if args.use_smoothing else "" +        \
                    ("_std") if args.use_standardization else "" +  \
                    ("_uni" if args.exp_uniform else "")
                exp_file = "OREBA_" + subject_id + pp_s + "." + args.exp_format
            else:
                exp_file = subject_id + "_inertial." + args.exp_format
            if args.exp_dir == args.src_dir:
                exp_path = os.path.join(args.exp_dir, subject_id, exp_file)
            else:
                if not os.path.exists(args.exp_dir):
                    os.makedirs(args.exp_dir)
                exp_path = os.path.join(args.exp_dir, exp_file)
            if os.path.isfile(exp_path):
                logging.info("Dataset file already exists. Skipping {0}.".format(subject_id))
                continue
            # Read acc and gyro
            logging.info("Reading raw data from Unisens")
            timestamps, left_acc, left_gyro, right_acc, right_gyro = \
                reader.read_inert(args.src_dir, subject_id)
            # Make hands uniform by flipping left to right if needed
            dominant_hand = reader.read_dominant(args.src_dir, subject_id, args.dom_hand_info_file_name)
            if args.exp_uniform == 'True' and dominant_hand == 'left':
                right_acc_temp = copy.deepcopy(right_acc)
                right_gyro_temp = copy.deepcopy(right_gyro)
                right_acc, right_gyro = flip(left_acc, left_gyro)
                left_acc, left_gyro = flip(right_acc_temp, right_gyro_temp)
            # Resample
            timestamps, left_acc, left_gyro = decimate(left_acc, left_gyro,
                timestamps, args.sampling_rate, OREBA_FREQUENCY)
            _, right_acc, right_gyro = decimate(right_acc, right_gyro,
                timestamps, args.sampling_rate, OREBA_FREQUENCY)
            # Preprocessing
            left_acc_0, left_gyro_0 = preprocess(left_acc, left_gyro,
                args.sampling_rate, args.smoothing_mode,
                args.smoothing_window_size, args.smoothing_order,
                args.use_vis, args.use_gravity_removal, args.use_smoothing,
                args.use_standardization)
            right_acc_0, right_gyro_0 = preprocess(right_acc, right_gyro,
                args.sampling_rate, args.smoothing_mode,
                args.smoothing_window_size, args.smoothing_order,
                args.use_vis, args.use_gravity_removal, args.use_smoothing,
                args.use_standardization)
            # Read annotations
            annotations = reader.read_annotations(args.src_dir, subject_id)
            label_1, label_2, label_3, label_4 = reader.get_labels(annotations, timestamps)
            # Write csv
            writer = OrebaWriter(exp_path)
            writer.write(subject_id, timestamps, left_acc_0, left_gyro_0,
                right_acc_0, right_gyro_0, dominant_hand,
                label_1, label_2, label_3, label_4, args.exp_uniform,
                args.exp_format)
        reader.done()

    elif args.database == 'Clemson':
        # For Clemson Cafeteria data
        # Read subjects
        data_dir = os.path.join(args.src_dir, "all-data")
        subject_ids = [x for x in next(os.walk(data_dir))[1]]
        reader = ClemsonReader()
        for subject_id in subject_ids:
            subject_dir = os.path.join(data_dir, subject_id)
            hand = reader.read_hand(args.src_dir, subject_id)
            sessions = [x for x in next(os.walk(subject_dir))[1]]
            for session in sessions:
                logging.info("Working on subject {}, session {}".format(subject_id, session))
                # Make sure export dir exists
                if not os.path.exists(args.exp_dir):
                    os.makedirs(args.exp_dir)
                # Make filename
                if args.exp_mode == 'dev':
                    pp_s = "" +                                             \
                        ("_grm") if args.use_gravity_removal else "" +      \
                        ("_smo") if args.use_smoothing else "" +            \
                        ("_std") if args.use_standardization else "" +      \
                        ("_uni" if args.exp_uniform else "")
                    exp_file = "Clemson_" + subject_id + "_" + session +    \
                        pp_s + "." + args.exp_format
                else:
                    exp_file = subject_id + "_" + session + "." + args.exp_format
                exp_path = os.path.join(args.exp_dir, exp_file)
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
                    continue
                # Read acc and gyro
                timestamps, acc, gyro = reader.read_inert(data_dir, subject_id, session)
                # Make hands uniform by flipping left to right if needed
                if args.exp_uniform == "True" and hand == 'left':
                    acc, gyro = flip(acc, gyro)
                # Preprocessing
                acc_0, gyro_0 = preprocess(acc, gyro, args.sampling_rate,
                    args.smoothing_mode, args.smoothing_window_size,
                    args.smoothing_order, args.use_vis, args.use_gravity_removal,
                    args.use_smoothing, args.use_standardization)
                # Read annotations
                annotations = reader.read_annotations(gesture_dir, bite_dir)
                label_1, label_2, label_3, label_4, label_5 = reader.get_labels(annotations, timestamps)
                # Write csv
                writer = ClemsonWriter(exp_path)
                writer.write(subject_id, session, timestamps, acc_0,
                    gyro_0, hand, label_1, label_2, label_3, label_4,
                    label_5, args.exp_format)

    elif args.database == "FIC":
        # For Food Intake Cycle (FIC) dataset
        # Make sure pickle file exists
        pickle_path = os.path.join(args.src_dir, "fic_pickle.pkl")
        if not os.path.isfile(pickle_path):
            raise RunimeError('Pickle file not found')
        reader = FICReader()
        data = reader.read_pickle(pickle_path)
        for i in range(0, len(data['subject_id'])):
            subject_id = data['subject_id'][i]
            session_id = data['session_id'][i]
            logging.info("Working on subject {}, session {}".format(subject_id, session_id))
            # Make sure export dir exists
            if not os.path.exists(args.exp_dir):
                os.makedirs(args.exp_dir)
            exp_file = "FIC_" + str(subject_id) + "_" + str(session_id) + \
                "." + args.exp_format
            exp_path = os.path.join(args.exp_dir, exp_file)
            # Skip if export file already exists
            if os.path.isfile(exp_path):
                logging.info("Dataset file already exists. Skipping {}_{}.".format(subject_id, session_id))
                continue
            # Read acc and gyro
            timestamps = data['signals'][i][:,0]
            acc = data['signals'][i][:,1:4]
            gyro = data['signals'][i][:,4:7]
            annotations = data['bite_gt'][i]
            label_1 = reader.get_labels(annotations, timestamps)
            # Standardize data
            acc = standardization(acc)
            gyro = standardization(gyro)
            # Write data
            writer = FICWriter(exp_path)
            writer.write(subject_id, session_id, timestamps, acc, gyro,
                label_1, args.exp_format)

    elif args.database == 'Experiment':
        reader = ExperimentReader()
        timestamps, left_acc, left_gyro, right_acc, right_gyro = \
            reader.read_inert(args.src_dir, "Exp2", "02363", "02366")
        left_acc, right_acc = np.array(left_acc), np.array(right_acc)
        left_gyro, right_gyro = np.array(left_gyro), np.array(right_gyro)
        right_acc, right_gyro = flip(right_acc, right_gyro)
        #plt.plot(left_acc[:,2])
        #plt.plot(right_acc[:,2])
        plt.plot(left_gyro[:,2])
        plt.plot(right_gyro[:,2])
        plt.show()
        exit()

    else: raise RuntimeError('No valid reader selected')

    if args.organise_data:
        DataOrganiser.organise(src_dir=args.exp_dir, des_dir=args.des_dir,
            make_subfolders_val=get_bool(args.make_subfolders_val),
            make_subfolders_test=get_bool(args.make_subfolders_test))

    logging.info("Done")

def str2bool(v):
    """Boolean type for argparse"""
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process inertial sensor data')
    parser.add_argument('--src_dir', type=str, default=r'C:\H\PhD\ORIBA\Model\FileGen\OREBA\temp', nargs='?', help='Directory to search for data.')
    parser.add_argument('--exp_dir', type=str, default=r'C:\H\PhD\ORIBA\Model\FileGen\OREBA\temp.gen\64_grm_my_dec_std_uni', nargs='?', help='Directory for data export.')
    parser.add_argument('--database', choices=('OREBA', 'Clemson', 'FIC', 'Experiment'), default='OREBA', nargs='?', help='Which database reader/writer to use')
    parser.add_argument('--sampling_rate', type=int, default=64, nargs='?', help='Sampling rate of exported signals.')
    parser.add_argument('--use_vis', type=str2bool, default='False', nargs='?', help='Enable visualization')
    parser.add_argument('--use_gravity_removal', type=str2bool, default=True, help="Remove gravity during preprocessing?")
    parser.add_argument('--use_smoothing', type=str2bool, default=False, help="Apply smoothing during preprocessing?")
    parser.add_argument('--use_standardization', type=str2bool, default=True, help="Apply standardization during preprocessing?")
    parser.add_argument('--smoothing_window_size', type=int, default=1, nargs='?', help='Size of the smoothing window [number of frames].')
    parser.add_argument('--smoothing_order', type=int, default=1, nargs='?', help='The polynomial used in Savgol filter.')
    parser.add_argument('--smoothing_mode', type=str, choices=('medfilt', 'savgol_filter'), default='medfilt', nargs='?', help='smoothing mode')
    parser.add_argument('--exp_mode', type=str, choices=('dev', 'pub'), default='dev', nargs='?', help='Write file for publication or development')
    parser.add_argument('--exp_uniform', type=str, choices=('True', 'False'), default='True', nargs='?', help='Export uniform data by converting all dominant hands to right and all non-dominant hands to left')
    parser.add_argument('--exp_format', choices=('csv', 'tfrecord'), default='csv', nargs='?', help='Format for export')
    parser.add_argument('--organise_data', type=str2bool, default='False' , nargs='?', help='Organise data in separate subfolders if true.')
    parser.add_argument('--des_dir', type=str, default='', nargs='?', help='Directory to copy train, val and test sets using data organiser.')
    parser.add_argument('--make_subfolders_val', type=str, default='False' , nargs='?', help='Create sub folder per each file in validation set if true.')
    parser.add_argument('--make_subfolders_test', type=str, default='False' , nargs='?', help='Create sub folder per each file in test set if true.')
    parser.add_argument('--dom_hand_info_file_name', type=str, default='most_used_hand.csv' , nargs='?', help='the name of the file that contains the dominant hand info')
    args = parser.parse_args()
    main(args)
