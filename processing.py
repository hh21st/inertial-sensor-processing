import numpy as np
import quaternion
from visualization import PygameViewer
from fusion import MadgwickFusion
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
import oreba_dis
import clemson

FIC_FREQUENCY = 64
UPDATE_RATE = 16
FACTOR_MILLIS = 1000
FACTOR_MICROS = 1000000
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
        logging.info("Smoothing")
        def _up_to_odd_integer(f):
            return int(np.ceil(f) // 2 * 2 + 1)
        acc = smoothing(acc, mode=smoothing_mode,
            size=_up_to_odd_integer(smo_window_size), order=smoothing_order)
        gyro = smoothing(gyro, mode=smoothing_mode,
            size=_up_to_odd_integer(smo_window_size), order=smoothing_order)
    # 3, Apply standardization if enabled
    if use_standardization:
        logging.info("Standardizing")
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

def resample(acc, gyro, target_rate, time_factor, total_time):
    """Resample data using target frequency"""
    # Number of samples after resampling
    # Use nanoseconds during resampling
    calc_factor = FACTOR_NANOS / time_factor
    num = int(total_time / (1 / target_rate * time_factor))
    # Resample
    acc = signal.resample(acc, num)
    gyro = signal.resample(gyro, num)
    # Derive evenly spaced timestamps
    dt = time_factor / target_rate
    timestamps = np.arange(0, num*dt*calc_factor, int(dt*calc_factor))
    timestamps = np.array(timestamps / calc_factor)

    return timestamps, acc, gyro

def flip(acc, gyro, acc_signs, gyro_signs):
    """Flip left to right hand or right hand to left hand - position 1"""
    assert len(acc[0]) == 3 and len(gyro[0]) == 3, "Acc and Gyro must have 3 values"
    acc = np.multiply(acc, acc_signs)
    gyro = np.multiply(gyro, gyro_signs)
    return acc, gyro

def main(args=None):
    # Identify dataset
    if args.dataset == "OREBA":
        dataset = oreba_dis.Dataset(args.src_dir, args.exp_dir,
            args.dom_hand_spec, args.label_spec, args.label_spec_inherit,
            args.exp_uniform, args.exp_format)
    elif args.dataset == "FIC":
        dataset = fic.Dataset()
    elif args.dataset == "Clemson":
        dataset = clemson.Dataset(args.src_dir, args.exp_dir,
            args.dom_hand_spec, args.label_spec, args.label_spec_inherit,
            args.exp_uniform, args.exp_format)
    else:
        raise ValueError("Dataset {} not implemented!".format(args.dataset))

    # Session ids
    ids = dataset.ids()

    # Iterate all ids
    for id in ids:
        id_s = '_'.join(id) if isinstance(id, tuple) else id
        logging.info("Working on {}".format(id_s))

        if not dataset.check(id):
            continue

        # Output filename
        if args.exp_mode == 'dev':
            pp_s = "" +                                             \
                ("_grm" if args.use_gravity_removal else "") +      \
                ("_smo" if args.use_smoothing else "") +            \
                ("_std" if args.use_standardization else "") +      \
                ("_uni" if args.exp_uniform else "")
            exp_file = args.dataset + "_" + id_s + pp_s + "." + args.exp_format
        else:
            exp_file = id_s + "_inertial." + args.exp_format

        # Make exp_dir if it does not exist
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)

        # Export path
        exp_path = os.path.join(args.exp_dir, exp_file)
        if os.path.isfile(exp_path):
            logging.info("Dataset file already exists. Skipping {}.".format(id_s))
            #continue

        # Read timestamps and data
        timestamps, data = dataset.data(id)

        # Dominant hand
        dominant_hand = dataset.dominant(id)

        # If enabled, make hands uniform by flipping left to right if needed
        if args.exp_uniform and dominant_hand == 'left':
            logging.info("Flipping for uniformity")
            acc_signs, gyro_signs = dataset.get_flip_signs()
            if len(data) == 2:
                left, right = data["left"], data["right"]
                left_temp = (copy.deepcopy(left[0]), copy.deepcopy(left[1]))
                left = flip(right[0], right[1], acc_signs, gyro_signs)
                right = flip(left_temp[0], left_temp[1], acc_signs, gyro_signs)
                data["left"], data["right"] = left, right
            else:
                data["hand"] = (flip(data["hand"][0], data["hand"][1],
                    acc_signs, gyro_signs))

        # Decimate/resample if needed
        dataset_frequency = dataset.get_frequency()
        assert args.sampling_rate <= dataset_frequency, \
            "Desired sampling frequency cannot be higher than dataset frequency"
        if args.sampling_rate < dataset_frequency:
            if dataset_frequency % args.sampling_rate == 0:
                logging.info("Decimate")
                if len(data) == 2:
                    left, right = data["left"], data["right"]
                    timestamps, left_acc, left_gyro = decimate(left[0], left[1],
                        timestamps, args.sampling_rate, dataset_frequency)
                    _, right_acc, right_gyro = decimate(right[0], right[1],
                        timestamps, args.sampling_rate, dataset_frequency)
                    data["left"] = (left_acc, left_gyro)
                    data["right"] = (right_acc, right_gyro)
                else:
                    timestamps, acc, gyro = decimate(data[0], right[1],
                        timestamps, args.sampling_rate, dataset_frequency)
                    data["hand"] = (acc, gyro)
            else:
                logging.info("Resample")
                time_factor = dataset.get_time_factor()
                total_time = timestamps[len(timestamps)-1] - timestamps[0]
                if len(data) == 2:
                    left, right = data["left"], data["right"]
                    timestamps, left_acc, left_gyro = resample(left[0], left[1],
                        args.sampling_rate, time_factor, total_time)
                    _, right_acc, right_gyro = resample(right[0], right[1],
                        args.sampling_rate, time_factor, total_time)
                    data["left"] = (left_acc, left_gyro)
                    data["right"] = (right_acc, right_gyro)
                else:
                    timestamps, acc, gyro = resample(data["hand"][0], data["hand"][1],
                        args.sampling_rate, time_factor, total_time)
                    data["hand"] = (acc, gyro)

        # Processing
        if len(data) == 2:
            left, right = data["left"], data["right"]
            left_acc, left_gyro = preprocess(left[0], left[1],
                args.sampling_rate, args.smoothing_mode,
                args.smoothing_window_size, args.smoothing_order,
                args.use_vis, args.use_gravity_removal, args.use_smoothing,
                args.use_standardization)
            right_acc, right_gyro = preprocess(right[0], right[1],
                args.sampling_rate, args.smoothing_mode,
                args.smoothing_window_size, args.smoothing_order,
                args.use_vis, args.use_gravity_removal, args.use_smoothing,
                args.use_standardization)
            data["left"] = (left_acc, left_gyro)
            data["right"] = (right_acc, right_gyro)
        else:
            acc, gyro = preprocess(data["hand"][0], data["hand"][1],
                args.sampling_rate, args.smoothing_mode,
                args.smoothing_window_size, args.smoothing_order, args.use_vis,
                args.use_gravity_removal, args.use_smoothing,
                args.use_standardization)
            data["hand"] = (acc, gyro)

        # Read annotations
        labels = dataset.labels(id, timestamps)

        # Write data
        dataset.write(exp_path, id, timestamps, data, dominant_hand, labels)

    dataset.done()
    logging.info("Done")

def main1(args=None):
    """Preprocess data for the selected dataset."""

    if args.dataset == "FIC":
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

    elif args.dataset == 'Experiment':
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
    parser.add_argument('--src_dir', type=str, default='OREBA', nargs='?', help='Directory to search for data.')
    parser.add_argument('--exp_dir', type=str, default='Export', nargs='?', help='Directory for data export.')
    parser.add_argument('--dataset', choices=('OREBA', 'Clemson', 'FIC', 'Experiment'), default='OREBA', nargs='?', help='Which dataset reader/writer to use')
    parser.add_argument('--sampling_rate', type=int, default=64, nargs='?', help='Sampling rate of exported signals.')
    parser.add_argument('--use_vis', type=str2bool, default='False', nargs='?', help='Enable visualization')
    parser.add_argument('--use_gravity_removal', type=str2bool, default=True, help="Remove gravity during preprocessing?")
    parser.add_argument('--use_smoothing', type=str2bool, default=False, help="Apply smoothing during preprocessing?")
    parser.add_argument('--use_standardization', type=str2bool, default=True, help="Apply standardization during preprocessing?")
    parser.add_argument('--smoothing_window_size', type=int, default=1, nargs='?', help='Size of the smoothing window [number of frames].')
    parser.add_argument('--smoothing_order', type=int, default=1, nargs='?', help='The polynomial used in Savgol filter.')
    parser.add_argument('--smoothing_mode', type=str, choices=('medfilt', 'savgol_filter'), default='medfilt', nargs='?', help='smoothing mode')
    parser.add_argument('--exp_mode', type=str, choices=('dev', 'pub'), default='dev', nargs='?', help='Write file for publication or development')
    parser.add_argument('--exp_uniform', type=str2bool, default='True', nargs='?', help='Export uniform data by converting all dominant hands to right and all non-dominant hands to left')
    parser.add_argument('--exp_format', choices=('csv', 'tfrecord'), default='csv', nargs='?', help='Format for export')
    parser.add_argument('--label_spec', type=str, default='labels.xml', help='Filename of label specification')
    parser.add_argument('--label_spec_inherit', type=str2bool, default=True, help='Inherit label specification, e.g., if Serve not included, always keep sublabels as Idle')
    parser.add_argument('--dom_hand_spec', type=str, default='most_used_hand.csv' , nargs='?', help='the name of the file that contains the dominant hand info')
    parser.add_argument('--organise_data', type=str2bool, default='False' , nargs='?', help='Organise data in separate subfolders if true.')
    parser.add_argument('--des_dir', type=str, default='', nargs='?', help='Directory to copy train, val and test sets using data organiser.')
    parser.add_argument('--make_subfolders_val', type=str, default='False' , nargs='?', help='Create sub folder per each file in validation set if true.')
    parser.add_argument('--make_subfolders_test', type=str, default='False' , nargs='?', help='Create sub folder per each file in test set if true.')
    args = parser.parse_args()
    main(args)
