import argparse
import os
import numpy as np
import logging
from reader import OrebaReader, ClemsonReader, FICReader
from writer import OrebaWriter, ClemsonWriter, FICWriter

def main(args=None):
    """Main"""
    if args.database == "OREBA":
        subject_ids = [x for x in next(os.walk(args.src_dir))[1]]
        reader = OrebaReader()
        n_gestures, t_gestures, t_total = [], [], []
        for subject_id in subject_ids:
            timestamps, _, _, _, _ = reader.read_inert(args.src_dir, subject_id)
            annotations = np.array(reader.read_annotations(args.src_dir, subject_id))
            n_gestures.append(annotations.shape[1])
            t_gestures.append(np.sum(annotations[1,:].astype(np.int) -
                annotations[0,:].astype(np.int))/1000/1000)
            t_total.append((timestamps[len(timestamps)-1] - timestamps[0])/1000/1000)
        exp_path = os.path.join(args.exp_dir, "OREBA_summary.csv")
        writer = OrebaWriter(exp_path)
        writer.write_summary(subject_ids, n_gestures, t_gestures, t_total)

    elif args.database == "Clemson":
        data_dir = os.path.join(args.src_dir, "all-data")
        subject_ids = [x for x in next(os.walk(data_dir))[1]]
        reader = ClemsonReader()
        ids, n_gestures, t_gestures, t_total = [], [], [], []
        for subject_id in subject_ids:
            subject_dir = os.path.join(data_dir, subject_id)
            sessions = [x for x in next(os.walk(subject_dir))[1]]
            for session in sessions:
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
                # Read timestamps
                timestamps, _, _ = reader.read_inert(data_dir, subject_id, session)
                # Read annotations
                annotations = np.array(reader.read_annotations(gesture_dir, bite_dir))
                # Filter bite and drink
                annotations = annotations[:,np.isin(annotations[2,:], ['bite', 'drink'])]
                # Derive summaries
                ids.append(subject_id + "_" + session)
                n_gestures.append(annotations.shape[1])
                t_gestures.append(np.sum(annotations[1,:].astype(np.int) -
                    annotations[0,:].astype(np.int))/1000/1000)
                t_total.append((timestamps[len(timestamps)-1] - timestamps[0])/1000/1000)
        exp_path = os.path.join(args.exp_dir, "Clemson_summary.csv")
        writer = ClemsonWriter(exp_path)
        writer.write_summary(ids, n_gestures, t_gestures, t_total)

    elif args.database == "FIC":
        reader = FICReader()
        pickle_path = os.path.join(args.src_dir, "fic_pickle.pkl")
        data = reader.read_pickle(pickle_path)
        ids, n_gestures, t_gestures, t_total = [], [], [], []
        for i in range(0, len(data['subject_id'])):
            subject_id = data['subject_id'][i]
            session_id = data['session_id'][i]
            annotations = data['bite_gt'][i]
            metadata = data['metadata'][i]
            acc = data['raw_signals'][i]['accelerometer']
            gyro = data['raw_signals'][i]['gyroscope']
            start_time = np.max([acc[0,0], gyro[0,0]])
            end_time = np.min([acc[acc.shape[0]-1,0], gyro[gyro.shape[0]-1,0]])
            total_time = end_time - start_time
            if metadata['timestamps_raw_units'] == 'nanos':
                total_time /= 1000000000
            else:
                total_time /= 1000
            # Derive summaries
            ids.append(str(subject_id) + "_" + str(session_id))
            n_gestures.append(annotations.shape[0])
            t_gestures.append(np.sum(annotations[:,1].astype(np.int) -
                annotations[:,0].astype(np.int))/1000)
            t_total.append(total_time)
        exp_path = os.path.join(args.exp_dir, "FIC_summary.csv")
        writer = FICWriter(exp_path)
        writer.write_summary(ids, n_gestures, t_gestures, t_total)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summary of inertial sensor data')
    parser.add_argument('--src_dir', type=str, default='', nargs='?', help='Directory to search for data.')
    parser.add_argument('--exp_dir', type=str, default='export', nargs='?', help='Directory for data export.')
    parser.add_argument('--database', choices=('OREBA', 'Clemson', 'FIC'), default='OREBA', nargs='?', help='Which database reader/writer to use')
    args = parser.parse_args()
    main(args)
