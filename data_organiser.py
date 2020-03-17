import argparse
import os
import shutil
import logging

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S', level=logging.INFO)

class DataOrganiser:

    def __init__(self, src_dir, organise_dir, organise_subfolders):
        self.src_dir = src_dir
        self.organise_dir = organise_dir
        self.organise_subfolders = organise_subfolders

    def organise(self, train_ids, valid_ids, test_ids):

        train_dir = os.path.join(self.organise_dir, "train")
        valid_dir = os.path.join(self.organise_dir, "valid")
        test_dir = os.path.join(self.organise_dir, "test")

        all_files = os.listdir(self.src_dir)
        train_files = [f for f in all_files if any("_" + id in f for id in train_ids)]
        valid_files = [f for f in all_files if any("_" + id in f for id in valid_ids)]
        test_files = [f for f in all_files if any("_" + id in f for id in test_ids)]

        assert len(list(set(train_files) & set(valid_files))) == 0, \
            "Overlap between train and valid"
        assert len(list(set(train_files) & set(test_files))) == 0, \
            "Overlap between train and test"
        assert len(list(set(valid_files) & set(test_files))) == 0, \
            "Overlap between valid and test"

        def copy_to_dir(file, origin, dest):
            if not os.path.exists(dest):
                os.makedirs(dest)
            origin_file = os.path.join(origin, file)
            if os.path.isfile(origin_file):
                shutil.copy(origin_file, dest)
            else:
                raise RuntimeError('File {} does not exist'.format(origin_file))

        for file in train_files:
            copy_to_dir(file, self.src_dir, train_dir)

        for file in valid_files:
            copy_to_dir(file, self.src_dir, valid_dir)
            if self.organise_subfolders:
                subdir = os.path.join(valid_dir + "_sub", file)
                copy_to_dir(file, self.src_dir, subdir)

        for file in test_files:
            copy_to_dir(file, self.src_dir, test_dir)
            if self.organise_subfolders:
                subdir = os.path.join(test_dir + "_sub", file)
                copy_to_dir(file, self.src_dir, subdir)

        logging.info("Done organising")
