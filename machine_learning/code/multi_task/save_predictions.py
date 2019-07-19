import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import tables
import tqdm

from keras.models import load_model
from keras import backend as K


class DataGenerator(object):
    def __init__(self, data_path):
        self.h5 = tables.open_file(data_path)

        self.g = [x.decode() for x in self.h5.root.index_test[:]]
        self._i = 0

    def __iter__(self):
        return self

    def read_group(self, g):
        valid = g.valid[:].astype(np.float32)
        part_ent = g.part_entr[:]
        self_info = g.self_info[:]
        seq = g.seq[:]
        outputs = [g.dihedrals_sc[:]]
        inputs = [seq, self_info, part_ent]
        return inputs, outputs, [valid] * len(outputs)

    def __next__(self):
        if self._i >= len(self.g):
            self._i = 0
            raise StopIteration

        name = self.g[self._i]
        g = self.h5.get_node(name)
        self._i += 1

        return self.read_group(g)

    def __len__(self):
        return len(self.g)


if __name__ == '__main__':
    data_file_1 = '/home/david/research/PQ4/ss_pred/data/data_jhE3.ur50.v2_1.h5'
    testing_data = DataGenerator(data_file_1)

    double_model = load_model('dh_multitask_dropout_01_l2_1e-09_depth_8_width_64_saved_model.h5')
    phi_coll = []
    psi_coll = []

    phi_true = []
    psi_true = []

    valid = []

    for x, y, v in tqdm.tqdm(testing_data):
        pred = double_model.predict(x).squeeze()
        phi = np.arctan2(pred[:, 0], pred[:, 1])
        psi = np.arctan2(pred[:, 2], pred[:, 3])
        phi_coll.append(phi)
        psi_coll.append(psi)

        pred = y[0].squeeze()
        phi = np.arctan2(pred[:, 0], pred[:, 1])
        psi = np.arctan2(pred[:, 2], pred[:, 3])
        phi_true.append(phi)
        psi_true.append(psi)

        valid.append(np.convolve(v[0].squeeze(), [1, 1, 1], mode='same') == 3)

    np.savez('combined', phi_pred=np.concatenate(phi_coll), psi_pred=np.concatenate(psi_coll),
             phi_true=np.concatenate(phi_true), psi_true=np.concatenate(psi_true), valid=np.concatenate(valid))
    K.clear_session()

    phi_model = load_model('dh_single_phi_dropout_01_l2_1e-09_depth_8_width_64_saved_model.h5')
    phi_coll = []
    psi_coll = []

    valid = []
    for x, y, v in tqdm.tqdm(testing_data):
        pred = phi_model.predict(x).squeeze()
        phi = np.arctan2(pred[:, 0], pred[:, 1])
        phi_coll.append(phi)
        valid.append(np.convolve(v[0].squeeze(), [1, 1, 1], mode='same') == 3)
    K.clear_session()

    psi_model = load_model('dh_single_psi_dropout_01_l2_1e-09_depth_8_width_64_saved_model.h5')
    for x, y, v in tqdm.tqdm(testing_data):
        pred = psi_model.predict(x).squeeze()
        psi = np.arctan2(pred[:, 0], pred[:, 1])
        psi_coll.append(psi)

    np.savez('separated', phi_pred=np.concatenate(phi_coll), psi_pred=np.concatenate(psi_coll),
             valid=np.concatenate(valid))