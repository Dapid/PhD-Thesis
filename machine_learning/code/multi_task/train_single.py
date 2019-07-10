import random
import itertools

import numpy as np
import tables

import model_builder

class DataGenerator(object):
    def __init__(self, data_path, trim=None, mode='all', angle=None):
        assert angle in ('phi', 'psi')
        self.angle = angle
        self.h5 = tables.open_file(data_path)
        if mode == 'train':
            self.g = [x.decode() for x in self.h5.root.index_train[:]]
        elif mode == 'test':
            self.g = [x.decode() for x in self.h5.root.index_test[:]]
        elif mode == 'all':
            self.g = [x.decode() for x in self.h5.root.index[:]]
        else:
            raise ValueError('Unkown mode ' + str(mode))

        self._g = self.g
        if trim is not None:
            random.shuffle(self.g)
            self.g = itertools.cycle(self.g[:trim])
            self.trimmed = True
            self._trim = trim
        else:
            self.trimmed = False
            if mode == 'test':
                self.g = itertools.cycle(self.g)

    def __iter__(self):
        return self

    def read_group(self, g):
        valid = g.valid[:].astype(np.float32)
        part_ent = g.part_entr[:]
        self_info = g.self_info[:]
        seq = g.seq[:]
        if self.angle == 'phi':
            angle = g.dihedrals_sc[:][:, :, :2]
        elif self.angle == 'psi':
            angle = g.dihedrals_sc[:][:, :, 2:]
        else:
            raise RuntimeError(self.angle)
        outputs = [angle]
        inputs = [seq, self_info, part_ent]
        return inputs, outputs, [valid] * len(outputs)

    def __next__(self):
        while True:
            if isinstance(self.g, itertools.cycle):
                g = self.h5.get_node(next(self.g))
            else:
                g = self.h5.get_node(random.choice(self.g))
            inputs, outputs, valids = self.read_group(g)
            N = inputs[0].shape[1]
            if any(arr.shape[1] != N for arr in inputs + outputs):
                print('Problems with group', g)
                continue
            return inputs, outputs, valids

    def __len__(self):
        if self.trimmed:
            return self._trim
        if isinstance(self.g, itertools.cycle):
            return len(self._g)
        return len(self.g)

if __name__ == '__main__':
    for angle in ('phi', 'psi'):
        data_file = '/home/david/research/PQ4/ss_pred/data/data_jhE3.ur50.v2.h5'
        data_file_1 = '/home/david/research/PQ4/ss_pred/data/data_jhE3.ur50.v2_1.h5'
        data_file_2 = '/home/david/research/PQ4/ss_pred/data/data_jhE3.ur50.v2_2.h5'
        testing_data = DataGenerator(data_file_1, mode='test', angle=angle)
        training_data = DataGenerator(data_file_2, mode='train', angle=angle)
        x, y, v = next(training_data)
        print([i.shape for i in x])
        print([i.shape for i in y])
        print([i.shape for i in v])
        next(testing_data)
        print('Loaded', len(training_data), 'training samples and', len(testing_data), 'testing samples.')

        depth = 8
        width = 64
        L2_REG = 1e-9
        angle = 'phi'

        model = model_builder.build_model_single(L2_REG, depth, width, angle)
        prefix = model.name
        history = model.fit_generator(training_data, callbacks=model_builder.get_callbacks(prefix),
                                      steps_per_epoch=1000, epochs=200,
                                      validation_data=testing_data, validation_steps=len(testing_data),
                                      use_multiprocessing=True, max_queue_size=10, workers=1)
        model.save(prefix + '_saved_model.h5')
        model.save_weights(prefix + '_saved_weights.h5')

