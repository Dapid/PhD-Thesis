import os

from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Dropout
from keras.layers.merge import concatenate, add
from keras.regularizers import l2
from keras.layers.advanced_activations import ELU
from keras import backend as K
from keras import callbacks

ACTIVATION = ELU
INIT = 'he_normal'
DROPOUT = 0.1

bn_opts = dict(axis=2)


def add_1d_conv(input_layer, length, nb_filt, REG):
    layer = Conv1D(nb_filt, length, padding='same', kernel_initializer=INIT,
                   kernel_regularizer=l2(REG))(input_layer)
    layer = ACTIVATION()(layer)
    layer = BatchNormalization(**bn_opts)(layer)
    layer = Dropout(0.1)(layer)

    return layer


def add_1d_convresnet(input_layer, length, nb_filt, REG):
    """Following resnet v4, with dropout"""
    tower_1 = Conv1D(nb_filt, length, padding='same', kernel_initializer=INIT, kernel_regularizer=l2(REG))(input_layer)
    tower_1 = ACTIVATION()(tower_1)
    tower_1 = BatchNormalization(**bn_opts)(tower_1)
    tower_1 = Dropout(0.1)(tower_1)

    tower_1 = Conv1D(nb_filt, length, padding='same', kernel_initializer=INIT, kernel_regularizer=l2(REG))(tower_1)
    tower_1 = ACTIVATION()(tower_1)
    tower_1 = BatchNormalization(**bn_opts)(tower_1)
    tower_1 = Dropout(0.1)(tower_1)

    shortcut = Conv1D(nb_filt, length, padding='same', kernel_initializer=INIT, kernel_regularizer=l2(REG))(input_layer)
    shortcut = ACTIVATION()(shortcut)
    shortcut = BatchNormalization(**bn_opts)(shortcut)
    shortcut = Dropout(0.1)(shortcut)
    x = add([tower_1, shortcut])

    return x


def add_1d_resnet(input_layer, length, nb_filt, REG):
    """Following resnet, with dropout"""
    tower_1 = Conv1D(nb_filt, length, padding='same', kernel_initializer=INIT, kernel_regularizer=l2(REG))(input_layer)
    tower_1 = ACTIVATION()(tower_1)
    tower_1 = BatchNormalization(**bn_opts)(tower_1)
    tower_1 = Dropout(0.1)(tower_1)

    tower_1 = Conv1D(nb_filt, length, padding='same', kernel_initializer=INIT, kernel_regularizer=l2(REG))(tower_1)
    tower_1 = ACTIVATION()(tower_1)
    tower_1 = BatchNormalization(**bn_opts)(tower_1)
    tower_1 = Dropout(0.1)(tower_1)

    x = add([tower_1, input_layer])
    return x


def _build_feature_model(REG, total_depth, width):
    input_seq = Input(shape=(None, 22))
    input_self_info = Input(shape=(None, 23))
    input_part_ent = Input(shape=(None, 23))
    inputs = [input_seq, input_self_info, input_part_ent]

    _seq_inner_filters = width

    seq = []
    for input_ in inputs:
        layer = add_1d_conv(input_, 1, 16, REG)
        layer = add_1d_convresnet(layer, 3, _seq_inner_filters // 2, REG)
        for _ in range(total_depth // 2 - 1):
            layer = add_1d_resnet(layer, 3, _seq_inner_filters // 2, REG)
        seq.append(layer)

    merged = concatenate(seq, axis=-1)
    merged = add_1d_convresnet(merged, 3, _seq_inner_filters, REG)
    for _ in range(total_depth // 2 - 1):
        merged = add_1d_resnet(merged, 3, _seq_inner_filters, REG)

    feature_extraction_model = Model(inputs=inputs,
                                     outputs=merged)
    return feature_extraction_model


def build_model_joint(REG, depth, width):
    input_seq = Input(shape=(None, 22))
    input_self_info = Input(shape=(None, 23))
    input_part_ent = Input(shape=(None, 23))

    features_model = _build_feature_model(REG, depth, width)

    features = features_model([input_seq, input_self_info, input_part_ent])
    # Include two FC layers:
    features = add_1d_conv(features, 1, 512, REG)
    final_window = 1

    output_dihedrals_sc = Conv1D(4, final_window, padding='same', activation='tanh',
                                 kernel_initializer=INIT, name='dihedrals')(features)

    model = Model(inputs=[input_seq, input_self_info, input_part_ent],
                  outputs=[output_dihedrals_sc])
    model.compile('adam', 'mse', metrics=['mae'], sample_weight_mode='temporal')
    NAME = 'dh_multitask_dropout_01_l2_{}_depth_{}_width_{}'.format(REG, depth, width)
    model.name = NAME

    return model


def build_model_single(REG, depth, width, mode):
    assert mode in ('phi', 'psi')
    input_seq = Input(shape=(None, 22))
    input_self_info = Input(shape=(None, 23))
    input_part_ent = Input(shape=(None, 23))

    features_model = _build_feature_model(REG, depth, width)

    features = features_model([input_seq, input_self_info, input_part_ent])
    # Include two FC layers:
    features = add_1d_conv(features, 1, 512, REG)
    final_window = 1

    output_dihedrals_sc = Conv1D(2, final_window, padding='same', activation='tanh',
                                 kernel_initializer=INIT, name=mode)(features)

    model = Model(inputs=[input_seq, input_self_info, input_part_ent],
                  outputs=[output_dihedrals_sc])
    model.compile('adam', 'mse', metrics=['mae'], sample_weight_mode='temporal')
    NAME = 'dh_single_{}_dropout_01_l2_{}_depth_{}_width_{}'.format(mode, REG, depth, width)
    model.name = NAME

    return model


def get_callbacks(NAME):
    cbacks = [
            callbacks.ReduceLROnPlateau(factor=0.5, patience=30, min_lr=1e-9, verbose=True),
    ]
    depth = NAME.split('_depth_')[1].split('_')[0]
    if K._backend == 'tensorflow':
        logdir = './logs/{}'.format(depth)
        os.system('mkdir -p {}'.format(logdir))
        cbacks.append(callbacks.TensorBoard(log_dir=logdir, histogram_freq=0, write_images=True,
                                            batch_size=1, write_graph=True))
    return cbacks
