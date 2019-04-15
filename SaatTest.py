from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Input
from keras.models import Model
from keras.regularizers import *


def get_model():
	aliases = {}
	Input_1 = Input(shape=(1, 28, 28), name='Input_1')
	Flatten_1 = Flatten(name='Flatten_1')(Input_1)
	Dense_1 = Dense(name='Dense_1',output_dim= 512,activation= 'relu' )(Flatten_1)
	Dropout_1 = Dropout(name='Dropout_1',p= 0.3)(Dense_1)
	Dense_2 = Dense(name='Dense_2',output_dim= 512,activation= 'relu' )(Dropout_1)
	Dropout_2 = Dropout(name='Dropout_2',p= 0.3)(Dense_2)
	Dense_3 = Dense(name='Dense_3',output_dim= 10,activation= 'softmax' )(Dropout_2)

	model = Model([Input_1],[Dense_3])
	return aliases, model


from keras.optimizers import *

def get_optimizer():
	return Adadelta()

def is_custom_loss_function():
	return False

def get_loss_function():
	return 'categorical_crossentropy'

def get_batch_size():
	return 32

def get_num_epoch():
	return 10

def get_data_config():
	return '{"samples": {"test": 0, "training": 56000, "validation": 14000, "split": 1}, "shuffle": false, "datasetLoadOption": "full", "kfold": 1, "dataset": {"samples": 70000, "name": "mnist", "type": "public"}, "numPorts": 1, "mapping": {"Digit Label": {"port": "OutputPort0", "options": {}, "shape": "", "type": "Categorical"}, "Image": {"port": "InputPort0", "options": {"Width": 28, "horizontal_flip": false, "Scaling": 1, "Augmentation": false, "vertical_flip": false, "shear_range": 0, "pretrained": "None", "Resize": false, "height_shift_range": 0, "Normalization": false, "width_shift_range": 0, "Height": 28, "rotation_range": 0}, "shape": "", "type": "Image"}}}'