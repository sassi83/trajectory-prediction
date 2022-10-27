import os
import sys
sys.path.append("model/")
import argparse

parser = argparse.ArgumentParser(description='Training Aura Model')
parser.add_argument('--model', type=int, default=0, help='Model type: '
                                                         '0=aura_model_gaus_non_para, '
                                                         '1=aura_model_gaus_para, '
                                                         '2=aura_model_gum_non_para, '
                                                         '3=aura_model_gum_para'
                                                         '4=aura_model_gmm_non_para, '
                                                         '5=aura_model_gmm_para')
parser.add_argument('--agents', type=int, default=10, help='This is the number of predictions for each agent')
parser.add_argument('--output_dim', type=int, default=2, help='This is the size of the output variable')
parser.add_argument('--intput_dim', type=int, default=2, help='This is the size of the input variable')
parser.add_argument('--history', type=int, default=10, help='This is the size of the past trajecktory')
parser.add_argument('--future', type=int, default=150, help='This is the size of the predicted trajecktory')
parser.add_argument('--image_size', type=int, default=100, help='This is the size of the image row/column')
parser.add_argument('--latent_dim', type=int, default=10, help='This is the size of the latent variable')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
parser.add_argument('--gpu', type=int, default=0, help='Select a gpu')
parser.add_argument('--weights', type=bool, default=False, help='load old weights if true')
parser.add_argument('--components', type=int, default=10, help='# of components')
parser.add_argument('--test_cv_model', type=bool, default=True, help='test_cv_model if true')

args = parser.parse_args(sys.argv[1:])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

model_name_list = ['aura_model_gaus_non_para',
                   'aura_model_gaus_para',
                   'aura_model_gum_non_para',
                   'aura_model_gum_para',
                   'aura_model_gmm_non_para',
                   'aura_model_gmm_para']

model_name = model_name_list[args.model]


import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

import aura_model_vit as model_lib



filepath = "saved_models/checkpoints/" + str(model_name) + "/"
frozen_dir = "saved_models/frozen_models/" + str(model_name) + "/"



#######################################################   build model  #####################################################
# Instantiate the model
model_train = model_lib.get_model(obs_seq=args.history, pred_seq=args.future, number_of_outputs=args.output_dim, latent_dim=args.latent_dim, training=True, type=args.model, row=args.image_size, column=args.image_size)
model_train.summary()


if args.weights == True:
    # load weights
    model_train.load_weights(filepath)
    print('best weights loaded')

#### model infer  ######
model = model_lib.get_model(obs_seq=args.history, pred_seq=args.future, number_of_outputs=args.output_dim, latent_dim=args.latent_dim, training=False, type=args.model, row=args.image_size, column=args.image_size)
model.compile(loss='mse', optimizer='adam')
model.summary()

if args.weights == True:
    # copy weights
    model.get_layer('encoder_x').set_weights(model_train.get_layer('encoder_x_train').get_weights())
    model.get_layer('dense_z').set_weights(model_train.get_layer('dense_z').get_weights())
    model.get_layer('decoder_dense_0').set_weights(model_train.get_layer('decoder_dense_0').get_weights())
    model.get_layer('decoder_lstm_0').set_weights(model_train.get_layer('decoder_lstm_0').get_weights())
    model.get_layer('decoder_time_dist_0').set_weights(model_train.get_layer('decoder_time_dist_0').get_weights())
    if args.model == 1 or args.model == 3 or args.model == 5:
        model.get_layer('decoder_time_dist_1').set_weights(model_train.get_layer('decoder_time_dist_1').get_weights())
        model.get_layer('decoder_time_dist_2').set_weights(model_train.get_layer('decoder_time_dist_2').get_weights())

    if args.model == 4 or args.model == 5:
        model.get_layer('z_prior_mean').set_weights(model_train.get_layer('z_prior_mean').get_weights())
        model.get_layer('z_prior_sig').set_weights(model_train.get_layer('z_prior_sig').get_weights())

    print('weights copied')





# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda map, tra, z: model((map, tra, z)))
full_model = full_model.get_concrete_function(
    map=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype),
    tra=tf.TensorSpec(model.inputs[1].shape, model.inputs[1].dtype),
    z=tf.TensorSpec(model.inputs[2].shape, model.inputs[2].dtype)
)

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
layers = [op.name for op in frozen_func.graph.get_operations()]

###################  print graph ##############################


print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)


###################  save graph ##############################
# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_dir,
                  name=model_name + ".pb",
                  as_text=False)

print("====== Frozen Model created ===============")
# Load frozen graph using TensorFlow 1.x functions
with tf.io.gfile.GFile(frozen_dir + model_name + ".pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(f.read())


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    if print_graph == True:
        #print("-" * 50)
        #print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

# Wrap frozen graph to ConcreteFunctions
frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                inputs=["map:0", "tra:0", "z:0"],
                                outputs=["Identity:0"],
                                print_graph=True)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)




### test frozen graph

###### inputs ##########
import numpy as np

trajectory_x = np.random.rand(1, args.history, 2)
map_x = np.random.rand(1, args.image_size, args.image_size, 1)
if args.model == 0:
    z_sample = np.random.rand(1, args.latent_dim)
elif args.model == 1:
    z = np.eye(args.components)
    z_sample = np.reshape(z[0], (1, args.components))



# Get predictions
#frozen_graph_predictions = frozen_func(map=tf.convert_to_tensor(map_x, dtype='float'), tra=tf.convert_to_tensor(trajectory_x, dtype='float'), z=tf.convert_to_tensor(z_sample, dtype='float'))

#for i in range(10):
#    print(i)
#    print(frozen_graph_predictions.numpy())




