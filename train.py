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
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
parser.add_argument('--gpu', type=int, default=0, help='Select a gpu')
parser.add_argument('--weights', type=bool, default=False, help='load old weights if true')
parser.add_argument('--components', type=int, default=10, help='# of components')

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
print(model_name)
filepath = "saved_models/checkpoints/" + str(model_name) + "/"
filepath_best = "saved_models/checkpoints_best/" + str(model_name) + "/"
if not os.path.exists(filepath): os.mkdir(filepath)

# parametric/non-parametric
if args.model == 0 or args.model == 2 or args.model == 4:
    parametric = False
else:
    parametric = True


import tensorflow as tf
import tensorflow_probability as tfp

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


#######################################################   data  #####################################################
from datetime import datetime
import aura_model_vit as model_lib
#from Dataloader import DataGenerator
from Dataloader_2 import DataGenerator

#training_generator = DataGenerator(batch_size=args.batch_size, dataset="train")
#validation_generator = DataGenerator(batch_size=args.batch_size, dataset="test")
training_generator = DataGenerator(dataset="train")
validation_generator = DataGenerator(dataset="test")

#from Dataloader import DataGenerator2

#validation_generator = DataGenerator2(base_path="dataset/", batch_size=args.batch_size, dataset="test", output_dim=args.output_dim, history=args.history, future=args.future, image_size=args.image_size)


#######################################################   build model  #####################################################


# Instantiate the model
model = model_lib.get_model(obs_seq=args.history, pred_seq=args.future, number_of_outputs=args.output_dim, latent_dim=args.latent_dim, training=True, type=args.model, row=args.image_size, column=args.image_size)


mse = tf.keras.losses.MeanSquaredError()

# non-parametric loss
def non_parametric_loss(y, pred):
    # mse loss
    mse_loss = mse(y, pred)

    model.mse_tracker.update_state(mse_loss)
    model.dkl_tracker.update_state(tf.reduce_sum(model.losses))

    return (mse_loss * 4) + tf.reduce_sum(model.losses)



# parametric loss
def parametric_loss(y, pred):
    mean = pred[:, :, :args.output_dim]
    std = pred[:, :, args.output_dim:-1]
    #corr = pred[:, :, -1]

    gm = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=(std))
    log_likelihood = gm.log_prob(y)
    #mse loss
    mse_loss = mse(y, gm.mean())

    #log_likelihood = 0
    #for batch in range(args.batch_size):
    #    for time in range(args.future):
    #        mu = mean[batch, time]
    #        sx_val = std[batch, time, 0]
    #        sy_val = std[batch, time, 1]
    #        rho = corr[batch, time]
            # Extract covariance matrix
    #        cov = [[sx_val * sx_val, rho * sx_val * sy_val], [rho * sx_val * sy_val, sy_val * sy_val]]

    #        gm = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=tf.linalg.cholesky(cov))
    #        y_ = y[batch, time]
    #        log_likelihood += gm.log_prob(y_)

    # update loss tracker
    model.mse_tracker.update_state(mse_loss)
    model.dkl_tracker.update_state(model.losses)
    model.nll_tracker.update_state(- tf.math.reduce_sum(log_likelihood))

    return model.losses + (4 * mse_loss) - tf.math.reduce_sum(log_likelihood)



opt = tf.keras.optimizers.Adam(learning_rate=3e-5, beta_1=0.9, beta_2=0.999, decay=1e-6, amsgrad=False)

if parametric == True:
    loss_str = "loss function: parametric_loss"
    loss_ = parametric_loss
else:
    loss_str = "loss function: non_parametric_loss"
    loss_ = non_parametric_loss

model.compile(optimizer=opt, loss=loss_)
model.summary()

print("Start training model " + str(model_name) + " / " + loss_str)


if args.weights:
    model.load_weights(filepath)
    print('best weights loaded')



start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print("Start time: " + start)



checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True, monitor='val_loss', save_best_only=False, verbose=0, override=True, mode='min', save_format="h5")
checkpoint_best = tf.keras.callbacks.ModelCheckpoint(filepath_best, save_weights_only=True, monitor='val_loss', save_best_only=True, verbose=0, override=True, mode='min', save_format="h5")




# Create a TensorBoard callback
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '10,20')


callbacks_list = [checkpoint, checkpoint_best, tboard_callback]



model.fit(training_generator, validation_data=validation_generator, shuffle=True,
          epochs=args.epochs, batch_size=args.batch_size, verbose=1, callbacks=callbacks_list)

end = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print("Start time: " + start)
print("End time: " + end)