import os
import sys
sys.path.append("model/")
import argparse

parser = argparse.ArgumentParser(description='Training Aura Model')
parser.add_argument('--model', type=int, default=3, help='Model type: '
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
print(model_name)
filepath = "saved_models/checkpoints/" + str(model_name) + "/"
dataset = "dataset/"



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
import aura_model_vit as model_lib
from Dataloader import DataGenerator2

validation_generator = DataGenerator2(base_path=dataset, batch_size=args.batch_size, dataset="test", output_dim=args.output_dim, history=args.history, future=args.future, image_size=args.image_size)

#######################################################   build model  #####################################################
# Instantiate the model
model_train = model_lib.get_model(obs_seq=args.history, pred_seq=args.future, number_of_outputs=args.output_dim, latent_dim=args.latent_dim, training=True, type=args.model, row=args.image_size, column=args.image_size)
model_train.summary()


# load weights
model_train.load_weights(filepath)
print('best weights loaded')
print("\n################ Test @top%.0f of model %s ################" % (args.agents, model_name))

#### model infer  ######
model = model_lib.get_model(obs_seq=args.history, pred_seq=args.future, number_of_outputs=args.output_dim, latent_dim=args.latent_dim, training=False, type=args.model, row=args.image_size, column=args.image_size)
model.compile(loss='mse', optimizer='adam')
model.summary()


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

import numpy as np
sys.path.append("utils/")
from ranking import gauss_rank
from plots import plot_pred, plot_error_bar, plot_dist
from evaluation import get_errors
import matplotlib.pyplot as plt
import matplotlib.image as image
import cv2
from scipy.stats import gaussian_kde


def imcrop(img, bbox):
    x1, y1, x2, y2 = bbox
    offset = 0
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        if(0 - x1 > offset): offset = 0 - x1
        if (0 - y1 > offset): offset = 0 - y1
        if(x2 - img.shape[1] > offset): offset = x2 - img.shape[1]
        if(y2 - img.shape[0] > offset): offset = y2 - img.shape[0]

        # channel = 1 -> semantic image
        if img.shape[2] == 1:
            img = cv2.copyMakeBorder(img, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=[255])

        # channel = 3 -> color image
        if img.shape[2] == 3:
            img = cv2.copyMakeBorder(img, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return img[y1+offset:y2+offset, x1+offset:x2+offset]


def compute_kde_nll(target_traj, pred_traj):
    '''
    pred_traj: (batch, T, K, 2/4)
    '''
    kde_ll = 0.

    log_pdf_lower_bound = -20
    batch_size, _, T, _ = pred_traj.shape
    for batch_num in range(batch_size):
        for timestep in range(T):
            try:
                kde = gaussian_kde(pred_traj[batch_num, :, timestep, ].T)
                pdf = np.clip(kde.logpdf(target_traj[batch_num, timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf / (T * batch_size)
            except np.linalg.LinAlgError:
                kde_ll = np.nan
    return -kde_ll




def cv_prediction(trajectory, num_points=150, avg_points=1):
    # a simple prediction function that predict straight line with constant velocity
    velocity_x = []
    velocity_y = []
    for i in range(1, avg_points + 1, 1):
        velocity_x.append(trajectory[-i, 0] - trajectory[-(1 + i), 0])
        velocity_y.append(trajectory[-i, 1] - trajectory[-(1 + i), 1])

    velocity_x = np.mean(velocity_x)
    velocity_y = np.mean(velocity_y)

    current_traj = trajectory[-1]
    results = np.zeros((len(trajectory) + num_points, 2))

    results[0:len(trajectory)] = trajectory

    for i in range(num_points):
        results[len(trajectory) + i] = np.array([current_traj[0] + velocity_x, current_traj[1] + velocity_y])
        current_traj = results[len(trajectory) + i]
    return results


for i in range(validation_generator.__len__()):
    #print("\nTest Model " + str(args.model) + " track " + str(i) + "/" + str(validation_generator.__len__())+"\n")
    [map_x, tra_x, map_y, tra_y], _ = validation_generator.__getitem__(i)
    last_pos_, px2mtr, rec_id, groundtruth_, track_object = validation_generator.get_meta_data()

    ### prediction para
    if parametric:
        predictions_, std_ = model_lib.test(model=model, type=args.model, batch_size=tra_y.shape[0], agents=args.agents,
                                          pred_length=args.future,
                                          map=map_x, groundtruth=tra_x, last_pos=last_pos_, latent_dim=args.latent_dim,
                                          components=args.components)

        # plot_error_bar(tra_y, predictions, std, batches=10, agents=10, args=args, save=True)
        plot_dist(groundtruth_, predictions_, std_, args=args, save=True)

    ### prediction non-para
    else:
        predictions_ = model_lib.test(model=model, type=args.model, batch_size=tra_y.shape[0], agents=args.agents,
                                      pred_length=args.future,
                                      map=map_x, groundtruth=tra_x, last_pos=last_pos_, latent_dim=args.latent_dim,
                                      components=args.components)

    if i > 0:
        predictions = np.concatenate((predictions, predictions_), axis=0)
        groundtruth = np.concatenate((groundtruth, groundtruth_), axis=0)
        last_pos = np.concatenate((last_pos, last_pos_), axis=0)

        if parametric:
            std = np.concatenate((std, std_), axis=0)
    else:
        predictions = predictions_
        groundtruth = groundtruth_
        last_pos = last_pos_
        if parametric:
            std = std_

# get ade/fde over all data
print('Predicting done!')
print(predictions.shape)


# cv_model
pred_cv = []
for batch in range(groundtruth.shape[0]):
    pred_cv.append(cv_prediction(trajectory=groundtruth[batch, :args.history], num_points=150, avg_points=1))
pred_cv = np.array(pred_cv)
pred_cv = np.reshape(pred_cv, (groundtruth.shape[0], 1, groundtruth.shape[1], groundtruth.shape[2]))

if args.test_cv_model:
    print("\n ################ Evaluation results cv_model ################")
    print("\nError 50:\n")
    errors = get_errors(groundtruth[:, args.history:args.history + 50], pred_cv[:, :, args.history:args.history + 50])
    print("\nError 100:\n")
    errors = get_errors(groundtruth[:, args.history:args.history + 100], pred_cv[:, :, args.history:args.history + 100])
    print("\nError 150:\n")
    errors = get_errors(groundtruth[:, args.history:], pred_cv[:, :, args.history:])



# plot
#        steps = 1
#        for batch in range(0, tra_y.shape[0] - steps, steps):
#            plot_pred(groundtruth[batch:batch+steps], predictions[batch:batch+steps], N=steps, groundtruth=True)

# Get the errors for ADE, DEF, Hausdorff distance, speed deviation, heading error
print("\n################ Evaluation results @top%.0f of model %s ################" % (args.agents, model_name))
print("\nError 50:\n")
get_errors(groundtruth[:, args.history:args.history + 50], predictions[:, :, :50])

print("\nError 60:\n")
get_errors(groundtruth[:, args.history:args.history + 60], predictions[:, :, :60])

print("\nError 100:\n")
get_errors(groundtruth[:, args.history:args.history + 100], predictions[:, :, :100])

print("\nError 150:\n")
get_errors(groundtruth[:, args.history:], predictions)

print("\nnll:\n")
print(compute_kde_nll(groundtruth[:, args.history:], predictions))

# check_collision(tra_y)

##
## Get the first time prediction by g
ranked_prediction = []
for prediction in predictions:
    ranks = gauss_rank(prediction)
    ranked_prediction.append(prediction[np.argmax(ranks)])
ranked_prediction = np.reshape(ranked_prediction, [-1, 1, args.future, 2])
print("\nEvaluation results for most-likely predictions")
ranked_errors = get_errors(groundtruth[:, args.history:], ranked_prediction)




recording_id = 0
t = np.linspace(0, 2 * np.pi, 150)


# visualization
if parametric:
    # visualization
    for object in range(predictions.shape[0]):
        if object > 0 and object % 10 == 0:
            recording_id += 1

        for agent in range(args.agents):
            print("recording_id " + str(recording_id) + " / object " + str(object)+ " / agent " + str(agent))
            ############################# show semantic with trajectory ######################################
            # load semantic frame image
            #image_mp = image.imread("dataset/semantic_" + str(recording_id) + ".png")
            # load original image
            if recording_id < 10:
                image_mp = image.imread("dataset/0" + str(recording_id) + "_background.png")
            else:
                image_mp = image.imread("dataset/" + str(recording_id) + "_background.png")

            scale_factor = 12
            if (recording_id > 32): scale_factor = 10

            # plot ground truth in map
            plt.figure()

            # get size for cropping image
            diff = 500 / 2

            point_1 = np.array((int((((groundtruth[object, 0, 0]) / px2mtr[recording_id]) / scale_factor)), int((((groundtruth[object, 0, 1]) / px2mtr[recording_id]) / scale_factor))))
            point_2 = np.array((int((((groundtruth[object, -1, 0]) / px2mtr[recording_id]) / scale_factor)), int((((groundtruth[object, -1, 1]) / px2mtr[recording_id]) / scale_factor))))
            distance = np.sqrt(np.sum(np.square(point_1 - point_2)))

            if distance > diff:
                diff = int(distance) + 10


            # plot center point
            plt.plot(diff, diff, marker='x', color="red", markersize=1)

            center_x = int(((last_pos[object, 0]) / px2mtr[recording_id]) / scale_factor)
            center_y = int((-(last_pos[object, 1]) / px2mtr[recording_id]) / scale_factor)

            bbox = int(center_x - diff), \
                   int(center_y - diff), \
                   int(center_x + diff), \
                   int(center_y + diff)

            cropped_image = imcrop(image_mp, bbox)

            offset_x = bbox[0]
            offset_y = bbox[1]

            # plot history
            #center_point_x = (((tra_x[object, :, 0] + last_pos[object, 0]) / px2mtr[recording_id]) / scale_factor) - offset_x
            #center_point_y = ((-(tra_x[object, :, 1] + last_pos[object, 1]) / px2mtr[recording_id]) / scale_factor) - offset_y
            #plt.plot(np.array(center_point_x, int), np.array(center_point_y, int), marker='.', color="yellow", markersize=1)

            # plot groundtruth
            center_point_x_ = (((groundtruth[object, :, 0]) / px2mtr[recording_id]) / scale_factor) - offset_x
            center_point_y_ = ((-(groundtruth[object, :, 1]) / px2mtr[recording_id]) / scale_factor) - offset_y
            plt.plot(np.array(center_point_x_, int), np.array(center_point_y_, int), marker='.', color="black", markersize=1)

            # plot cv_model
            pred_cv = np.reshape(pred_cv, (groundtruth.shape[0], groundtruth.shape[1], groundtruth.shape[2]))
            center_point_x_ = (((pred_cv[object, args.history:, 0]) / px2mtr[recording_id]) / scale_factor) - offset_x
            center_point_y_ = ((-(pred_cv[object, args.history:, 1]) / px2mtr[recording_id]) / scale_factor) - offset_y
            plt.plot(np.array(center_point_x_, int), np.array(center_point_y_, int), marker='.', color="yellow", markersize=1, alpha=0.2)

            # plot trajectory
            center_point_x_ = (((predictions[object, agent, :, 0]) / px2mtr[recording_id]) / scale_factor) - offset_x
            center_point_y_ = ((-(predictions[object, agent, :, 1]) / px2mtr[recording_id]) / scale_factor) - offset_y
            plt.plot(np.array(center_point_x_, int), np.array(center_point_y_, int), marker='.', color="red", markersize=1, alpha=0.2)

            # plot distribution for 50th point
            x_std = (((std[object, agent, 49, 0]) / px2mtr[recording_id]) / scale_factor)
            y_std = (((std[object, agent, 49, 1]) / px2mtr[recording_id]) / scale_factor)
            Ell = np.array([x_std * np.cos(t), y_std * np.sin(t)])
            plt.plot(center_point_x_[49] + Ell[0, :], center_point_y_[49] + Ell[1, :], c='blue', alpha=0.5)

            # plot distribution for 100th point
            x_std = (((std[object, agent, 99, 0]) / px2mtr[recording_id]) / scale_factor)
            y_std = (((std[object, agent, 99, 1]) / px2mtr[recording_id]) / scale_factor)
            Ell = np.array([x_std * np.cos(t), y_std * np.sin(t)])
            plt.plot(center_point_x_[99] + Ell[0, :], center_point_y_[99] + Ell[1, :], c='blue', alpha=0.5)

            # plot distribution for end point
            x_std = (((std[object, agent, -1, 0]) / px2mtr[recording_id]) / scale_factor)
            y_std = (((std[object, agent, -1, 1]) / px2mtr[recording_id]) / scale_factor)
            Ell = np.array([x_std * np.cos(t), y_std * np.sin(t)])
            plt.plot(center_point_x_[-1] + Ell[0, :], center_point_y_[-1] + Ell[1, :], c='blue', alpha=0.5)

            # plt.imshow(image_mp)
            plt.imshow(cropped_image)
            #plt.show()
            #plt.gcf().clear()
            #plt.close()

            save_folder = "images/record_id_" + str(recording_id) + "_object_" + str(object) + "_agent_" + str(agent) + ".jpg"
            plt.savefig(save_folder, bbox_inches='tight', dpi=150)
            plt.gcf().clear()
            plt.close()
else:
    for object in range(predictions.shape[0]):

        if object > 0 and object % 10 == 0:
            recording_id += 1

        print("recording_id " + str(recording_id) + " / object " + str(object))
        ############################# show semantic with trajectory ######################################
        # load semantic frame image
        # image_mp = image.imread("dataset/semantic_" + str(recording_id) + ".png")
        # load original image
        if recording_id < 10:
            image_mp = image.imread("dataset/0" + str(recording_id) + "_background.png")
        else:
            image_mp = image.imread("dataset/" + str(recording_id) + "_background.png")

        scale_factor = 12
        if (recording_id > 32): scale_factor = 10

        # plot ground truth in map
        plt.figure()

        # get size for cropping image
        diff = 500 / 2

        point_1 = np.array((int((((groundtruth[object, 0, 0]) / px2mtr[recording_id]) / scale_factor)),
                            int((((groundtruth[object, 0, 1]) / px2mtr[recording_id]) / scale_factor))))
        point_2 = np.array((int((((groundtruth[object, -1, 0]) / px2mtr[recording_id]) / scale_factor)),
                            int((((groundtruth[object, -1, 1]) / px2mtr[recording_id]) / scale_factor))))
        distance = np.sqrt(np.sum(np.square(point_1 - point_2)))

        if distance > diff:
            diff = int(distance) + 10

        # plot center point
        plt.plot(diff, diff, marker='x', color="red", markersize=1)

        center_x = int(((last_pos[object, 0]) / px2mtr[recording_id]) / scale_factor)
        center_y = int((-(last_pos[object, 1]) / px2mtr[recording_id]) / scale_factor)

        bbox = int(center_x - diff), \
               int(center_y - diff), \
               int(center_x + diff), \
               int(center_y + diff)

        cropped_image = imcrop(image_mp, bbox)

        offset_x = bbox[0]
        offset_y = bbox[1]

        # plot history
        # center_point_x = (((tra_x[object, :, 0] + last_pos[object, 0]) / px2mtr[recording_id]) / scale_factor) - offset_x
        # center_point_y = ((-(tra_x[object, :, 1] + last_pos[object, 1]) / px2mtr[recording_id]) / scale_factor) - offset_y
        # plt.plot(np.array(center_point_x, int), np.array(center_point_y, int), marker='.', color="yellow", markersize=1)

        # plot groundtruth
        center_point_x_ = (((groundtruth[object, :, 0]) / px2mtr[recording_id]) / scale_factor) - offset_x
        center_point_y_ = ((-(groundtruth[object, :, 1]) / px2mtr[recording_id]) / scale_factor) - offset_y
        plt.plot(np.array(center_point_x_, int), np.array(center_point_y_, int), marker='.', color="black",
                 markersize=1)

        # plot cv_model
        pred_cv = np.reshape(pred_cv, (groundtruth.shape[0], groundtruth.shape[1], groundtruth.shape[2]))
        center_point_x_ = (((pred_cv[object, args.history:, 0]) / px2mtr[recording_id]) / scale_factor) - offset_x
        center_point_y_ = ((-(pred_cv[object, args.history:, 1]) / px2mtr[recording_id]) / scale_factor) - offset_y
        plt.plot(np.array(center_point_x_, int), np.array(center_point_y_, int), marker='.', color="yellow",
                 markersize=1, alpha=0.2)

        # plot trajectory
        for agent in range(args.agents):
            center_point_x_ = (((predictions[object, agent, :, 0]) / px2mtr[recording_id]) / scale_factor) - offset_x
            center_point_y_ = ((-(predictions[object, agent, :, 1]) / px2mtr[recording_id]) / scale_factor) - offset_y
            plt.plot(np.array(center_point_x_, int), np.array(center_point_y_, int), marker='.', color="red",
                     markersize=1, alpha=0.2)
            # plot distribution for end point
            if parametric:
                x_std = (((std[object, agent, -1, 0]) / px2mtr[recording_id]) / scale_factor)
                y_std = (((std[object, agent, -1, 1]) / px2mtr[recording_id]) / scale_factor)
                Ell = np.array([x_std * np.cos(t), y_std * np.sin(t)])
                plt.plot(center_point_x_[-1] + Ell[0, :], center_point_y_[-1] + Ell[1, :], c='blue', alpha=0.2)

        # plt.imshow(image_mp)
        plt.imshow(cropped_image)
        # plt.show()
        # plt.gcf().clear()
        # plt.close()

        save_folder = "images/record_id_" + str(recording_id) + "_object_" + str(object) + ".jpg"
        plt.savefig(save_folder, bbox_inches='tight', dpi=150)
        plt.gcf().clear()
        plt.close()



    # plot most likely predictions
#    steps = 1
#    for batch in range(0, tra_y.shape[0] - steps, steps):
#        plot_pred(groundtruth[batch:batch+steps], ranked_prediction[batch:batch+steps], N=steps, groundtruth=True, save=False, index=batch)



