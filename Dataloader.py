import tensorflow as tf
import numpy as np
import time
import os
import glob
import sys
from loguru import logger
sys.path.append("src/")
import tracks_import as ti
import cv2


def imcrop(img, bbox):
    x1, y1, x2, y2 = bbox
    offset = 0
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        if(0 - x1 > offset): offset = 0 - x1
        if (0 - y1 > offset): offset = 0 - y1
        if(x2 - img.shape[1] > offset): offset = x2 - img.shape[1]
        if(y2 - img.shape[0] > offset): offset = y2 - img.shape[0]

        img = cv2.copyMakeBorder(img, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=[255])
    return img[y1+offset:y2+offset, x1+offset:x2+offset]


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset="train", base_path="np_data/", batch_size=1, history=10, future=150, image_size=300):


        self.batch_size = batch_size
        self.dataset = dataset
        self.base_path = base_path + str(self.dataset)
        if self.dataset != "train": self.dataset = "test"
        self.history = history
        self.future = future
        self.image_size = image_size
        self.tracks_files = glob.glob(self.base_path + "/*.npz")
        self.compute_length()
        self.counter = 0


    def compute_length(self):
        self.data_len = len(self.tracks_files)/self.batch_size

    def __len__(self):
        return int(self.data_len)

    def __getitem__(self, index):

        file = self.base_path + "/tra" + str(self.counter) + ".npz"
        loaded = np.load(file)
        self.map_x = loaded['map_x']
        self.map_y = loaded['map_y']
        self.tra_x = loaded['tra_x']
        self.tra_y = loaded['tra_y']

        if self.dataset != "train":
            self.groundtruth = loaded['groundtruth']
            self.last_pos = loaded['last_pos']
            self.track_object = loaded['track_object']
            self.px2mtr = loaded['px2mtr']
            self.recording_id = loaded['recording_id']


        if self.batch_size > 1:
            for i in range(1, self.batch_size):
                file = self.base_path + "/tra" + str(self.counter + i)+".npz"
                loaded = np.load(file)
                self.map_x = np.concatenate((self.map_x, loaded['map_x']), axis=0)
                self.map_y = np.concatenate((self.map_y, loaded['map_y']), axis=0)
                self.tra_x = np.concatenate((self.tra_x, loaded['tra_x']), axis=0)
                self.tra_y = np.concatenate((self.tra_y, loaded['tra_y']), axis=0)

                if self.dataset != "train":
                    self.groundtruth = np.concatenate((self.groundtruth, loaded['groundtruth']), axis=0)
                    self.last_pos = np.concatenate((self.last_pos, loaded['last_pos']), axis=0)
                    self.track_object = np.concatenate((self.track_object, loaded['track_object']), axis=0)



        self.counter += self.batch_size
        if self.counter >= self.data_len:
            self.counter = 0

        return [self.map_x, self.tra_x, self.map_y, self.tra_y], self.tra_y

    def get_meta_data(self):
        return self.last_pos, self.px2mtr, self.recording_id, self.groundtruth, self.track_object




class DataGenerator2(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset="train", base_path="dataset/", batch_size=None, history=20, future=150, input_dim=2, output_dim=2, image_size=227, radius=50):

        self.base_path = base_path
        self.batch_size = batch_size

        self.history = history
        self.future = future
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dataset = dataset
        if self.dataset != "train": self.dataset = "test"

        self.tracks_files = sorted(glob.glob(self.base_path + "*_tracks.csv"))
        self.static_tracks_files = sorted(glob.glob(self.base_path + "*_tracksMeta.csv"))
        self.recording_meta_files = sorted(glob.glob(self.base_path + "*_recordingMeta.csv"))

        self.length = self.history + self.future
        self.recording_id = 0
        self.radius = radius # 50 m
        self.image_input_size = image_size # 224
        self.scale_factor = 12  # scale factor inD=12 rounD=10

        # compute data length
        self.data_len = 0
        self.index = 0
        self.px2mtr = np.zeros(35)

        self.compute_length()
        self.track_number = 0

        self.get_track_data()


    def compute_length(self):

        for track_file, static_tracks_file, recording_meta_file in zip(self.tracks_files,
                                                                       self.static_tracks_files,
                                                                       self.recording_meta_files):
            # log
            logger.info("Loading csv files {}, {} and {}", track_file, static_tracks_file, recording_meta_file)
            # load data for track
            tracks, static_info, meta_info = ti.read_from_csv(track_file, static_tracks_file, recording_meta_file)
            if self.dataset == "train":
                counter = 0
                for object in range(len(tracks)):
                    if tracks[object]['xCenter'].shape[0] >= 2 * self.length:
                        counter += 1

                self.data_len += int(counter / self.batch_size)
            else:
                if self.batch_size < 10:
                    # for small batch sizes min. 10 test trajectories per track
                    self.data_len += int(10 / self.batch_size)
                else:
                    self.data_len += 1

    def __len__(self):
        return int(self.data_len)

    def __getitem__(self, index):

        # load new data
        if self.index >= self.track_length/self.batch_size:
            self.index = 0
            self.recording_id += 1
            if self.recording_id < len(self.tracks_files):
                self.get_track_data()

        map_x = self.map[self.index:self.index + self.batch_size, :self.history]
        tra_x = self.trajectory[self.index:self.index + self.batch_size, :self.history, :self.input_dim]
        map_y = self.map[self.index:self.index + self.batch_size, self.history:]
        tra_y = self.trajectory[self.index:self.index + self.batch_size, self.history:, :self.output_dim]

        if self.dataset != "train":
            self.last_pos_batch = self.last_pos[self.index:self.index + self.batch_size]
            self.groundtruth_batch = self.groundtruth[self.index:self.index + self.batch_size]
            self.track_object_batch = self.track_object[index:index + self.batch_size]

        self.index += self.batch_size

        return [map_x, tra_x, map_y, tra_y], tra_y

    def get_meta_data(self):

        return self.last_pos_batch, self.px2mtr, int(self.recording_id), self.groundtruth_batch, self.track_object_batch


    def get_track_data(self):
        # load data for track
        tracks, static_info, meta_info = ti.read_from_csv(self.tracks_files[self.recording_id],
                                                          self.static_tracks_files[self.recording_id],
                                                          self.recording_meta_files[self.recording_id])
        # log
        logger.info("Loading csv -- recording_id {}", self.recording_id)

        # get track length
        self.track_length = 0

        # scale factor inD=12 rounD=10
        if self.recording_id > 32:
            self.scale_factor = 10

        self.px2mtr[self.recording_id] = meta_info[0]["orthoPxToMeter"]
        # convert to m -> pixel
        px2mtr = meta_info[0]["orthoPxToMeter"]
        image_radius = int(self.radius // (px2mtr * self.scale_factor))

        length_ = self.batch_size


        if self.dataset == "train":
            self.trajectory = np.zeros((len(tracks) - 10, self.length, 5), np.float32)
            self.last_pos = np.zeros((len(tracks) - 10, 2), np.float32)


            self.map = np.zeros((len(tracks) - 10, self.length, self.image_input_size, self.image_input_size, 1))
        else:

            if self.batch_size < 10: length_ = 10

            self.trajectory = np.zeros((length_, self.length, 5), np.float32)
            self.groundtruth = np.zeros((length_, self.length, 2), np.float32)
            self.last_pos = np.zeros((length_, 2), np.float32)
            self.map = np.zeros((length_, self.length, self.image_input_size, self.image_input_size, 1))
            self.track_object = np.zeros((length_, 2))


        # filepath for image frame
        filepath = self.base_path + str(self.recording_id)
        count = 0
        for object in range(len(tracks)):
            # ignore short trajectories
            if tracks[object]['xCenter'].shape[0] >= 2 * self.length:

                # skip first trajectories
                if self.dataset == "train" and count < length_:
                    count += 1
                    continue

                # stop after test data is complete
                elif self.dataset != "train" and self.track_length >= length_:
                    break


                ################################## create trajectories ###############################
                # get every second value 25 Hz -> 12.5 Hz
                x = tracks[object]['xCenter'][0:2 * self.length:2]
                y = tracks[object]['yCenter'][0:2 * self.length:2]
                v_x = tracks[object]['xVelocity'][0:2 * self.length:2] ** 2
                v_y = tracks[object]['yVelocity'][0:2 * self.length:2] ** 2
                # compute velocity
                v = np.sqrt(v_x+v_y)
                h = tracks[object]['heading'][0:2 * self.length:2]

                # last pos
                self.last_pos[self.track_length, 0] = x[self.history - 1]
                self.last_pos[self.track_length, 1] = y[self.history - 1]

                # fill trajectory array and normalize
                x_norm = x - self.last_pos[self.track_length, 0]
                y_norm = y - self.last_pos[self.track_length, 1]
                self.trajectory[self.track_length, :, 0] = x_norm
                self.trajectory[self.track_length, :, 1] = y_norm
                self.trajectory[self.track_length, :, 2] = v
                self.trajectory[self.track_length, :, 3] = h // 360.0

                if self.dataset != "train":
                    self.groundtruth[self.track_length, :, 0], self.groundtruth[self.track_length, :, 1] = x, y
                    self.track_object[self.track_length, 0], self.track_object[self.track_length, 1] = self.recording_id, object
                    #print("Dataloader")
                    #print(self.groundtruth[self.track_length, :10])
                    #print(self.trajectory[self.track_length, :10, 0] + self.last_pos[self.track_length, 0])
                    #print(self.trajectory[self.track_length, :10, 1] + self.last_pos[self.track_length, 1])
                    #print()


                ################################## create images ######################################
                count_frame = 0
                for current_frame in range(0, 2 * self.length, 2):
                    # center point and heading
                    center_point = tracks[object]["centerVis"][current_frame] / self.scale_factor

                    # load semantic frame image
                    image = cv2.imread(
                        str(filepath) + '/image_frames_' + str(self.recording_id) + '_' + str(current_frame) + '.png',
                        cv2.IMREAD_GRAYSCALE)

                    # crop image
                    diff = image_radius / 2
                    bbox = int(center_point[0] - diff), \
                           int(center_point[1] - diff), \
                           int(center_point[0] + diff), \
                           int(center_point[1] + diff)

                    cropped_image = imcrop(image, bbox)

                    self.map[self.track_length, count_frame, :, :, 0] = cv2.resize(cropped_image, (self.image_input_size, self.image_input_size),
                                                    interpolation=cv2.INTER_AREA)
                    count_frame += 1
                # count self.track_length
                self.track_length += 1
        return self.trajectory, self.map


    def get_object_list(self, tracks):
        object_list = {}
        for i in range(len(tracks)):
            object_list[i] = tracks[i]['xCenter'].shape
        return object_list







'''

recording_id = 0
for track_file, static_tracks_file, recording_meta_file in zip(tracks_files,
                                                               static_tracks_files,
                                                               recording_meta_files):

    if recording_id > 32:

        # log
        logger.info("Loading csv files {}, {} and {}", track_file, static_tracks_file, recording_meta_file)
        # load data for track
        tracks, static_info, meta_info = ti.read_from_csv(track_file, static_tracks_file, recording_meta_file)

        # parameter
        radius = 50 #50 m
        # scale factor inD=12 rounD=10
        scale_factor = 12
        if recording_id > 32:
            scale_factor = 10
        # convert to m -> pixel
        px2mtr = meta_info[0]["orthoPxToMeter"]
        image_radius = int(50//(px2mtr * scale_factor))
        image_input_size = 224

        # get number of objects in track
        object_list = get_object_list(tracks)

        trajectory = np.zeros((len(object_list), length, output_dim), np.float32)
        map = np.zeros((len(object_list), length, image_input_size, image_input_size))

        # filepath for image frame
        filepath = base_path + str(recording_id)
        for object in range(len(object_list)):

            if tracks[object]['xCenter'].shape[0] >= 2 * length:

                ################################## create trajectories ###############################

                # get every second value 25 Hz -> 12.5 Hz
                x = tracks[object]['xCenter'][0:2 * length:2]
                y = tracks[object]['yCenter'][0:2 * length:2]
                #v_x = tracks[object]['xVelocity'][0:2 * length:2] ** 2
                #v_y = tracks[object]['yVelocity'][0:2 * length:2] ** 2
                # compute velocity
                #v = np.sqrt(v_x+v_y)
                #h = tracks[object]['heading'][0:2 * length:2]

                trajectory[object, :, 0], trajectory[object, :, 1] = x, y

                ################################## create images ######################################

                initialFrame = static_info[object]["initialFrame"]
                finalFrame = static_info[object]["finalFrame"]

                #frames = finalFrame - initialFrame
                count = 0
                for current_frame in range(0, 2*length, 2):

                    # center point and heading
                    center_point = tracks[object]["centerVis"][current_frame] / scale_factor
                    heading = tracks[object]["headingVis"][current_frame]

                    # load semantic frame image
                    image = cv2.imread(str(filepath) + '/image_frames_' + str(recording_id) + '_' + str(current_frame) + '.png', cv2.IMREAD_GRAYSCALE)

                    # crop image
                    diff = image_radius / 2
                    bbox = int(center_point[0] - diff), int(center_point[1] - diff), int(center_point[0] + diff), int(center_point[1] + diff)
                    cropped_image = imcrop(image, bbox)

                    # rotate image
                    #Matrix = cv2.getRotationMatrix2D((int(cropped_image.shape[0]/2), int(cropped_image.shape[1]/2)), heading, 1.0)
                    #rotated_image = cv2.warpAffine(cropped_image, Matrix, (cropped_image.shape[:2]))

                    # resize for model input
                    #input_image = cv2.resize(cropped_image, (image_input_size, image_input_size), interpolation=cv2.INTER_AREA)

                    map[object][count] = cv2.resize(cropped_image, (image_input_size, image_input_size), interpolation=cv2.INTER_AREA)
                    count += 1

                    # visualization only
                    #cv2.imshow("image", image)
                    #resized_image = cv2.resize(input_image, (800, 800), interpolation=cv2.INTER_AREA)
                    #cv2.imshow("resized", input_image)
                    #cv2.waitKey(0)

    # track id
    recording_id += 1




x_m = all_tracks[0]['xCenter']
y_m = all_tracks[0]['yCenter']


x_px = x_m/px2mtr
y_px = -(y_m)/px2mtr

'''

'''

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, args, datatype="train"):
        self.args = args
        self.datatype = datatype
        self.data_len = 0

        self.data = np.load("../processed_data/train/train_merged0.npz")
        split = int(len(self.data["offsets"]) * self.args.split)
        # train data
        if self.datatype == "train":
            self.offsets = self.data["offsets"][:split]
            self.occupancy = self.data["occupancy"][:split]
        # val data
        else:
            self.offsets = self.data["offsets"][split:]
            self.occupancy = self.data["occupancy"][split:]

        self.count = int(round((len(self.offsets) - self.args.batch_size)/ self.args.batch_size))

        self.counter = 0
        self.set = 0
        self.ind = 0
        # compute data length
        self.compute_length()


    def compute_length(self):
        for i in range(3):
            data = np.load("../processed_data/train/train_merged%s.npz" % (str(i)))
            split = int(len(data["offsets"]) * self.args.split)
            self.data_len = 0
            # train data
            if self.datatype == "train":
                self.data_len += round((len(data["offsets"][:split]) - self.args.batch_size) / self.args.batch_size)
            # val data
            else:
                self.data_len += round((len(data["offsets"][split:]) - self.args.batch_size)/ self.args.batch_size)

    def __len__(self):
        return int(self.data_len)


    def __getitem__(self, index):

        if self.ind == self.count:
            self.counter = 0
            self.ind = 0
            self.set += 1
            if self.set == 3:
                self.set = 0
            self.data = np.load("../processed_data/train/train_merged%s.npz" % (str(self.set)))
            split = int(len(self.data["offsets"]) * self.args.split)
            # train data
            if self.datatype == "train":
                self.offsets = self.data["offsets"][:split]
                self.occupancy = self.data["occupancy"][:split]
            # val data
            else:
                self.offsets = self.data["offsets"][split:]
                self.occupancy = self.data["occupancy"][split:]

            self.count = int(round((len(self.offsets) - self.args.batch_size)/ self.args.batch_size))

        train_x = self.offsets[self.counter:self.counter + self.args.batch_size, :self.args.obs_seq - 1, 4:6]
        train_occu = self.occupancy[self.counter:self.counter + self.args.batch_size, :self.args.obs_seq - 1, ...,
                     :self.args.enviro_pdim[-1]]
        train_y = self.offsets[self.counter:self.counter + self.args.batch_size, self.args.obs_seq - 1:, 4:6]
        train_y_occu = self.occupancy[self.counter:self.counter + self.args.batch_size, self.args.obs_seq - 1:, ...,
                       :self.args.enviro_pdim[-1]]
        self.counter += self.args.batch_size
        self.ind += 1
        return [train_occu, train_x, train_y_occu, train_y], train_y



class DataGenerator2(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, args, datatype="train"):
        self.args = args
        self.datatype = datatype
        self.data_len = 0

        self.data = np.load("../processed_data/train/train_merged0.npz")
        split = int(len(self.data["offsets"]) * self.args.split)
        # train data
        if self.datatype == "train":
            self.offsets = self.data["offsets"][:split]
            self.occupancy = self.data["occupancy"][:split]
        # val data
        else:
            self.offsets = self.data["offsets"][split:]
            self.occupancy = self.data["occupancy"][split:]

        self.count = int(round((len(self.offsets) - self.args.batch_size)/ self.args.batch_size))

        self.counter = 0
        self.set = 0
        self.ind = 0
        # compute data length
        self.compute_length()


    def compute_length(self):
        for i in range(3):
            data = np.load("../processed_data/train/train_merged%s.npz" % (str(i)))
            split = int(len(data["offsets"]) * self.args.split)
            self.data_len = 0
            # train data
            if self.datatype == "train":
                self.data_len += round((len(data["offsets"][:split]) - self.args.batch_size) / self.args.batch_size)
            # val data
            else:
                self.data_len += round((len(data["offsets"][split:]) - self.args.batch_size)/ self.args.batch_size)

    def __len__(self):
        return int(self.data_len)


    def __getitem__(self, index):

        if self.ind == self.count:
            self.counter = 0
            self.ind = 0
            self.set += 1
            if self.set == 3:
                self.set = 0
            self.data = np.load("../processed_data/train/train_merged%s.npz" % (str(self.set)))
            split = int(len(self.data["offsets"]) * self.args.split)
            # train data
            if self.datatype == "train":
                self.offsets = self.data["offsets"][:split]
                self.occupancy = self.data["occupancy"][:split]
            # val data
            else:
                self.offsets = self.data["offsets"][split:]
                self.occupancy = self.data["occupancy"][split:]

            self.count = int(round((len(self.offsets) - self.args.batch_size)/ self.args.batch_size))

        train_x = self.offsets[self.counter:self.counter + self.args.batch_size, :self.args.obs_seq - 1, 4:6]
        train_occu = self.occupancy[self.counter:self.counter + self.args.batch_size, :self.args.obs_seq - 1, ...,
                     :self.args.enviro_pdim[-1]]
        train_y = self.offsets[self.counter:self.counter + self.args.batch_size, self.args.obs_seq - 1:, 4:6]
        self.counter += self.args.batch_size
        self.ind += 1
        return [train_occu, train_x], train_y
        
'''



