import cv2
import numpy as np
import os
import glob
import sys
sys.path.append("src/")
import tracks_import as ti
from loguru import logger

base_path = "dataset/"
tracks_files = sorted(glob.glob(base_path + "*_tracks.csv"))
static_tracks_files = sorted(glob.glob(base_path + "*_tracksMeta.csv"))
recording_meta_files = sorted(glob.glob(base_path + "*_recordingMeta.csv"))

recording_id = 0
start_id = 33
for track_file, static_tracks_file, recording_meta_file in zip(tracks_files,
                                                               static_tracks_files,
                                                               recording_meta_files):


    logger.info("Loading csv files {}, {} and {}", track_file, static_tracks_file, recording_meta_file)

    # tracks, info, meta
    tracks, static_info, meta_info = ti.read_from_csv(track_file, static_tracks_file, recording_meta_file)

    # create folder for frame_images
    filepath = base_path + str(recording_id)
    if not os.path.exists(filepath): os.mkdir(filepath)

    # scale factor inD=12 rounD=10
    scale_factor = 12
    if recording_id > 32:
        scale_factor = 10

    object_list = {}
    for i in range(len(tracks)):
        object_list[i] = tracks[i]['xCenter'].shape

    maximum_frames = np.max([static_info[track["trackId"]]["finalFrame"] for track in tracks])


    # Save ids for each frame
    ids_for_frame = {}
    for i_frame in range(maximum_frames):
        indices = [i_track for i_track, track in enumerate(tracks)
                   if
                   static_info[track["trackId"]]["initialFrame"] <= i_frame <= static_info[track["trackId"]][
                       "finalFrame"]]
        ids_for_frame[i_frame] = indices


    def get_color(type=None):
        if(type == 'car') or (type == 'van'): return 27
        elif (type == 'truck_bus') or (type == 'truck') or (type == 'bus') or (type == 'trailer'): return 28
        elif (type == 'motorcycle'): return 34
        elif (type == 'bicycle'): return 32
        elif (type == 'pedestrian'): return 42

    if recording_id >= start_id:
        for current_frame in range(maximum_frames):
            # load semantic image
            image = cv2.imread(str(base_path) + 'semantic_' + str(recording_id) + '.png', cv2.IMREAD_GRAYSCALE)


            for track_ind in ids_for_frame[current_frame]:
                track = tracks[track_ind]

                track_id = track["trackId"]
                static_track_information = static_info[track_id]
                initial_frame = static_track_information["initialFrame"]
                current_index = current_frame - initial_frame

                object_class = static_track_information["class"]
                is_vehicle = object_class in ["car", "truck_bus", "motorcycle", "bicycle"]
                bounding_box = track["bboxVis"][current_index] / scale_factor
                center_point = track["centerVis"][current_index] / scale_factor


                # pedestrian 10x10 pixel/ one pixel 4x4 cm
                image = cv2.circle(image, center=(int(center_point[0]), int(center_point[1])),
                                   radius=5, color=get_color(object_class), thickness=-1)

                if is_vehicle:
                    # Polygon corner points coordinates
                    pts = np.array(bounding_box, np.int32)
                    image = cv2.polylines(image, [pts], True, get_color(object_class), thickness=1)

                    # add direction
                    triangle_factor = 0.75
                    a_x = bounding_box[3, 0] + ((bounding_box[2, 0] - bounding_box[3, 0]) * triangle_factor)
                    b_x = bounding_box[0, 0] + ((bounding_box[1, 0] - bounding_box[0, 0]) * triangle_factor)
                    c_x = bounding_box[2, 0] + ((bounding_box[1, 0] - bounding_box[2, 0]) * 0.5)

                    a_y = bounding_box[3, 1] + ((bounding_box[2, 1] - bounding_box[3, 1]) * triangle_factor)
                    b_y = bounding_box[0, 1] + ((bounding_box[1, 1] - bounding_box[0, 1]) * triangle_factor)
                    c_y = bounding_box[2, 1] + ((bounding_box[1, 1] - bounding_box[2, 1]) * 0.5)

                    pts = np.array([[a_x, a_y], [b_x, b_y], [c_x, c_y]], np.int32)
                    image = cv2.polylines(image, [pts], True, get_color(object_class), thickness=1)


            #cv2.imshow("cropped", image)
            #cv2.waitKey(0)

            image = cv2.imwrite(str(filepath) + '/image_frames_' + str(recording_id) + '_' + str(current_frame) + '.png', image)
            logger.info("Image write recording id {}/34 frame {}/{}", recording_id, current_frame, maximum_frames)

    # count track id
    recording_id += 1
    #show last image
    #cv2.imshow("image", image)
    #cv2.waitKey(0)







