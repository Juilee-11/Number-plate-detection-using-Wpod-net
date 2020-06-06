import glob
from os.path import splitext, basename
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
from utils import detect_lp


def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model

    except Exception as e:
        print(e)


def preprocess_image(image_path,dir_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'{dir_path}preprocess_img.jpg', img)
    img = img / 255
    return img


def get_plate(image_path, dir_path):
    Dmax = 608
    Dmin = 288
    vehicle = preprocess_image(image_path,dir_path)
    # print("preprocessing ends-> !! ")
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])  # 350-197
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _, LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor


def draw_box(image_path, cor, thickness=5):
    vehicle_image = cv2.imread(image_path)
    pts = []
    x_coordinates = cor[0][0]
    y_coordinates = cor[0][1]
    # store the top-left, top-right, bottom-left, bottom-right 
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]), int(y_coordinates[i])])
    pts = np.array(pts, np.int32)
    # pts = pts.reshape((-1, 1, 2))

    # vehicle_image = preprocess_image(image_path)
    cv2.polylines(vehicle_image, [pts], True, (0, 255, 0), thickness)
    return vehicle_image


if __name__ == "__main__":
    wpod_net_path = "wpod-net.json"
    wpod_net = load_model(wpod_net_path)
    image_paths = glob.glob("/home/edlabadkar/projects/wpod/Plate_detect_and_recognize/Data-Images_Input/Number_Plate_Data_paralaxiom/*.jpg")
    print("Found %i images..." % (len(image_paths)))
    c = 0
    for one_img_path in image_paths:
        frame_name = (one_img_path.split('/')[-1]).split('.')[0]

        dir_path = f'/home/edlabadkar/output6/{frame_name}/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        img=cv2.imread(one_img_path)
        cv2.imwrite(f'{dir_path}ori_img.jpg', img)
        LpImg, cor = get_plate(one_img_path, dir_path)
        try:
            cv2.imwrite(f'{dir_path}LpImg.jpg', LpImg[0])
        except:
            print("Error")
        if not cor:
            continue
        img_v = draw_box(one_img_path, cor)
        # print("Coordinate of plate(s) in image: \n", cor[0].tolist())
        cv2.imwrite(f'{dir_path}final_bbox.jpg'.format(c), img_v)
        print("Done process!!")
        c += 1
    