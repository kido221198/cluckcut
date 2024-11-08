"""
Making YOLOv8 annotations from Colored Labels (of oranges)
"""

import shutil
import numpy as np
import cv2
import os
import imutils
from copy import deepcopy

root_path = './20231201-labeled_oranges-ITI'
original_path = f"{root_path}/JPEGImages/processed"
anno_path = f"{root_path}/SegmentationClass/processed"
new_anno_path = "./oranges_annotations"
new_image_path = "./oranges_images"
new_mask_path = "./oranges_masks"

LABELS = {
    # 0: (0, 0, 0),        # background, ignored
    1: (78, 153, 115),    # blackhead
    2: (165, 236, 223),       # crack
    3: (26, 170, 44),   # deformation
    4: (81, 185, 10),         # other
    5: (173, 197, 71),       # rotten
    6: (230, 214, 73),         # scar
    7: (93, 149, 13),         # stain
}


def smooth_piece(contour, smoothing_rate=4, delta=4):
    """
    Smooth contour from integer pixels to floating point pixels
    https://arxiv.org/abs/2109.03908

    :param contour: 3-dimension array
    :param smoothing_rate: iterations
    :param delta: distance between the on-contour points
    :return: 3-dimension array
    """
    for _ in range(smoothing_rate):
        r_contour = [contour[0]]
        dists = [0]     # Geodesic(?) Distance from point #0 towards others
        l = len(contour)
        for i in range(1, l + 1):
            dists.append(dists[-1] + np.linalg.norm(contour[i % l] - contour[(i - 1) % l]))
        num_point = int(dists[-1] / delta)

        # Adjust points
        for i in range(1, num_point):
            desired_dist = delta * i
            ix = [k for k in range(l) if dists[k] < desired_dist <= dists[k + 1]][0]
            t = (desired_dist - dists[ix]) / (dists[ix + 1] - dists[ix])
            r_contour.append((1 - t) * contour[ix % l] + t * contour[(ix + 1) % l])
        contour = r_contour[:]
    return np.array(contour).astype(np.int32)


def filter_by_color(frame, color):
    """
    Get the mask of the designated RGB color in input frame

    :param frame: 3-channel RGB image
    :param color: RGB color code
    :return: 1-channel mask
    """
    upper = np.clip(np.array(color) + 10, 0, 255)[::-1]
    lower = np.clip(np.array(color) - 10, 0, 255)[::-1]
    """Create a mask for filtered color"""
    output_mask = cv2.inRange(frame, lower, upper)
    return output_mask


class App:
    """
    User Interface for making annotation
    """

    def __init__(self, visualize=False):
        self.img_idx = 1
        self.width = 0
        self.height = 0
        self.contours = dict(zip(LABELS.keys(), [] * len(LABELS)))
        self.original_mask = None
        self.original_image = None
        self.visualize = visualize

        if visualize:
            cv2.namedWindow("YOLOv8 Prep")

    def __save_contours(self):
        with open(f'{new_anno_path}/{self.img_idx:06d}.txt', 'w') as f:
            for ix, contours in self.contours.items():
                for contour in contours:
                    line = f'{ix}'
                    for pt in contour:
                        line += f' {pt[0][0] / self.width:.6f} {pt[0][1] / self.height:.6f}'
                    f.write(line + '\n')

    def __reset(self):
        # {idx: [cnt[], cnt[], ...]}
        self.contours = dict(zip(LABELS.keys(), [] * len(LABELS)))

    def __visualize(self):
        # Overlay a polygon covering the feature
        # Mixing two weighted images: one original, one with overlays
        lower_layer = deepcopy(self.original_image)
        higher_layer = deepcopy(self.original_image)
        for ix, contours in self.contours.items():
            for contour in contours:
                # Filled contour on top layer
                higher_layer = cv2.drawContours(higher_layer, [contour], -1, color=LABELS[ix][::-1], thickness=-1)
                # Contour with visible red edge on both layers
                higher_layer = cv2.drawContours(higher_layer, [contour], -1, color=(0, 0, 255), thickness=2)
                lower_layer = cv2.drawContours(lower_layer, [contour], -1, color=(0, 0, 255), thickness=2)
        # Mix/merge both layers
        merged_image = cv2.addWeighted(higher_layer, 0.4, lower_layer, 0.6, 0)
        cv2.imshow("YOLOv8 Prep", merged_image)
        cv2.imwrite(f'{new_mask_path}/{self.img_idx:06d}.jpg', merged_image)
        # while True:
        #     k = cv2.waitKey(40) # delay in ms
        #         if k == ord('n'):
        #             break

    def __find_contours(self):
        """Finds contours based on the mask image and stores coordinates in a list"""

        for ix, color in LABELS.items():
            mask = filter_by_color(self.original_mask, color)   # 1-channel image
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)     # list of 3-dimension arrays
            self.contours[ix] = contours

    def run(self):
        """Main loop of the program"""

        # (Re)create the output directories
        try:
            shutil.rmtree(new_image_path)
            shutil.rmtree(new_anno_path)
        except FileNotFoundError:
            pass
        finally:
            os.makedirs(new_image_path)
            os.makedirs(new_anno_path)

        # Recursively access to the images
        # Must align with hierarchy of the input directories
        for subfolder_processed in os.listdir(original_path):
            # bad_ and good_ folders
            processed_path = f'{original_path}/{subfolder_processed}'
            for subfolder_index in os.listdir(processed_path):
                # ID folders
                index_path = f'{processed_path}/{subfolder_index}'
                for image in os.listdir(index_path):
                    # Open image and save to a new location for future shuffling
                    image_path = f'{index_path}/{image}'
                    self.original_image = cv2.imread(image_path)
                    cv2.imwrite(f'{new_image_path}/{self.img_idx:06d}.jpg',
                                self.original_image)

                    # Change directory to the mask
                    temp_path = image_path.split('/')
                    temp_path[-5] = 'SegmentationClass'
                    mask_path = '/'.join(temp_path)

                    # Check if mask exists
                    if os.path.isfile(mask_path):
                        # Get the dimensions then make annotations
                        self.original_mask = cv2.imread(f'{mask_path}')
                        self.height, self.width = self.original_mask.shape[:2]
                        self.__find_contours()
                        self.__save_contours()

                    # Visualize if initialized
                    if self.visualize:
                        self.__visualize()
                        print(f"Showing #{self.img_idx:06d} {image_path}", end="\r")
                    else:
                        print(f"Screened #{self.img_idx:06d} {image_path}", end="\r")

                    # Reset the temporary recognized masks
                    self.__reset()
                    self.img_idx += 1

        print()


if __name__ == '__main__':
    app = App(visualize=True)
    app.run()
