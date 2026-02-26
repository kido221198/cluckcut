"""
Making YOLOv8 annotations with Grabcut and SAM
Create a UI with 4 screens:
- Screen 0 for original image and interactions for Grabcut
- Screen 1 for SAM-ed image with contours
- Screen 2 for merging contours from SAM and Grabcut to form an instance contour
- Screen 3 for multiple instance contours and box contour
Interfaces:
- Middle click Screen 0: leaving Grabcut marks
- Left click Screen 1: select SAM contours
- Right click Screen 1 (not developed): select SAM contours as exclusive
- Left click Screen 2: select merged contours as instance contour
- Right click Screen 2: remove merged contours
- Middle click Screen 2: select merged contours as box contour
- Right click Screen 3: remove instance/box contours
- Middle click Screen 3: make annotations from selected contours and move to next image
"""


import threading
import time

import numpy as np
import cv2
import os
import imutils
from copy import deepcopy

import torch
from segment_anything import build_sam_vit_h, SamAutomaticMaskGenerator, SamPredictor
from pycocotools import mask as cocomask

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
# original_path = "./captures"
original_path = "./color_image_1200"
# new_anno_path = "./capture_annotation"
new_anno_path = "./annotation_1200"
QUEUE = {}


def grabcut(input_image, mask, rect):
    """
    Find foreground pixel using Grabcut algorithm with polygon

    :param input_image: 3D numpy array
    :param mask: 1D numpy array with values of FOREGROUND, BACKGROUND, PROB_FOREGROUND, PROB_BACKGROUND
    :param rect: tuple of integers, (x, y, width, height)
    :return: numpy array of the biggest contour
    """
    image = input_image.copy()

    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, rect,
                backgroundModel, foregroundModel,
                3, cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    contours = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    if len(contours) > 1: contours = [max(contours, key=cv2.contourArea)]
    return contours


def check_overlap(contour1, contour2, width, height):
    """
    Get IoU of two contours
    Place two white filled contours on black masks and use AND/OR bitwise
    Count pixels of resulting area for quick interpretation
    Can consider using cv2.contourArea() for higher precision

    :param contour1, contour2: numpy arrays of the input contours
    :param width, height: dimensions of the frame
    :return: float value of IoU ratio
    """
    mask1 = np.zeros((height, width, 3), dtype=np.uint8)
    mask2 = np.zeros((height, width, 3), dtype=np.uint8)

    mask1 = cv2.drawContours(mask1, [contour1], -1, WHITE, thickness=cv2.FILLED)
    mask2 = cv2.drawContours(mask2, [contour2], -1, WHITE, thickness=cv2.FILLED)

    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)

    intersection_cnt = filter_by_color(intersection, WHITE)
    intersection_cnt = cv2.countNonZero(intersection_cnt)

    union_cnt = filter_by_color(union, WHITE)
    union_cnt = cv2.countNonZero(union_cnt)

    return intersection_cnt / union_cnt


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
        dists = [0]  # Geodesic(?) Distance from point #0 towards others
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
    upper = np.clip(np.array(color) + 10, 0, 255)[::-1]     # RGB ↔ BGR
    lower = np.clip(np.array(color) - 10, 0, 255)[::-1]
    """Create a mask for filtered color"""
    output_mask = cv2.inRange(frame, lower, upper)
    return output_mask


class SAM:
    """
    Segment Anything Model
    https://github.com/facebookresearch/segment-anything
    """
    def __init__(self, sam_checkpoint="sam_vit_h_4b8939.pth", device="cuda"):
        """
        Create an idle Mask Generator
        :param sam_checkpoint: recommended SAM model by default
        :param device: "cuda", "cuda:0", or "cpu"
        """
        sam = build_sam_vit_h(checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def run(self, img):
        """
        Generate mask with initialized mask generator
        Input frame is resized to (max:640)x(max:640) to avoid overload GPU if necessary

        :param img: 3-channel input frame
        :return: all-resized 3-channel image with contours, contours, 3-channel image, and scaling ratio
        """
        # Resize if necessary
        h, w = img.shape[:2]
        if max(h, w) > 640:
            ratio = 640 / max(h, w)
            img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
        else:
            ratio = 1.0

        # Get masks from SAM
        new_img = deepcopy(img)
        masks = self.mask_generator.generate(img)

        # Generate contours and image with contours
        height, width = img.shape[:2]
        contours = []
        for mask in masks:
            # SAM mask is 1-channel frame with True/False value telling the contour
            # It makes the finding the sequence of on-contour points difficult
            # Reformat contour from mask
            temp_contour = np.column_stack(np.where(mask['segmentation']==True))
            temp_contour = np.array([[[pt[1], pt[0]]] for pt in temp_contour])
            # Get mask of unordered contour
            blank_image = np.zeros((height, width, 3), dtype=np.uint8)
            blank_image = cv2.drawContours(blank_image, temp_contour, -1, WHITE, thickness=-1)
            temp_mask = filter_by_color(blank_image, WHITE)
            temp_contour = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            temp_contour = imutils.grab_contours(temp_contour)
            # Add all found contours into return
            [contours.append(cnt) for cnt in temp_contour]
            # temp_contour = smooth_piece(temp_contour, 3, 2)
            # Draw on resized frame
            img = cv2.drawContours(img, temp_contour, -1, (0, 0, 255), thickness=1)

        return img, contours, new_img, ratio


class App:
    """
    User Interface for making annotation
    """

    def __init__(self):
        cv2.namedWindow("YOLOv8 Prep")
        cv2.setMouseCallback('YOLOv8 Prep', self.__mouse_event)

        self.sam = SAM()    # mask generator
        self.sam_contours = []  # SAM masks to show on Screen 1
        self.original_image = None  # buffer for original image to show on Screen 0
        self.temp_instance_contours = []    # selected SAM masks in instance contour to show on Screen 2
        # self.temp_non_instance_contours = []      # selected SAM masks to exclude from instance contour to show on Screen 2
        self.temp_merged_contour = None   # overlapped selected SAM masks and Grabcut mask and (not developed) exclusive SAM masks
        self.box_contour = None     # formed box contour to show on Screen 3
        self.instance_contours = []     # formed instance contours to show on Screen 3
        self.gc_contour = None  # Grabcut mask
        self.gc_bbox = None     # Grabcut bounding box
        self.screen_0 = None    # Show original image and Grabcut marking
        self.screen_1 = None    # Show SAM-ed image with SAM contours
        self.screen_2 = None    # Show merged contour from SAM and Grabcut
        self.screen_3 = None    # Show formed instance contours and box contour
        self.screens = None     # Stacked screens
        self.click_sequence = []    # Grabcut markers
        self.clicked_screen = -1    # check whether mouse clicks (UP & DOWN) are within same screen

        self.width = 0
        self.height = 0
        self.done = False   # Flag to switch between images
        self.current_id = None  # Image index

        # For drag and drop, not in use
        self.prev_x = 0
        self.prev_y = 0

        # For SAM outputting images in a concurrent thread
        self.queue = {}

    def __mouse_event(self, event, x, y, flags, param):
        if event == 0:
            return

        # # Right click Screen 1
        # # Select smallest SAM contour containing clicked point to exclude from merging instance contours
        # elif event == cv2.EVENT_RBUTTONDOWN and \
        #         flags == cv2.EVENT_FLAG_RBUTTON and \
        #         self.width <= x and y < self.height:
        #     self.clicked_screen = 1
        #
        #     buffer = []
        #     for contour in self.sam_contours:
        #         if cv2.pointPolygonTest(contour, (x - self.width, y), False) > 0:
        #             buffer.append(contour)
        #             print("Right Clicked Screen 1 → Removed SAM contour.")
        #
        #     # Take the smallest contour
        #     if len(buffer) > 0:
        #         self.temp_non_instance_contours.append(min(buffer, key=cv2.contourArea))
        #     # else:
        #     #     self.temp_non_instance_contours.append(buffer[0])
        #     # Update screen 2
        #     self.__visualize_screen_2()

        # Left click Screen 1
        # Select smallest SAM contour containing clicked point to include in merging instance contours
        elif event == cv2.EVENT_LBUTTONDOWN and \
                flags == cv2.EVENT_FLAG_LBUTTON and \
                self.width <= x and y < self.height:
            self.clicked_screen = 1

            buffer = []
            for contour in self.sam_contours:
                if cv2.pointPolygonTest(contour, (x - self.width, y), False) > 0:
                    buffer.append(contour)
                    print("Left Clicked Screen 1 → Added SAM contour.")

            # Take the smallest contour
            if len(buffer) > 0:
                self.temp_instance_contours.append(min(buffer, key=cv2.contourArea))
            # else:
            #     self.temp_instance_contours.append(buffer[0])

            # Update screen 2
            self.__visualize_screen_2()

        # Multiple middle clicks 0
        # Form a mask for Grabcut based on polygon made by clicks
        elif event == cv2.EVENT_MBUTTONDOWN and \
                flags == cv2.EVENT_FLAG_MBUTTON and \
                x < self.width and y < self.height:
            self.clicked_screen = 0
            self.click_sequence.append([x, y])

        elif event == cv2.EVENT_MBUTTONUP and \
                x < self.width and y < self.height and \
                self.clicked_screen == 0:
            # Start processing if more than 3 points are made
            if len(self.click_sequence) > 2:
                contour = np.array([[pt] for pt in self.click_sequence]).astype(np.int32)
                self.gc_bbox = (np.min(contour[:, 0, 0]), np.min(contour[:, 0, 1]),
                                np.max(contour[:, 0, 0]) - np.min(contour[:, 0, 0]),
                                np.max(contour[:, 0, 1]) - np.min(contour[:, 0, 1]))

                # Continue if only Grabcut bounding box is a rectangle
                if self.gc_bbox[2] != 0 and self.gc_bbox[3] != 0:
                    mask = np.zeros((self.height, self.width), dtype=np.uint8)
                    # All pixels outside bounding box is background
                    # All pixels inside bounding box is probably foreground
                    mask[:self.gc_bbox[1], :] = 127
                    mask[self.gc_bbox[1] + self.gc_bbox[3]:, :] = 127
                    mask[:, :self.gc_bbox[0]] = 127
                    mask[:, self.gc_bbox[0] + self.gc_bbox[2]:] = 127
                    mask[mask == 0] = cv2.GC_PR_BGD
                    mask[mask == 127] = cv2.GC_BGD
                    mask[mask == 255] = cv2.GC_FGD

                    # Check pixels inside bounding box and their distances to Grabcut polygon
                    for px in range(self.gc_bbox[0], self.gc_bbox[0] + self.gc_bbox[2] + 1):
                        for py in range(self.gc_bbox[1], self.gc_bbox[1] + self.gc_bbox[3] + 1):
                            dist = cv2.pointPolygonTest(contour, [px, py], True)
                            # Inside more than 5 pixels to polygon → foreground
                            if dist > 5:
                                mask[py, px] = cv2.GC_FGD
                            # Inside less than 5 pixels to polygon → probably foreground
                            elif 0 < dist <= 5:
                                mask[py, px] = cv2.GC_PR_FGD
                            # Outside less than 5 pixels to polygon → probably background
                            elif -5 < dist <= 0:
                                mask[py, px] = cv2.GC_PR_BGD
                            # Outside more than 5 pixels to polygon → background
                            else:
                                mask[py, px] = cv2.GC_BGD

                    # Take image and Grabcut mask to Grabcut algorithm
                    self.gc_contour = grabcut(self.original_image, mask, self.gc_bbox)
                    # Update screen 2
                    self.__visualize_screen_2()
            print("Multiple Left Clicks Screen 0 → Polygon.")

        # Left Clicked Screen 2
        # Take merged contour as a contour of an instance if only clicked precise inside
        # Then reset all the selected SAM contour and Grabcut marks
        elif event == cv2.EVENT_LBUTTONDOWN and \
                flags == cv2.EVENT_FLAG_LBUTTON and \
                x <= self.width and self.height <= y:
            self.clicked_screen = 2
            # Add merged contour if clicked precise
            self.__add_instance(x, y - self.height)
            # Reset Grabcut marks
            self.click_sequence = []
            self.gc_bbox = None
            self.gc_contour = None
            # Reset Screen 2 and Update screen 3
            self.screen_2 = deepcopy(self.original_image)
            self.__visualize_screen_3()
            print("Left Clicked Screen 2 → Add instance.")

        # Middle Clicked Screen 2
        # Take merged contour as a contour of the box if only clicked precise inside
        # Then reset all the selected SAM contour and Grabcut marks
        elif event == cv2.EVENT_MBUTTONDOWN and \
                flags == cv2.EVENT_FLAG_MBUTTON and \
                x <= self.width and self.height <= y:
            self.clicked_screen = 2
            # Add merged contour if clicked precise
            self.__add_box(x, y - self.height)
            # Reset Grabcut marks
            self.click_sequence = []
            self.gc_bbox = None
            self.gc_contour = None
            # Reset Screen 2 and Update screen 3
            self.screen_2 = deepcopy(self.original_image)
            self.__visualize_screen_3()
            print("Left Clicked Screen 2 → Add instance.")

        # Right Clicked Screen 2
        # Remove merged contour and reset all the selected SAM contour and Grabcut marks
        elif event == cv2.EVENT_RBUTTONDOWN and \
                flags == cv2.EVENT_FLAG_RBUTTON and \
                x < self.width and self.height <= y:
            self.clicked_screen = 2
            self.temp_instance_contours = []
            # self.temp_non_instance_contours = []
            self.click_sequence = []
            self.gc_bbox = (0, 0, self.width, self.height)
            self.gc_contour = None
            # Reset Screen 2
            self.screen_2 = deepcopy(self.original_image)
            print("Right Clicked Screen 2 → remove composed masks.")

        # Right Clicked Screen 3
        # Remove smallest formed contour containing clicked point
        elif event == cv2.EVENT_RBUTTONDOWN and \
                flags == cv2.EVENT_FLAG_RBUTTON and \
                self.width <= x and self.height <= y:
            self.clicked_screen = 3
            self.__remove_instance(x - self.width, y - self.height)
            self.click_sequence = []
            # Update Screen 3
            self.__visualize_screen_3()
            print("Right Clicked Screen 3 → remove instance mask.")

        # Middle Clicked Screen 3
        # Save all contours
        elif event == cv2.EVENT_MBUTTONDOWN and \
                flags == cv2.EVENT_FLAG_MBUTTON and \
                self.width <= x and self.height <= y:
            self.clicked_screen = 3
            self.__save_contours()
            print("Middle Clicked Screen 3 → save contours.")

        # Update window by stacking all screens
        if event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_MBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.clicked_screen = -1
            self.__visualize_all_screen()

    def __visualize_screen_2(self):
        """Draw all contour parts of an instance in screen 2"""

        blank_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Included SAM contours as white masks
        for contour in self.temp_instance_contours:
            blank_image = cv2.drawContours(blank_image, contour, -1, WHITE, thickness=cv2.FILLED)
            blank_image = cv2.drawContours(blank_image, contour, -1, WHITE, thickness=3)

        # # Excluded SAM contours as black masks
        # for contour in self.temp_non_instance_contours:
        #     blank_image = cv2.drawContours(blank_image, contour, -1, BLACK, thickness=cv2.FILLED)
        #     blank_image = cv2.drawContours(blank_image, contour, -1, BLACK, thickness=4)

        # Included Grabcut contour as white mask
        blank_image = cv2.drawContours(blank_image, self.gc_contour, -1, WHITE, thickness=cv2.FILLED)
        blank_image = cv2.drawContours(blank_image, self.gc_contour, -1, WHITE, thickness=3)

        # Getting overlapped white mask as instance contour
        merged_mask = filter_by_color(blank_image, WHITE)
        self.temp_merged_contour = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.temp_merged_contour = imutils.grab_contours(self.temp_merged_contour)

        # Draw contour of the largest overlapped white mask
        if len(self.temp_merged_contour) > 0:
            self.temp_merged_contour = max(self.temp_merged_contour, key=cv2.contourArea)
            self.temp_merged_contour = smooth_piece(self.temp_merged_contour, 3, 2)
            self.screen_2 = deepcopy(self.original_image)
            self.screen_2 = cv2.drawContours(self.screen_2, self.temp_merged_contour, -1, (0, 0, 255), thickness=2)
            print(f"Visualized {len(self.temp_instance_contours)} composed masks and GrabCut in Screen 2.")

    def __visualize_screen_3(self, alpha=0.7):
        """Draw contours of all instance in screen 3"""

        back = deepcopy(self.original_image)    # underneath layer as original image
        front = deepcopy(self.original_image)   # top layer as original image with masks

        # Visualize box contour
        if self.box_contour is not None:
            front = cv2.drawContours(self.screen_3, [self.box_contour], -1, (51, 197, 255), thickness=-1)

        # Visualize instance contours
        for contour in self.instance_contours:
            front = cv2.drawContours(self.screen_3, [contour], -1, (255, 0, 0), thickness=-1)

        # Fuse two layers
        self.screen_3 = cv2.addWeighted(back, alpha, front, 1 - alpha, 0)
        print(f"Visualized masks of {len(self.instance_contours)} instances in Screen 3.")

    def __visualize_all_screen(self):
        """Restacking all screen"""

        # Refresh screen 0 with Grabcut marks if exists
        self.screen_0 = deepcopy(self.original_image)

        if len(self.click_sequence) > 0:
            contour = np.array([pt for pt in self.click_sequence]).astype(np.int32)
            self.screen_0 = cv2.polylines(self.screen_0, [contour], True, (255, 0, 0))

        if self.gc_bbox is not None:
            self.screen_0 = cv2.rectangle(self.screen_0, self.gc_bbox, (0, 0, 255), 2)

        # Stacking all screens
        self.screens = np.vstack((np.hstack((self.screen_0, self.screen_1)),
                                  np.hstack((self.screen_2, self.screen_3))))

    def __add_instance(self, x, y):
        """Add merged contour in Screen 2 to Screen 3 as an instance if clicked precisely"""
        print(f"Finding contour wrapping {x} {y}", end=" ... ")
        # Check if clicked inside contour
        if cv2.pointPolygonTest(self.temp_merged_contour, (x, y), False) == 1:
            self.instance_contours.append(deepcopy(self.temp_merged_contour))
            self.temp_merged_contour = None
            self.temp_instance_contours = []
            # self.temp_non_instance_contours = []
            print(f"Instance #{len(self.instance_contours)} added.")
        else:
            print("Not found.")

    def __add_box(self, x, y):
        """Add merged contour in Screen 2 to Screen 3 as box if clicked precisely"""
        print(f"Finding contour wrapping {x} {y}", end=" ... ")
        # Check if clicked inside contour
        if cv2.pointPolygonTest(self.temp_merged_contour, (x, y), False) == 1:
            self.box_contour = deepcopy(self.temp_merged_contour)
            self.temp_merged_contour = None
            self.temp_instance_contours = []
            # self.temp_non_instance_contours = []
            print(f"Box added.")
        else:
            print("Not found.")

    def __remove_instance(self, x, y):
        """Remove contour in Screen 3 containing clicked point"""
        print(f"Removing contour wrapping {x} {y}")
        # Check if any instance contours containing clicked point
        for ix, contour in enumerate(self.instance_contours):
            if cv2.pointPolygonTest(contour, (x, y), False) == 1:
                self.instance_contours.pop(ix)
                return
        # Check if box contour is clicked
        if (self.box_contour is not None and
                cv2.pointPolygonTest(self.box_contour, (x, y), False) == 1):
            self.box_contour = None

    def __save_contours(self):
        """Save all contours shown in Screen 3 and reset all buffers"""
        if len(self.instance_contours) > 0:
            # Save annotations to text file
            # Annotation as {idx} {x1/w} {y1/h} {x2/w} {y2/h} ...
            with open(f'{new_anno_path}/{self.current_id}.txt', 'w') as f:
                for contour in self.instance_contours:
                    line = '0'  # cb
                    for pt in contour:
                        line += f' {pt[0][0] / self.width:.6f} {pt[0][1] / self.height:.6f}'
                    f.write(line + '\n')
                if self.box_contour is not None:
                    line = '1'  # box
                    for pt in self.box_contour:
                        line += f' {pt[0][0] / self.width:.6f} {pt[0][1] / self.height:.6f}'
                    f.write(line + '\n')
        self.__reset()

    def __reset(self):
        """Set all buffers to empty/None for new image"""
        self.original_image = None
        self.temp_instance_contours = []
        # self.temp_non_instance_contours = []
        self.temp_merged_contour = None
        self.instance_contours = []
        self.box_contour = None
        self.click_sequence = []
        self.gc_contour = None
        self.gc_bbox = None
        self.done = True    # Move to next image

    def subthread(self):
        """A thread to SAM in concurrent"""
        i = self.start
        while i < len(QUEUE.keys()):
            try:
                QUEUE[i]['img'], QUEUE[i]['cnts'], QUEUE[i]['original'], QUEUE[i]['ratio'] = self.sam.run(
                    deepcopy(cv2.imread(f'{original_path}/{QUEUE[i]['fname']}')))
                print(f"\u001b[32mFinish SAM image {QUEUE[i]['fname']}\u001b[37m")
            except:
                print(f"\u001b[31mFinish SAM image {QUEUE[i]['fname']}\u001b[37m")
                pass
            finally:
                time.sleep(1)
                i += 1

    def __read_premasked_anno(self):
        """Read already made annotations to continue on the current image"""
        if not os.path.isfile(f'{new_anno_path}/{self.current_id}.txt'): return
        with open(f'{new_anno_path}/{self.current_id}.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Annotation as {idx} {x1/w} {y1/h} {x2/w} {y2/h} ...
                arr = line.split(' ')
                idx = int(arr[0])
                arr = [float(ele) for ele in arr]
                arr = np.array(arr[1:])
                cnt = arr.reshape((int(len(arr) / 2), 1, 2))
                cnt = cnt * np.array([self.width, self.height])
                if idx == 0:    # cb
                    self.instance_contours.append(cnt.astype(int))
                elif idx == 1:  # box
                    self.box_contour = cnt.astype(int)
            # Update screen 3
            self.__visualize_screen_3()

    def run(self, start=0):
        """Start a subthread for SAM and mainthread for UI"""
        # Input all input images to the map for SAM subthread writes to and mainthread reads from
        global QUEUE
        QUEUE = {i: {'fname': n, 'img': None, 'cnts': [], 'original': None, 'ratio': 1.0}
                 for i, n in enumerate(os.listdir(original_path))}
        self.start = start
        threading.Thread(target=self.subthread).start()

        i = 0
        while i < len(QUEUE.keys()):
            # Skip images
            if i < start:
                i += 1
                continue

            # Wait a sec if SAM is not finished
            if QUEUE[i]['img'] is None:
                time.sleep(1)
                continue

            # Get information from finished SAM
            self.sam_contours = QUEUE[i]['cnts']
            self.current_id = QUEUE[i]['fname'].split('.')[0]
            self.original_image = QUEUE[i]['original']

            # Initialize all screens
            self.screen_0 = deepcopy(self.original_image)
            self.screen_1 = QUEUE[i]['img']
            self.screen_2 = deepcopy(self.original_image)
            self.screen_3 = deepcopy(self.original_image)
            self.height, self.width = self.original_image.shape[0:2]
            self.__read_premasked_anno()
            self.gc_bbox = (0, 0, self.width, self.height)
            self.__visualize_all_screen()

            print(f'\u001b[33mShowing image #{self.current_id} at {QUEUE[i]['ratio']:.2f} scaling ratio with '
                  f'{len(self.instance_contours)} instance contours and '
                  f'{0 if self.box_contour is None else 1} box contours\u001b[37m')

            while not self.done:
                cv2.imshow('YOLOv8 Prep', cv2.resize(self.screens, None, fx=1, fy=1))
                key = cv2.waitKey(1)
                if key == 32:   # Spacebar
                    self.__reset()
                    break

            self.done = False   # Reset flag
            i += 1


if __name__ == '__main__':
    torch.cuda.empty_cache()
    app = App()
    app.run(start=0)
