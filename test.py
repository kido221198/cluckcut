from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
from datetime import datetime
import copy

if __name__ == '__main__':

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("./runs/segment/train1/weights/best.pt")
    im0 = cv2.imread("test_images/000001.jpg")

    names = model.model.names
    ts = datetime.now()
    results = model.predict(im0)
    print("Period:", datetime.now() - ts)
    annotator = Annotator(im0, line_width=1)

    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        for mask, cls in zip(masks, clss):
            color = colors(int(cls), True)
            txt_color = annotator.get_txt_color(color)
            annotator.seg_bbox(mask=mask, mask_color=color, label=names[int(cls)], txt_color=txt_color)

        results[0].show()
        results[0].save(filename="test_images/result.jpg")
