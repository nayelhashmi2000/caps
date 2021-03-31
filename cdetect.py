import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from scipy.spatial import distance as dist
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh,xywh2xyxy, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from torchvision import transforms


class V5:
    def __init__(self , weights):
        # Initialize
        set_logging()
        device = select_device('')
        half = device.type != 'cpu'  # half precision only supported on CUDA
        
        # Load model
        self.model = attempt_load(weights, map_location=device)
        self.txt_path = "runs/detect/exp/labels"

    def detect(self , img1,image_name):
        
        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        trans = transforms.ToTensor()   
        # while True:
        # Run inference
        t0 = time.time()

        # img1 = cv2.imread('data/images/1.jpg')
        img1 = cv2.cvtColor(img1 , cv2.COLOR_BGR2RGB)
        img  = trans(img1)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, 0.3, 0.45, classes=None, agnostic=False)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  
            # print(det)
            # print("im here")
            # gn = torch.tensor(img1.shape)[[1, 0, 1, 0]]  # norma
            if len(det):
                # print("im here now")
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img1.shape).round()
            

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    xywh = (xywh2xyxy(torch.tensor(xywh).view(1, 4))).view(-1).tolist()
                    line = (cls, *xywh, conf)
                    path = "runs/detect/exp3/labels"
                    with open(f"{path}/{image_name}" + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    print('xyxy = ',xyxy , 'conf = ',conf , 'cls = ',cls)
                    label = f'{names[int(cls)]} {conf:.2f}'
                    x1=int(xyxy[0].item())
                    y1=int(xyxy[1].item())
                    x2=int(xyxy[2].item())
                    y2=int(xyxy[3].item())


                    
                    display_str_dict = {'name':names[int(cls)] , 'score':f'{conf:.2f}'}
                    display_str_dict['ymin'] = y1
                    display_str_dict['xmin'] = x1
                    display_str_dict['ymax'] = y2
                    display_str_dict['xmax'] = x2
                    display_str_dict['area'] = (x2 - x1) * (y2 - y1)
                    print(label, xyxy)
                    # cv2.rectangle(img1 , (x1,y1) ,(x2,y2) , (255,0,0),3  )
                    # cv2.putText(img1, f'{label} {conf:.2f}', (x1 , y1-10), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,0,0), 3, cv2.LINE_AA)       
                    plot_one_box(xyxy, img1, label=label, color=colors[int(cls)], line_thickness=3)
            print(f'Done. ({t2 - t1:.3f}s)')
            # cv2.imshow(str('a'), img1)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return img1

                    




if __name__ == '__main__':
    detect()
