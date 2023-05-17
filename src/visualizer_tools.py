import numpy as np
import matplotlib.pyplot as plt

from src.exception import CustomException
from src.logger import logging


class PlottingUtils:
    
    def __init__(self):
        pass

    def show_mask(self, mask, ax, random_color=False):
        try:
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)
            
            logging.info('Mask on a given axis in an image added succesfully !')
            
        except Exception as e:
            raise CustomException(e,sys)

    def show_points(self, coords, labels, ax, marker_size=375):
        try:
            pos_points = coords[labels==1]
            neg_points = coords[labels==0]
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
            
            logging.info('Points with labels on a given axis in an image added succesfully !')
            
        except Exception as e:
            raise CustomException(e,sys)

    def show_box(self, box, ax):
        try:
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
            
            logging.info('bounding box on a given axis in an image added succesfully !')
            
        except Exception as e:
            raise CustomException(e,sys)

    def show_anns(self, anns):
        try:
            if len(anns) == 0:
                return
            sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in sorted_anns:
                m = ann['segmentation']
                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.random.random((1, 3)).tolist()[0]
                for i in range(3):
                    img[:,:,i] = color_mask[i]
                ax.imshow(np.dstack((img, m*0.35)))
                
            logging.info('Annotations on a plot added succesfully !')
            
        except Exception as e:
            raise CustomException(e,sys)