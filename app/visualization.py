from keras_retinanet.utils.colors import label_color

import cv2
import numpy as np

labels_to_names = {0: 'Bookcase', 1: 'Bathtub', 2: 'Pillow', 3: 'Couch', 4: 'Gas stove', 5: 'Washing machine', 6: 'Bed', 
                  7: 'Refrigerator', 8: 'Bathroom accessory', 9: 'Kitchen & dining room table', 10: 'Television', 11: 'Sink', 
                  12: 'Sofa bed', 13: 'Kitchenware', 14: 'Toilet', 15: 'Ceiling fan', 16: 'Microwave oven', 17: 'Furniture', 
                  18: 'Coffeemaker', 19: 'Cupboard', 20: 'Dishwasher', 21: 'Shower', 22: 'Clock', 23: 'Countertop', 
                  24: 'Mug', 25: 'Table'}

def visualize_image(image, box, score, label):
  
    color = label_color(label)
    
    b = np.array(box.astype(int)).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, 2, cv2.LINE_AA)
    
    caption = "{} {:.3f}".format(labels_to_names[label], score)

    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
  
