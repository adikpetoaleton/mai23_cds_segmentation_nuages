import numpy as np
import cv2 
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

OUTPUT_SHAPE = (8, 8)
CLASSES = ['Fish', 'Flower', 'Gravel', 'Sugar']

def display_info(txt, color='Black'):
    st.write("<span style='color:" + color + ";'>_" + txt + "_</span>", unsafe_allow_html=True)

def display_info_list_items(items, color='Black'):
    st.write("- " + "\n- ".join(items))



def get_mask_origine(mask):
    
    white_pixels = np.array(np.where(mask == 255))
    first_white_pixel = white_pixels[:,0]
    last_white_pixel = white_pixels[:,-1]
    
    return (first_white_pixel[1], first_white_pixel[0]), (last_white_pixel[1], last_white_pixel[0])

def rle_to_mask(rle_string, width, height):
   
    rows, cols = height, width
    
    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1,2)
        img = np.zeros(rows*cols, dtype=np.uint8)
        for index, length in rle_pairs:
            index -= 1
            img[index:index+length] = 255
        img = img.reshape(cols,rows)
        img = img.T
        return img

def show_ground(image_label, ax, masks, w, h, image_path, hide_axis=False, show_mask=False):

    alpha = 0.2

    filename = image_label.split('_')[0]
    
    image_path = image_path
    img = cv2.imread(image_path + filename)

    if show_mask:
        # Get RLE encoded masks of an image by its image_label and related labels (Flower, Fish...)
        masks_filtered_byId = masks[masks['Image_Label']==image_label]
        img_masks = masks_filtered_byId['EncodedPixels'].tolist()
        img_masks_labels = masks_filtered_byId['Label'].tolist()
    
        # Convert RLE encoded masks into a binary encoded grids
        all_masks = np.zeros((h, w))
        one_mask = np.zeros((h, w))
        mask_origines = []
        for rle_mask in img_masks:
            one_mask = rle_to_mask(rle_mask, w, h)
            mask_origines.append(get_mask_origine(one_mask))
            all_masks += one_mask

    # Displays images and related masks
    if hide_axis:
        ax.axis('off')

    if show_mask:
        # Displays images and related masks
        for origine, label in zip(mask_origines, img_masks_labels):
            ax.annotate(text=label + " 0", xy=origine[0], xytext=(20, -30), xycoords='data', color='yellow', fontsize=10, fontweight='bold', textcoords='offset pixels')
            ax.annotate(text=label + " 1", xy=origine[1], xytext=(-90, 20), xycoords='data', color='yellow', fontsize=10, fontweight='bold', textcoords='offset pixels') 

        cross_size = 75
        cross_0_x = mask_origines[0][0][0]
        cross_0_y = mask_origines[0][0][1]
        cross_1_x = mask_origines[0][1][0]
        cross_1_y = mask_origines[0][1][1]
        
        cross_0_line1 = lines.Line2D([cross_0_x, cross_0_x], [cross_0_y - cross_size, cross_0_y + cross_size], color='y')
        cross_0_line2 = lines.Line2D([cross_0_x - cross_size, cross_0_x + cross_size], [cross_0_y, cross_0_y], color='y')
        cross_1_line1 = lines.Line2D([cross_1_x, cross_1_x], [cross_1_y - cross_size, cross_1_y + cross_size], color='y')
        cross_1_line2 = lines.Line2D([cross_1_x - cross_size, cross_1_x + cross_size], [cross_1_y, cross_1_y], color='y')
    
        # # Add the cross lines to the plot
        ax.add_line(cross_0_line1)
        ax.add_line(cross_0_line2)
        ax.add_line(cross_1_line1)
        ax.add_line(cross_1_line2)
        ###
    
    ax.set_title(image_label)
    
    ax.imshow(img)

    if show_mask:
        ax.imshow(all_masks, alpha=alpha)

def show_bounding_box(im, bbox, class_pred, ax, normalised=True, color='r'):
    
    # Signification de bbox
    x, y, w, h, fish, flower, gravel, sugar = bbox
    
    # Convertir les cordonées (x,y,w,h) en (x1,x2,y1,y2)
    x1 = x - w / 2
    x2 = x + w / 2
    y1 = y - h / 2
    y2 = y + h / 2
    
    # redimensionner en cas de normalisation
    if normalised:
        x1 = x1 * im.shape[1]
        x2 = x2 * im.shape[1]
        y1 = y1 * im.shape[0]
        y2 = y2 * im.shape[0]

    delta_x = 2
    delta_y = 7
    if y1 - delta_y < 0:
        delta_y_resolved = y1 + delta_y
    else:
        delta_y_resolved = y1 - delta_y

    ax.text(x1 + delta_x, delta_y_resolved, class_pred, fontsize=10, ha='left', va='center', bbox=dict(boxstyle='square', alpha=0.8, facecolor='orange', edgecolor='none'))
    
    # Afficher l'image
    ax.set_title(class_pred)
    ax.imshow(im)
    
    # Afficher la bounding box
    ax.plot([x1, x2, x2, x1, x1],[y1, y1, y2, y2, y1], 'orange')

def generate_yolo_grid(g):
    
    c_x = tf.cast(tf.reshape(tf.tile(tf.range(g), [g]), (1, g, g)), 'float32')
    c_y = tf.transpose(c_x, (0,2,1))
    
    return tf.stack([tf.reshape(c_x, (-1, g*g)), tf.reshape(c_y, (-1, g*g))] , -1)


c_grid = generate_yolo_grid(OUTPUT_SHAPE[0])

def proccess_xy(y_true_raw):

    y_true_conf = y_true_raw[..., :1]
    y_true_xy = ((y_true_raw[..., 1:3] + 1) / 2 + c_grid) / OUTPUT_SHAPE[0]
    y_true_wh = y_true_raw[..., 3:5]
    y_true_class = y_true_raw[..., 5:]
    
    return tf.concat([y_true_conf, y_true_xy, y_true_wh, y_true_class], -1)

def pred_bboxes(y, threshold=0.3):
    
    y_xy = tf.cast(y, tf.float32)
    y_xy = tf.expand_dims(y_xy, axis=0)
    y_xy = proccess_xy(y_xy)[0]
    
    bboxes =  sorted(y_xy.numpy(), key=lambda x: x[0], reverse=True)
    bboxes = np.array(bboxes)
    result = bboxes[bboxes[:,0] > threshold]
    
    if len(result)== 0:
        return bboxes[[0]]
        
    return result

def transform_netout(y_pred_raw):

    y_pred_conf = tf.sigmoid(y_pred_raw[..., :1])

    y_pred_xy = (tf.nn.tanh(y_pred_raw[..., 1:3]))
    
    y_pred_wh = tf.sigmoid(y_pred_raw[..., 3:5])

    y_pred_class = tf.nn.softmax(y_pred_raw[..., 5:])    
    #y_pred_class = tf.sigmoid(y_pred_raw[..., 5:])    

    return tf.concat([y_pred_conf, y_pred_xy, y_pred_wh, y_pred_class], -1)

def show_prediction(img, model, ax, threshold=0.2):
    
    pred = model(np.array([img], dtype=np.float32))[0]

    pred = transform_netout(pred)

    bboxes_pred = pred_bboxes(pred, threshold)
    
    class_pred = CLASSES[np.argmax(bboxes_pred[0, 5:])]
    box_prob = round(bboxes_pred[0, 0] * 100, 2)
    class_prob = round(max(bboxes_pred[0, 5:]) * 100, 2)

    plot_title = class_pred + ": " + str(class_prob) + " % - Bbox: " + str(box_prob) + " %"
    
    for bbox in bboxes_pred:
        bbox = bbox[1:]
        show_bounding_box(img/255, bbox, plot_title, ax)
    
    return bboxes_pred