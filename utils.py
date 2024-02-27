import numpy as np
import cv2 
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Convert RLE string to a numpy array
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

# Returns the pixels that delimit the edges of a region in a mask
def get_mask_origine(mask):
    
    white_pixels = np.array(np.where(mask == 255))
    first_white_pixel = white_pixels[:,0]
    last_white_pixel = white_pixels[:,-1]
    
    return (first_white_pixel[1], first_white_pixel[0]), (last_white_pixel[1], last_white_pixel[0])

# Displays one mask on top of its originate image
def displayMasks(imageid, ax, masks, w, h, cmap, alpha, image_path):
    
    img_id = imageid.split('_')[0]
    
    image_path = image_path
    img = cv2.imread(image_path + img_id + '.jpg')
    
    # Get RLE encoded masks of an image by its imageid and related labels (Flower, Fish...)
    masks_filtered_byId = masks[masks['ImageId']==imageid]
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
    ax.axis('off')

    for origine, label in zip(mask_origines, img_masks_labels):
        ax.annotate(text=label + " 0", xy=origine[0], xytext=(20, -40), xycoords='data', color='yellow', fontsize=10, fontweight='bold', textcoords='offset pixels', arrowprops=dict(arrowstyle="-|>", color='yellow'))
        ax.annotate(text=label + " 1", xy=origine[1], xytext=(-100, 20), xycoords='data', color='yellow', fontsize=10, fontweight='bold', textcoords='offset pixels', arrowprops=dict(arrowstyle="-|>", color='yellow')) 

    ax.set_title(imageid)
    
    ax.imshow(img)
    ax.imshow(all_masks, cmap=cmap, alpha=alpha)

def get_single_image_bounding_box(data, imageid, image_path, img_width, img_height, resize, pixels_count):
    
    img_id = imageid.split('_')[0]
    
    img = cv2.imread(image_path + img_id + '.jpg')
    
    # Get RLE encoded masks of an image by its img_id and related labels (Flower, Fish...)
    mask_filtered_byId = data[data['ImageId']==imageid]
    img_mask = mask_filtered_byId['EncodedPixels']
    img_mask = img_mask.values[0]
    img_mask_label = mask_filtered_byId['Label']
    
    # Convert one RLE encoded mask into a binary encoded grid
    one_mask = np.zeros((img_height, img_width))
    one_mask = rle_to_mask(img_mask, img_width, img_height)
    
    # Reduce Mask size
    one_mask = cv2.resize(one_mask, dsize=resize)
    
    one_mask_pixels_count = np.count_nonzero(one_mask == 255)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(one_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tmp_edges = []
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        tmp_edges.append({'left':x, 'top':y, 'width':w, 'height':h})
    
    edges = pd.DataFrame(tmp_edges)
 
    left = edges['left'].min()
    right = edges[['left', 'width']].sum(axis=1).max()
    top = edges['top'].min()
    bottom = edges[['top', 'height']].sum(axis=1).max()
    
    x = (left + right) / 2
    y = (top + bottom) / 2
    w = right - left
    h = bottom - top
    
    bbox = {'X': x, 'Y': y, 'W': w, 'H': h}
    
    resized_img = cv2.imread(image_path + 'small/' + img_id + '.jpg')
    
    return imageid, bbox, one_mask, resized_img, one_mask_pixels_count, w * h

def displayBoundingBox(imageid, ax, x, y, w, h):
   
    img_id = imageid.split('_')[0]
    im = cv2.imread('images/small/' + img_id + '.jpg')

    x1 = x - w/2
    x2 = x + w/2
    y1 = y - h/2
    y2 = y + h/2

    #ax.axis('off')
    ax.set_title(imageid)
    ax.imshow(im)
    ax.plot([x1, x2 ,x2, x1, x1],[y1, y1, y2, y2, y1],'yellow')

def display_info(txt, color='Black'):
    st.write("<span style='color:" + color + ";'>_" + txt + "_</span>", unsafe_allow_html=True)

def markDuplicate(data, group_field, count_field):
    g = pd.DataFrame(data.groupby([group_field]).agg({count_field:'count'}).rename({count_field:'Count'}, axis=1))
    g.reset_index(drop=False, inplace=True)
    l = list(g[g['Count'] > 1]['FileId'])
    data['Multiple'] = data['FileId'].apply(lambda fieldid: True if fieldid in l else False )
    return data

def display_info_list_items(items, color='Black'):
    st.write("- " + "\n- ".join(items))

def displayMask(imageid, ax, masks, w, h, image_path, hide_axis=False, show_mask=False):

    cmap = 'viridis'
    alpha = 0.2

    img_id = imageid.split('_')[0]
    
    img = cv2.imread(image_path + img_id + '.jpg')

    # if show_mask:
    #     # Get RLE encoded masks of an image by its imageid and related labels (Flower, Fish...)
    #     masks_filtered_byId = masks[masks['ImageId']==imageid]
    #     img_masks = masks_filtered_byId['EncodedPixels'].tolist()
    #     img_masks_labels = masks_filtered_byId['Label'].tolist()
    
    #     # Convert RLE encoded masks into a binary encoded grids
    #     all_masks = np.zeros((h, w))
    #     one_mask = np.zeros((h, w))
    #     mask_origines = []
    #     for rle_mask in img_masks:
    #         one_mask = rle_to_mask(rle_mask, w, h)
    #         mask_origines.append(get_mask_origine(one_mask))
    #         all_masks += one_mask

    # # Displays images and related masks
    # if hide_axis:
    #     ax.axis('off')

    # if show_mask:
    #     # Displays images and related masks
    #     for origine, label in zip(mask_origines, img_masks_labels):
    #         ax.annotate(text=label + " 0", xy=origine[0], xytext=(20, -40), xycoords='data', color='yellow', fontsize=10, fontweight='bold', textcoords='offset pixels', arrowprops=dict(arrowstyle="-|>", color='yellow'))
    #         ax.annotate(text=label + " 1", xy=origine[1], xytext=(-100, 20), xycoords='data', color='yellow', fontsize=10, fontweight='bold', textcoords='offset pixels', arrowprops=dict(arrowstyle="-|>", color='yellow')) 

    # ax.set_title(imageid)
    
    # ax.imshow(img)

    # if show_mask:
    #     ax.imshow(all_masks, cmap=cmap, alpha=alpha)

def showImages(ImageIds, grid_x, grid_y, df, img_width, img_height, image_path, hide_axis=True, show_mask=False):
    fig, axes = plt.subplots(grid_x, grid_y, figsize=(20, 10), layout='constrained')
    for axe, img_id in zip(axes.flat, ImageIds):
        displayMask(img_id, axe, df, img_width, img_height, image_path, hide_axis, show_mask)
    st.pyplot(fig)

    

