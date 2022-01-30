import matplotlib 
import matplotlib.pyplot as plt 
import numpy as np
import re 
import pickle
import skimage.io as skio
import skimage.draw as draw

def save(img, name):
    skio.imsave(name, img)


# PART 2. RECOVERING HOMOGRAPHY 

def coefficient_matrix_row(p1, p2): 
    #Credit to https://cseweb.ucsd.edu//classes/wi07/cse252a/homography_estimation/homography_estimation.pdf 
    first = np.array([p2[0], p2[1], 1,0,0,0, -p2[0]*p1[0], -p2[1]*p1[0]])
    second = np.array([0,0,0, p2[0], p2[1], 1, -p2[0]*p1[1], -p2[1]*p1[1]])
    point_matrix = np.vstack([first, second]) 
    return point_matrix 



def computeH(im1_pts,im2_pts): 

    # Constructing coefficient matrix  
    k = len(im1_pts)
    i = 0 
    out_mat = np.zeros((k*2, 1))
    for i in range(k): 
       out_mat[i*2] = im1_pts[i][0]
       out_mat[i*2+1] = im1_pts[i][1]
    A = coefficient_matrix_row(im1_pts[0], im2_pts[0]) 
    for i in range(1, k): 
        coeff_row = (coefficient_matrix_row(im1_pts[i], im2_pts[i]))
        A = np.vstack([A, coeff_row])

    # Least squares to calculate homography matrix 
    H_arr = (np.linalg.lstsq(A, out_mat, rcond=-1)[0]).T[0]
    H = np.matrix([[H_arr[0], H_arr[1], H_arr[2]],
                   [H_arr[3], H_arr[4], H_arr[5]],
                   [H_arr[6], H_arr[7], 1.]])
    return H 



def draw_mask(right_edge,left_edge, bottom_edge, top_edge): 
    #Drawing edges 
    mask = draw.polygon([0, right_edge + abs(left_edge), right_edge + abs(left_edge), 0],
                        [0, 0, bottom_edge + abs(top_edge), bottom_edge + abs(top_edge)]) 
    mask = np.matrix(np.vstack([mask, np.ones(len(mask[0]))]))
    return mask 




# PART 4. RECTIFICATION 
#Image rectification is a transformation process used to project multiple images onto a common image surface. 
#It is used to correct a distorted image into a standard coordinate system.  

#HELPER FUNCTIONS 
def boundary_box_rectify(img, img_pts, H): 
    shape = img.shape 
    max_x, max_y = shape[1],shape[0] 

    
    edge_btm_left = np.matrix([[0], [max_y], [1]])
    edge_btm_right = np.matrix([[max_x], [max_y], [1]])
    edge_top_left = np.matrix([[0], [0], [1]])
    edge_top_right = np.matrix([[max_x], [0], [1]])

    edges = [edge_btm_left, edge_btm_right, edge_top_left, edge_top_right]
    edges = [np.matrix([[img_pt[0]], [img_pt[1]], [1]]) for img_pt in img_pts]
    edges = [H @ point for point in edges]
    edges = [point / point[2] for point in edges] 

    return edges 


# RECTIFICATION ALGORITHM  

def rectifyImage(img, img_pts, H):

    edges = boundary_box_rectify(img, img_pts, H)  

    #STRETCHING 
    transformed_x_max = max(edges, key=lambda x: x[0])[0].astype(np.int)
    transformed_y_max = max(edges, key=lambda x: x[1])[1].astype(np.int)

    transformed_x_min = min(edges, key=lambda x: x[0])[0].astype(np.int)
    transformed_y_min = min(edges, key=lambda x: x[1])[1].astype(np.int)
    
    bottom_edge, right_edge, top_edge, left_edge = [transformed_y_max[0, 0], transformed_x_max[0, 0], transformed_y_min[0, 0], transformed_x_min[0, 0]]

    right_edge = max(right_edge, img.shape[1])
    bottom_edge = max(bottom_edge, img.shape[0])

    #MASKING 
    mask = draw_mask(right_edge,left_edge, bottom_edge, top_edge)  
    mask_inv = np.linalg.inv(H) @ mask


    colmask, rowmask, w = mask_inv
    colmask = np.squeeze(np.asarray(colmask))
    rowmask = np.squeeze(np.asarray(rowmask))
    w = np.squeeze(np.asarray(w))
    colmask = (colmask / w).astype(np.int)
    rowmask = (rowmask / w).astype(np.int)
    final_img = np.zeros((bottom_edge + abs(top_edge) + 1, right_edge + abs(left_edge) + 1, 3), dtype="uint8")

    #OVERLAP 
    overlap = np.where((colmask >= 0) & (colmask < img.shape[1]) & (rowmask >= 0) & (rowmask < img.shape[0]))
    colmask = colmask[overlap]
    rowmask = rowmask[overlap]

    x_init, y_init, _ = mask
    x_init = np.squeeze(np.asarray(x_init))
    x_init = x_init[overlap].astype(np.int)

    y_init = np.squeeze(np.asarray(y_init)) 
    y_init = y_init[overlap].astype(np.int)

    final_img[y_init, x_init] = img[rowmask, colmask]

    return final_img



def rectify(img_name, points):
    #Read image 
    image = skio.imread(img_name) 

    #Load points, points is a list of tuples 
    img_pts = points 

    #Squared Rectification 
    min_dim = min(image.shape[0], image.shape[1])
    square = [[0, 0], [0, min_dim], [min_dim, min_dim], [min_dim, 0]]
    H = computeH(square, img_pts)

    #Compute Rectification 
    rectified = rectifyImage(image, img_pts, H)
    save(rectified, img_name)


# TO-DO: Handle one-channel images 

# TO-DO: Handle other formats: png, jpeg 