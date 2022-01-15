import matplotlib 
import matplotlib.pyplot as plt 
import numpy as np
import cv2 
import re 
import pickle
import skimage.io as skio
import skimage.draw as draw

def save(img, name):
    skio.imsave(name, img)


# PART 1. ANNOTATIONS FOR IMAGES 

def define_points(img, points_num, img_name): 
    # Manual selection of points 
    points = []
    plt.imshow(img)
    points = plt.ginput(points_num, 0)
    plt.close()

    # Saving points file in the folder called "points"
    pickle_name = re.split("\.", img_name)[0] + ".points"
    pickle.dump(points, open("points/" + pickle_name, "wb"))
    return points


def load_points(img_name):
    # Retrieving points file of the image_name from the "points" folder 
    points = pickle.load(open("points/" + img_name + ".points", "rb"))
    return np.array(points)




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


# PART 3. IMAGE WARPING & MOSAICING   

# Since homography only relies to the available information of the original image, the projection 
# might stretch certain elements in the image to compensate for the missing information. This can be done through a set 
# of transformation operations of the image (stretching to the left / right of the corners) 
# Warping will try to stretch the destiny image into the original image, using the homography H matrix. 


#HELPER FUNCTIONS 
def boundary_box(img2, H):
    # Defining image corners after warping 
    shape = img2.shape 
    y_max = shape[0]
    x_max = shape[1] 

    edge_btm_left = np.array([[0], [y_max], [1]]) 
    edge_top_left = np.array([[0], [0], [1]]) 
    edge_btm_right = np.array([[x_max], [y_max], [1]]) 
    edge_top_right = np.array([[x_max], [0], [1]]) 

    edges = [edge_btm_left, edge_btm_right, edge_top_left, edge_top_right]
    edges = [H @ point for point in edges] 
    edges = [point / point[2] for point in edges] 
    return edges 

def draw_mask(right_edge,left_edge, bottom_edge, top_edge): 
    #Drawing edges 
    mask = draw.polygon([0, right_edge + abs(left_edge), right_edge + abs(left_edge), 0],
                        [0, 0, bottom_edge + abs(top_edge), bottom_edge + abs(top_edge)]) 
    mask = np.matrix(np.vstack([mask, np.ones(len(mask[0]))]))
    return mask 

def alpha_mask(img1): 
    #For blending in warping 
    pi = 3.141592628/2 
    alpha = np.cos(np.linspace(0, pi, int(img1.shape[1]/2))) ** 8
    alpha = np.hstack([np.ones(int(img1.shape[1]/2), dtype="float64"), alpha])
    maskAlpha = alpha
    for _ in range(img1.shape[0] - 1):
        maskAlpha = np.vstack([maskAlpha, alpha])
    gray = maskAlpha.reshape((maskAlpha.shape[0], maskAlpha.shape[1], 1))
    maskAlpha = np.dstack([gray, gray, gray]) 
    return maskAlpha 


#WARPING 
def warpImage(img1, img2, H):
    edges = boundary_box(img2, H)

    #STRETCHING 
    # Images stretched to the available corners (left / right) that are defined by the above function boundary_box 
    transformed_x_max = max(edges, key=lambda x: x[0])[0].astype(np.int)
    transformed_x_min = min(edges, key=lambda x: x[0])[0].astype(np.int)

    transformed_y_max = max(edges, key=lambda x: x[1])[1].astype(np.int)
    transformed_y_min = min(edges, key=lambda x: x[1])[1].astype(np.int)

    bottom_edge, right_edge, top_edge, left_edge = [transformed_y_max[0, 0], transformed_x_max[0, 0], transformed_y_min[0, 0], transformed_x_min[0, 0]]

    right_edge = max(right_edge, img1.shape[1], img2.shape[1])
    bottom_edge = max(bottom_edge, img1.shape[0], img2.shape[0])

    # MASK 
    mask = draw_mask(right_edge,left_edge, bottom_edge, top_edge)  
    mask_inv = np.linalg.inv(H) @ mask   
    colmask, rowmask, w = mask_inv
    colmask = np.squeeze(np.asarray(colmask))
    rowmask = np.squeeze(np.asarray(rowmask))
    w = np.squeeze(np.asarray(w))
    colmask = (colmask / w).astype(np.int)
    rowmask = (rowmask / w).astype(np.int)

    #OVERLAP MOSAICING 
    # To avoid repeating the warping on the overlapping images  
    final_img = np.zeros((bottom_edge + abs(top_edge) + 1, right_edge + abs(left_edge) + 1, 3), dtype="uint8")
    overlap = np.where((colmask >= 0) & (colmask < img2.shape[1]) & (rowmask >= 0) & (rowmask < img2.shape[0]))
    rowmask = rowmask[overlap] 
    colmask = colmask[overlap]
    

    x_init, y_init, _ = draw_mask(right_edge,left_edge, bottom_edge, top_edge) 
    x_init = np.squeeze(np.asarray(x_init))
    x_init = x_init[overlap].astype(np.int) 

    #Offset X 
    diff_x = abs(min(left_edge, 0)) 
    x_init += diff_x 
    img1_x = img1.shape[1] + diff_x 

    y_init = np.squeeze(np.asarray(y_init))
    y_init = y_init[overlap].astype(np.int)

    #Offset Y
    diff_y = abs(min(top_edge, 0))
    y_init += diff_y
    img1_y = img1.shape[0] + diff_y

    #Placing images together 
    final_img[y_init, x_init] = img2[rowmask, colmask]
    final_img[diff_y : img1_y, diff_x : img1_x] = img1

    # BLENDING 
    maskAlpha = alpha_mask(img1) 
    alpha_img1 = img1 * maskAlpha
    final_img[diff_y : img1_y, diff_x : img1_x] = alpha_img1 * maskAlpha + final_img[diff_y : img1_y, diff_x : img1_x] * (1 - maskAlpha)
    
    return final_img


#MOSAICING 
def warp(img1, img2, points = True):
    #Read images 
    left_img = skio.imread("img/" + img1 + ".jpg") 
    right_img = skio.imread("img/" + img2 + ".jpg") 

    #Load / define points 
    if points is False:
        define_points(left_img, 4, img1)
        define_points(right_img, 4, img2)
    left_pts = load_points(img1)
    right_pts = load_points(img2)

    #Compute homography 
    H = computeH(left_pts, right_pts)
    warp = warpImage(left_img, right_img, H)

    #Show final result 
    skio.imshow(warp)
    skio.show()   




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



def rectify(img, points = True):
    #Read image 
    image = skio.imread("img/" + img + ".jpg") 

    #Load points 
    if points is False:
        define_points(image, 4, img)
    img_pts = load_points(img)
 
    #Squared Rectification 
    min_dim = min(image.shape[0], image.shape[1])
    square = [[0, 0], [0, min_dim], [min_dim, min_dim], [min_dim, 0]]
    H = computeH(square, img_pts)

    #Compute Rectification 
    rectified = rectifyImage(image, img_pts, H)
    skio.imshow(rectified)
    skio.show() 

#warp("swiss_mos1", "swiss_mos2", points = False)  
#warp("syria2_mos1", "syria2_mos2", points = False) 
#warp("syr_mos1", "syr_mos2", points = False)
#rectify("syria5", points = False)
    