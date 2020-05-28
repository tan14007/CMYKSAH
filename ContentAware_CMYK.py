import cv2
import os
import numpy as np
import random
import math
from PIL import Image

# cv::Rect is [x, y, width, height]

def bgr2cmyk(img):
    # Using PIL Image since conversion has boundary cases for many colors
    im = Image.fromarray(np.uint8(img*255)[:,:,::-1])
    cmyk_img = im.convert('CMYK')
    return np.array(cmyk_img)/255

def cmyk2bgr(img):
    bgr_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    bgr_img[:,:,0] = (1-img[:,:,2])*(1-img[:,:,3])
    bgr_img[:,:,1] = (1-img[:,:,1])*(1-img[:,:,3])
    bgr_img[:,:,2] = (1-img[:,:,0])*(1-img[:,:,3])
    return bgr_img

org_img = cv2.imread('./test_halftone.png')
width = org_img.shape[1]
height = org_img.shape[0]
resize_ratio = 1
width = int(width*resize_ratio)
height = int(height*resize_ratio)
org_img = cv2.resize(org_img, (width, height))
org_img = org_img.astype(np.float32)
org_img /= 255
cmyk_img = bgr2cmyk(org_img)


def clamp(value, low, high):
    if value < low:
        return low
    elif value > high:
        return high
    return value

d_table =[
    [13, 0, 5],
    [13, 0, 5],
    [21, 0, 10],
    [7, 0, 4],
    [8, 0, 5],
    [47, 3, 28],
    [23, 3, 13],
    [15, 3, 8],
    [22, 6, 11],
    [43, 15, 20],
    [7, 3, 3],
    [501, 224, 211],
    [249, 116, 103],
    [165, 80, 67],
    [123, 62, 49],
    [489, 256, 191],
    [81, 44, 31],
    [483, 272, 181],
    [60, 35, 22],
    [53, 32, 19],
    [237, 148, 83],
    [471, 304, 161],
    [3, 2, 1],
    [481, 314, 185],
    [354, 226, 155],
    [1389,866, 685],
    [227, 138, 125],
    [267, 158, 163],
    [327, 188, 220],
    [61, 34, 45],
    [627, 338, 505],
    [1227,638, 1075],
    [20, 10, 19],
    [1937,1000,1767],
    [977, 520, 855],
    [657, 360, 551],
    [71, 40, 57],
    [2005,1160,1539],
    [337, 200, 247],
    [2039,1240,1425],
    [257, 160, 171],
    [691, 440, 437],
    [1045,680, 627],
    [301, 200, 171],
    [177, 120, 95],
    [2141,1480,1083],
    [1079,760, 513],
    [725, 520, 323],
    [137, 100, 57],
    [2209,1640,855],
    [53, 40, 19],
    [2243,1720,741],
    [565, 440, 171],
    [759, 600, 209],
    [1147,920, 285],
    [2311,1880,513],
    [97, 80, 19],
    [335, 280, 57],
    [1181,1000,171],
    [793, 680, 95],
    [599, 520, 57],
    [2413,2120,171],
    [405, 360, 19],
    [2447,2200,57],
    [11, 10, 0],
    [158, 151, 3],
    [178, 179, 7],
    [1030,1091,63],
    [248, 277, 21],
    [318, 375, 35],
    [458, 571, 63],
    [878, 1159,147],
    [5, 7, 1],
    [172, 181, 37],
    [97, 76, 22],
    [72, 41, 17],
    [119, 47, 29],
    [4, 1, 1],
    [4, 1, 1],
    [4, 1, 1],
    [4, 1, 1],
    [4, 1, 1],
    [4, 1, 1],
    [4, 1, 1],
    [4, 1, 1],
    [4, 1, 1],
    [65, 18, 17],
    [95, 29, 26],
    [185, 62, 53],
    [30, 11, 9],
    [35, 14, 11],
    [85, 37, 28],
    [55, 26, 19],
    [80, 41, 29],
    [155, 86, 59],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [5, 3, 2],
    [305, 176, 119],
    [155, 86, 59],
    [105, 56, 39],
    [80, 41, 29],
    [65, 32, 23],
    [55, 26, 19],
    [335, 152, 113],
    [85, 37, 28],
    [115, 48, 37],
    [35, 14, 11],
    [355, 136, 109],
    [30, 11, 9],
    [365, 128, 107],
    [185, 62, 53],
    [25, 8, 7],
    [95, 29, 26],
    [385, 112, 103],
    [65, 18, 17],
    [395, 104, 101],
    [4, 1, 1]
]

def d(i, j):
    res = 0
    if (i > 127):
        i = 255 - i
    res = d_table[i][j] / (d_table[i][0] + d_table[i][1] + d_table[i][2])
    return res

def d10(i):
    return d(i,0)

def d11(i):
    return d(i,1)

def d01(i):
    return d(i,2)

def ssim(src_roi, halftone_roi):
    blur_kernel = (11, 11)
    # print("ROI", halftone_roi)
    C1 = 6.5025/255.0/255.0
    C2 = 58.5225/255.0/255.0
    img1 = org_img[src_roi[1]:(src_roi[1]+src_roi[3]), src_roi[0]:(src_roi[0]+src_roi[2]), :]
    img2 = cmyk2bgr(halftone_image[halftone_roi[1]:(halftone_roi[1]+halftone_roi[3]), halftone_roi[0]:(halftone_roi[0]+halftone_roi[2]), :])
    
    img1_sq = img1*img1
    img2_sq = img2*img2
    img1_2 = img1*img2

    mu1 = cv2.GaussianBlur(img1, blur_kernel, 1.5, 1.5)
    mu2 = cv2.GaussianBlur(img2, blur_kernel, 1.5, 1.5)

    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_2 = mu1*mu2

    sigma1 = cv2.GaussianBlur(img1_sq, blur_kernel, 1.5, 1.5)
    sigma2 = cv2.GaussianBlur(img2_sq, blur_kernel, 1.5, 1.5)
    sigma1_2 = cv2.GaussianBlur(img1_2, blur_kernel, 1.5, 1.5)


    sigma1 -= mu1_sq
    sigma2 -= mu2_sq
    sigma1_2 -= mu1_2

    temp1 = mu1_2*2 + C1
    temp2 = sigma1_2*2 + C2
    temp3 = temp1*temp2 

    temp1 = mu1_sq + mu2_sq + C1
    temp2 = sigma1 + sigma2 + C2
    temp1 = temp1 * temp2

    del img1
    del img2
    del img1_sq
    del img2_sq
    del img1_2
    del mu1
    del mu2
    del mu1_sq
    del mu2_sq
    del mu1_2
    del sigma1
    del sigma2
    del sigma1_2
    # print("SSIM_FINISH")
    # Return SSIM img
    return temp3/temp1


def objectiveFunc(roi):
    wg = .5
    wt = 1 - wg

    blur_range = 11

    lu = (max(0, roi[0] - 2*blur_range), max(0, roi[1] - 2*blur_range)) 
    rd = (min(width, roi[0] + roi[2] + 2*blur_range), min(height, roi[1] + roi[3] + 2*blur_range))
    offset1 = [blur_range, blur_range]

    if ( roi[0] - blur_range < 0):
        offset1[0] = 0
    elif (roi[0] - blur_range*2 < 0):
        offset1[0] = roi[0] - blur_range
    
    if ( roi[1] - blur_range < 0):
        offset1[1] = 0
    elif (roi[1] - blur_range*2 < 0):
        offset1[1] = roi[1] - blur_range

    new_roi = (lu[0], lu[1], rd[0]-lu[0], rd[1]-lu[1])

    src_roi = new_roi
    halftone_roi = new_roi
    # src_roi = [new_roi[0], new_roi[1], new_roi[0]+new_roi[2], new_roi[1],new_roi[1]+new_roi[3]]
    # halftone_roi = [new_roi[0],new_roi[0]+new_roi[2], new_roi[1],new_roi[1]+new_roi[3]]

    sub_roi = ((offset1[0], offset1[1]), min(roi[2] + 2*blur_range, new_roi[2] - offset1[0]), min(roi[3] + 2*blur_range, new_roi[3] - offset1[1]))
    ssim_map = ssim(new_roi, new_roi)

    mean_ssim = np.mean(ssim_map)
    gI = cv2.GaussianBlur(org_img[src_roi[1]:src_roi[1]+src_roi[3], src_roi[0]:src_roi[0]+src_roi[2], :], (blur_range, blur_range), 0)
    gH = cv2.GaussianBlur(cmyk2bgr(halftone_image[halftone_roi[1]:halftone_roi[1]+halftone_roi[3], halftone_roi[0]:halftone_roi[0]+halftone_roi[2], :]), (blur_range, blur_range), 0)
    se = gI - gH
    se *= se

    gaussian_diff = np.mean(se)

    return wg*gaussian_diff + wt*(1.0 - mean_ssim)

def ostromoukhovHalftone(img):
    n_pix = img.shape[0]*img.shape[1]
    vd = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            vd.append(img[i,j])
    vd.sort(reverse=True)
    avg_lum = np.mean(img)
    idx = int(round(avg_lum*n_pix))
    thres = vd[idx]
    
    temp = np.copy(img)
    res = np.zeros(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            err = 0
            val = temp[i,j]
            if(val > avg_lum):
                res[i,j] = 1
                err = val - 1
            else:
                res[i,j] = 0
                err = val
            level = int(round(val*255))
            level = clamp(level, 0, 255)
            if(j + 1 < width):
                temp[i, j+1] += err*d10(level)
            if(i + 1 < height and j - 1 >=0):
                temp[i + 1, j - 1] += err*d11(level)
            if(i + 1 < height):
                temp[i+ 1 , j] += err*d01(level)
    return res

def computeSAH():
    block_size = 4
    temperature = 0.2
    anneal_factor = 0.8

    best_im = None
    min_e = 9999

    while True:
        for block_i in range(0, height, block_size):
            print("block_i", block_i, height)
            for block_j in range(0, width, block_size):
                b_indices = [[], [], [], []]
                w_indices = [[], [], [], []]

                ii = 0
                for c in range(4):
                    while(ii < block_size and block_i + ii < height):
                        jj = 0
                        while(jj < block_size and block_j + jj < width):
                            i = block_i + ii
                            j = block_j + jj
                            if(halftone_image[i,j,c] > 0):
                                w_indices[c].append((i,j))
                            else:
                                b_indices[c].append((i,j))
                            jj += 1
                        ii += 1

                roi = [block_j, block_i, min(block_size, width-block_j), min(block_size, height-block_i)]
                e_old = objectiveFunc(roi)
                if(e_old < min_e):
                    min_e = e_old
                    best_im = np.copy(halftone_image)
                ex_time = block_size * block_size
                # ex_time = width*height
                k = 0
                while k < ex_time:
                    for c in range(4):
                        if(len(b_indices[c]) == 0 or len(w_indices[c]) == 0):
                            continue
                        rand1 = random.randint(0, len(b_indices[c])-1)
                        rand2 = random.randint(0, len(w_indices[c])-1)
                        ind1 = b_indices[c][rand1]
                        ind2 = w_indices[c][rand2]

                        halftone_image[ind1[0], ind1[1], c] = 1
                        halftone_image[ind2[0], ind2[1], c] = 0

                        e_new = objectiveFunc(roi)
                        dif_e = e_new - e_old
                        rv = random.random() 
                        if(dif_e < 0 or rv < math.exp(-dif_e/temperature*width*height)):
                            e_old = e_new
                            if(e_old < min_e):
                                min_e = e_old
                                best_im = np.copy(halftone_image)
                            print("Sim updated:", e_old, -dif_e/temperature*width*height, "Temp", math.exp(-dif_e/temperature*width*height))
                            b_indices[c][rand1] = ind2
                            w_indices[c][rand2] = ind1
                        else:
                            halftone_image[ind1[0], ind1[1], c] = 0
                            halftone_image[ind2[0], ind2[1], c] = 1
                    # cv2.imshow("Current Result", cmyk2bgr(halftone_image))
                    # cv2.waitKey(1)
                    k += 1
            
        temperature *= anneal_factor
        print("Current Temp", temperature)
        cv2.imshow("Current Result", cmyk2bgr(halftone_image))
        cv2.waitKey(100)
        if(temperature <= 1e-2):
            break
    return best_im

ostromoukhov_res = np.zeros((height, width, 4))
for i in range (4):
    ostromoukhov_res[:,:,i] = ostromoukhovHalftone(cmyk_img[:,:,i])
cv2.imshow("Ostromoukhov", cmyk2bgr(ostromoukhov_res))
cv2.imshow("Original", org_img)
halftone_image = np.copy(ostromoukhov_res)
before_ssim = ssim([0,0,width, height], [0,0,width,height]).mean()
cv2.imwrite("ostromoukhov_result.png", (cmyk2bgr(halftone_image)*255).astype(np.uint8))

best_im = computeSAH()
cv2.imshow("Result", cmyk2bgr(halftone_image))
cv2.imwrite("halftone_result.png", (cmyk2bgr(halftone_image)*255).astype(np.uint8))
print("Before MSSIM", before_ssim)
print("After MSSIM", ssim([0,0,width, height], [0,0,width,height]).mean())
for i in range (4):
    ostromoukhov_res[:,:,i] = ostromoukhovHalftone(cmyk_img[:,:,i])
halftone_image = np.copy(ostromoukhov_res)


halftone_image = np.copy(best_im)
print("Best MSSIM", ssim([0,0,width, height], [0,0,width,height]).mean())
cv2.imshow("Best", (cmyk2bgr(halftone_image)*255).astype(np.uint8))
cv2.waitKey(10000)