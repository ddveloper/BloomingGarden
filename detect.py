''' Dawei Zhang - daweizhang009@gmail.com
    Solve flash game 'BloomingGarden' @ https://www.miniclip.com/games/bloomin-gardens/en
    ----------
    File detect.py - detect game window, return game status
'''

import numpy as np
import mss
import cv2
import json
import matplotlib.pyplot as plt
from os import path

def remove_brd(img, mono_clr):
    ''' remove mono-color borders
    '''
    r, c, _ = img.shape
    rm, cm = r//2, c//2
    hs, hp = 0, c
    vs, vp = 0, r
    for i in range(c):
        if np.any(img[rm][i] != mono_clr): hs = i; break
    for i in range(c-1, -1, -1):
        if np.any(img[rm][i] != mono_clr): hp = i; break
    for i in range(r):
        if np.any(img[i][cm] != mono_clr): vs = i; break
    for i in range(r-1, -1, -1):
        if np.any(img[i][cm] != mono_clr): vp = i; break
    return img[vs:vp+1,hs:hp+1,:], hs, vs

centers = np.zeros((8,2))

def decode_flowers(grid):
    ''' decode board content/status by clustering images
        flowers [0-pink, 1-green, 2-yellow, 3-empty, 4-blue
                 5-purple, 6-white, 7-red]
    '''    
    global centers
    m, n, _ = grid.shape
    grid = grid.reshape((m*n,2))
    # define criteria and apply kmeans()
    if np.any(centers != 0): # already loaded
        dists = np.linalg.norm(centers-grid[:,None], axis=-1)
        label = np.argmin(dists, axis=1)
    elif path.exists("centers.txt"):
        centers = np.loadtxt("centers.txt", dtype=int)
        dists = np.linalg.norm(centers-grid[:,None], axis=-1)
        label = np.argmin(dists, axis=1)
    elif m != 9 or n != 9:
        raise "need whole board flowers to re-calculate"
    else: # re-calculate centers
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
        ret,label,centers=cv2.kmeans(grid, 8, None, criteria, 40, cv2.KMEANS_PP_CENTERS)
        np.savetxt("centers.txt", centers, fmt="%d")
    values = label.reshape(m,n); print(values)

    # # Now separate the data, Note the flatten()
    # A = []
    # for i in range(8): A.append(grid[label.ravel() == i])
    # # Plot the data
    # for i in range(8): plt.scatter(A[i][:,0],A[i][:,1])
    # plt.scatter(centers[:,0],centers[:,1],s = 80,c = 'y', marker = 's')
    # plt.xlabel('x'),plt.ylabel('y')
    # plt.show()
    
    return values

def get_contours(img_src):
    grayImage = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)    
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 200, 255, cv2.THRESH_BINARY)
    _, contours, hierachy = cv2.findContours(blackAndWhiteImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def save_contour_img(contour, img, i):
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    # Draw the contours on the mask
    cv2.drawContours(mask, contour, -1, 0, -1)
    # remove the contours from the image and show the resulting images
    img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(f"Mask{i}.png", mask)
    #cv2.imwrite("After.png", img)

def decode_score(img_score):
    shapes = get_contours(img_score)
    shapes = sorted(shapes, key=lambda x: x[0][0][0])
    img_digits = cv2.imread("digits.png")
    ref_shapes = get_contours(img_digits)
    ref_shapes = sorted(ref_shapes, key=lambda x: x[0][0][0])
    # for i, ss in enumerate(shapes): save_contour_img(ss, img_score, i)

    score = 0
    for i, s in enumerate(shapes[::-1]):
        min_j, min_diff = 0, float('inf')
        for j, ss in enumerate(ref_shapes):
            ret = cv2.matchShapes(s, ss, 1, 0.0)
            if ret < min_diff:
                min_diff = ret; min_j = j
        score += min_j * (10 ** i)
    print(f"score={score}")
    return score

def replace_bg_color(img, bg_clr, new_clr):
    dist = np.linalg.norm(img-bg_clr, axis=-1)
    mask = np.where(dist < 20)
    img[mask] = [26.2, 51.25, 75.1]

def detect_wnd():
    ''' detect game window, return status
    '''
    # capture game window, open https://www.miniclip.com/games/bloomin-gardens/en
    # scroll down to game widget on top (the "Games > Puzzle > Bloomin'Gardens" on top)
    monitor = {"top": 180, "left": 300, "width": 800, "height": 520}
    sct = mss.mss()
    img = np.array(sct.grab(monitor))[:,:,:-1]
    img, hs, vs = remove_brd(img, 255)
    assert img.shape == (500, 738, 3), 'Instruction: max https://www.miniclip.com/games/bloomin-gardens/en/ and run!'
    # cv2.imwrite("capture.png", img)

    # get scores from img
    img_score = img[55:100,20:115]
    # cv2.imwrite("score.png", img_score)
    score = decode_score(img_score)
    
    # get next flowers from img
    img_next = [img[70:98,666:714], img[136:164,666:714], img[202:230,666:714]]
    grid_next = np.zeros((3,1,2))
    for i, im in enumerate(img_next):
        # cv2.imwrite(f"next{i}.png", im)
        replace_bg_color(im, [28,87,20], [26.2, 51.25, 75.1]) # bg green
        replace_bg_color(im, [19,73,9], [26.2, 51.25, 75.1]) # bg stripe green
        # cv2.imwrite(f"next{i}_new.png", im)
        next_Ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
        grid_next[i][0] = np.mean(next_Ycc[:,:,1:], axis=(0,1))

    # projective transform to get a square board
    src = np.array([(143,106), (25,456), (591,106), (709,456)]) # lt, lb, rt, rb
    dst = np.array([(59,100), (59,784), (743,100), (743,784)]) # lt, lb, rt, rb
    H, _ = cv2.findHomography(src, dst)
    img_dst = cv2.warpPerspective(img, H, (804, 804))
    ptLT = (64,106); delta = 75
    img_Ycc = cv2.cvtColor(img_dst, cv2.COLOR_BGR2YCR_CB)
    grid = np.zeros((9,9,2),dtype=np.float32)
    ddv, duv, dlh, drh = 10, 10, delta//5, delta*4//5
    for i in range(9):
        # cv2.line(img_dst,(ptLT[0], ptLT[1]+i*delta), (ptLT[0]+9*delta, ptLT[1]+i*delta), (0,255,0), 2)
        # cv2.line(img_dst,(ptLT[0]+i*delta, ptLT[1]), (ptLT[0]+i*delta, ptLT[1]+9*delta), (0,255,0), 2)
        for j in range(9):
            dvs, dvp = ptLT[1]+i*delta-duv, ptLT[1]+i*delta+ddv
            dhs, dhp = ptLT[0]+j*delta+dlh, ptLT[0]+j*delta+drh
            grid[i][j] = np.mean(img_Ycc[dvs:dvp,dhs:dhp,1:], axis=(0,1))
            # cv2.imwrite(f"flower_{i}_{j}.png", img_dst[dvs:dvp,dhs:dhp])
    # for rval in grid: print([f'{x[0]:05.1f},{x[1]:05.1f}' for x in rval]) # print mean numbers
    cv2.imwrite("board.png", img_dst)

    # classify region pixels to be 8 categories (7 flowers + 1 blank)
    values = decode_flowers(grid)

    # classify next flowers
    nexts = decode_flowers(grid_next)

    return score, values, nexts


if __name__ == '__main__':
    detect_wnd()