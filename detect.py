''' Dawei Zhang - daweizhang009@gmail.com
    Solve flash game 'BloomingGarden' @ https://www.miniclip.com/games/bloomin-gardens/en
    ----------
    File detect.py - detect game window, return game status
'''

import numpy as np
import mss
import cv2

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
    return img[vs:vp+1,hs:hp+1,:]

def detect_wnd():
    ''' detect game window, return status
    '''
    # capture game window, open https://www.miniclip.com/games/bloomin-gardens/en
    # scroll down to game widget on top (the "Games > Puzzle > Bloomin'Gardens" on top)
    monitor = {"top": 180, "left": 300, "width": 800, "height": 520}
    sct = mss.mss()
    img = np.array(sct.grab(monitor))
    img = remove_brd(img, 255)
    assert img.shape == (500, 738, 4), 'Instruction: max https://www.miniclip.com/games/bloomin-gardens/en/ and run!'
    #cv2.imwrite("capture.png", img)

    # projective transform to get a square board
    src = np.array([(143,106), (25,456), (591,106), (709,456)]) # lt, lb, rt, rb
    dst = np.array([(59,100), (59,784), (743,100), (743,784)]) # lt, lb, rt, rb
    H, _ = cv2.findHomography(src, dst)
    img_dst = cv2.warpPerspective(img, H, (804, 804))
    ptLT = (64,106); delta = 75
    img_Hsv = cv2.cvtColor(img_dst, cv2.COLOR_BGR2HSV)
    grid = np.zeros((9,9)); ddv, duv, dh = 10, 4, delta
    for i in range(9):
        cv2.line(img_dst,(ptLT[0], ptLT[1]+i*delta), (ptLT[0]+9*delta, ptLT[1]+i*delta), (0,255,0), 2)
        cv2.line(img_dst,(ptLT[0]+i*delta, ptLT[1]), (ptLT[0]+i*delta, ptLT[1]+9*delta), (0,255,0), 2)
    #     for j in range(9):
    #         dvs, dvp = ptLT[1]+i*delta-duv, ptLT[1]+i*delta+ddv
    #         dhs, dhp = ptLT[0]+j*delta, ptLT[0]+j*delta+dh
    #         grid[i][j] = np.mean(img_Hsv[dvs:dvp,dhs:dhp], axis=(0,1))[0]
    #         cv2.imwrite(f"flower_{i}_{j}.png", img_dst[dvs:dvp,dhs:dhp])
    # for rval in grid:
    #     print([''.join(f'{x:06.2f}') for x in rval])
    cv2.imwrite("square.png", img_dst)


    # gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # sift = cv2.xfeatures2d.SIFT_create()
    # kp_1, des_1 = sift.detectAndCompute(gray1, None)
    # # Test code below
    # img_sv = cv2.drawKeypoints(gray1, kp_1, img)
    # cv2.imwrite('sift_keypoints.jpg', img_sv)
    # cv2.imwrite("capture.png", img)

    #board = [(143,106), (25,456), (591,124), (709,456)] # lt, lb, rt, rb

if __name__ == '__main__':
    detect_wnd()