''' Dawei Zhang - daweizhang009@gmail.com
    Solve flash game 'BloomingGarden' @ https://www.miniclip.com/games/bloomin-gardens/en
    ----------
    File detect.py - detect game window, return game status
'''

import numpy as np
import mss
import cv2
import matplotlib.pyplot as plt

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

def detect_wnd():
    ''' detect game window, return status
    '''
    # capture game window, open https://www.miniclip.com/games/bloomin-gardens/en
    # scroll down to game widget on top (the "Games > Puzzle > Bloomin'Gardens" on top)
    monitor = {"top": 180, "left": 300, "width": 800, "height": 520}
    sct = mss.mss()
    img = np.array(sct.grab(monitor))
    img, hs, vs = remove_brd(img, 255)
    assert img.shape == (500, 738, 4), 'Instruction: max https://www.miniclip.com/games/bloomin-gardens/en/ and run!'
    # cv2.imwrite("capture.png", img)

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
    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    grid = grid.reshape((81,2))
    ret,label,center=cv2.kmeans(grid, 8, None, criteria, 40, cv2.KMEANS_PP_CENTERS)
    # # Now separate the data, Note the flatten()
    # A = []
    # for i in range(8): A.append(grid[label.ravel() == i])
    # # Plot the data
    # for i in range(8): plt.scatter(A[i][:,0],A[i][:,1])
    # plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
    # plt.xlabel('Height'),plt.ylabel('Weight')
    # plt.show()
    for i in range(9): print([x[0] for x in label[i*9:i*9+9]])

if __name__ == '__main__':
    detect_wnd()