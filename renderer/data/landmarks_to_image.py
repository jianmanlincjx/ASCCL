import numpy as np
from PIL import Image
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * x**2 + b * x + c

def linear(x, a, b):
    return a * x + b

def setColor(im, yy, xx, color):
    if len(im.shape) == 3:
        im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]
        #if (im[yy, xx] == 0).all():
        #    im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]
        #else:
        #    im[yy, xx, 0] = ((im[yy, xx, 0].astype(float) + color[0]) / 2).astype(np.uint8)
        #    im[yy, xx, 1] = ((im[yy, xx, 1].astype(float) + color[1]) / 2).astype(np.uint8)
        #    im[yy, xx, 2] = ((im[yy, xx, 2].astype(float) + color[2]) / 2).astype(np.uint8)
    else:
        im[yy, xx] = color[0]

def drawCircle(im, x, y, rad, color=(255,0,0)):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-rad, rad):
            for j in range(-rad, rad):
                yy = np.maximum(0, np.minimum(h-1, y+i))
                xx = np.maximum(0, np.minimum(w-1, x+j))
                if np.linalg.norm(np.array([i, j])) < rad:
                    setColor(im, yy, xx, color)

def drawEdge(im, x, y, bw=1, color=(255,255,255)):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h-1, y+i))
                xx = np.maximum(0, np.minimum(w-1, x+j))
                setColor(im, yy, xx, color)

def interpPoints(x, y):
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interpPoints(y, x)
        if curve_y is None:
            return None, None
    else:
        if len(x) < 3:
            popt, _ = curve_fit(linear, x, y)
        else:
            popt, _ = curve_fit(func, x, y)
            if abs(popt[0]) > 1:
                return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(x[0], x[-1], (x[-1]-x[0]))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
    return curve_x.astype(int), curve_y.astype(int)

def create_eyes_image(A_path, size, transform_scale, add_noise, pts=None):
    w, h = size
    eyes_image = np.zeros((h, w, 3), np.int32)

    if pts is None:
        keypoints = np.loadtxt(A_path, delimiter=' ')
        pts = keypoints[0:14,:].astype(np.int32)

    left_eye_pts = np.concatenate([pts[0:6, :], pts[12:13, :]], axis=0)
    right_eye_pts = np.concatenate([pts[6:12, :], pts[13:14, :]], axis=0)

    #left_eye_pts[1:3,1] += 1 # dvp video.
    #left_eye_pts[6,1] += 1 # dvp video.
    #left_eye_pts[0,0] -= 1 # dvp video.
    #left_eye_pts[3,0] += 1 # dvp video.
    #right_eye_pts[0,0] -= 1 # dvp video.
    #right_eye_pts[3,0] += 1 # dvp video.

    if add_noise:
        scale_noise = 2 * np.random.randn(1)
        scale = 1 + scale_noise[0] / 100
        left_eye_mean = np.mean(left_eye_pts, axis=0, keepdims=True)
        right_eye_mean = np.mean(right_eye_pts, axis=0, keepdims=True)
        left_eye_pts = (left_eye_pts - left_eye_mean) * scale + left_eye_mean
        right_eye_pts = (right_eye_pts - right_eye_mean) * scale + right_eye_mean
        # add noise to eyes distance (x dimension)
        d_noise = 2 * np.random.randn(2)
        left_eye_pts[:, 0] += d_noise[0]
        right_eye_pts[:, 0] -= d_noise[1]

    pts = np.concatenate([left_eye_pts, right_eye_pts], axis=0).astype(np.int32)

    # Draw
    '''
    face_list = [ [[0,1,2,3], [3,4,5,0]], # left eye
                  [[7,8,9,10], [10,11,12,7]], # right eye
                 ]
    for edge_list in face_list:
            for edge in edge_list:
                for i in range(0, max(1, len(edge)-1)):
                    sub_edge = edge[i:i+2]
                    x, y = pts[sub_edge, 0], pts[sub_edge, 1]
                    curve_x, curve_y = interpPoints(x, y)
                    drawEdge(eyes_image, curve_x, curve_y)
    '''
    radius_left = (np.linalg.norm(pts[1]-pts[4]) + np.linalg.norm(pts[2]-pts[5])) / 4
    radius_right = (np.linalg.norm(pts[8]-pts[11]) + np.linalg.norm(pts[9]-pts[12])) / 4
    drawCircle(eyes_image, pts[6, 0], pts[6, 1], int(radius_left))
    drawCircle(eyes_image, pts[13, 0], pts[13, 1], int(radius_right))
    eyes_image = transform_scale(Image.fromarray(np.uint8(eyes_image)))
    return eyes_image
