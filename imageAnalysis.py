"""
file: imageAnalysis.py
----------------------
This is the main driver file, which implements a RANSAC algorithm for
homography estimation. The algorithm follows the one in "Multiple View Geometry
in computer vision" by Richard Hartley and Andrew Zisserman. RANSAC stands for
RAndom SAmple Consensus.
"""
import argparse
import csv
from datetime import datetime
import os
from os import listdir
import glob
import cv2
import numpy as np
import scipy as nd
import scipy.ndimage as ndi
from matplotlib import pyplot as plt
from PIL import Image
from itertools import product
import util
import math
import time

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from PIL import Image
from itertools import product

THRESHOLD = 0.6
NUM_ITERS = 1000
WALL_START_TIME = 0.0
WALL_END_TIME = 0.0
exPreSampleTime = 0.0
cpuPreSampleTime = 0.0
CPU_START_TIME = 0.0
CPU_END_TIME = 0.0

def roundup(x):
    return round(x) if x % 100 == 0 else round(x + 100 - x % 100)
    
def rounddown(x):
    return round(x) if x % 100 == 0 else round(x - x % 100)
    
#COARSENESS ALGORITHM FROM - https://github.com/MarshalLeeeeee/Tamura-In-Python/blob/master/tamura-numpy.py
def coarseness(image, kmax):
	image = np.array(image)
	w = image.shape[0]
	h = image.shape[1]
	kmax = kmax if (np.power(2,kmax) < w) else int(np.log(w) / np.log(2))
	kmax = kmax if (np.power(2,kmax) < h) else int(np.log(h) / np.log(2))
	average_gray = np.zeros([kmax,w,h])
	horizon = np.zeros([kmax,w,h])
	vertical = np.zeros([kmax,w,h])
	Sbest = np.zeros([w,h])

	for k in range(kmax):
		window = np.power(2,k)
		for wi in range(w)[window:(w-window)]:
			for hi in range(h)[window:(h-window)]:
				average_gray[k][wi][hi] = np.sum(image[wi-window:wi+window, hi-window:hi+window])
		for wi in range(w)[window:(w-window-1)]:
			for hi in range(h)[window:(h-window-1)]:
				horizon[k][wi][hi] = average_gray[k][wi+window][hi] - average_gray[k][wi-window][hi]
				vertical[k][wi][hi] = average_gray[k][wi][hi+window] - average_gray[k][wi][hi-window]
		horizon[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))
		vertical[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))

	for wi in range(w):
		for hi in range(h):
			h_max = np.max(horizon[:,wi,hi])
			h_max_index = np.argmax(horizon[:,wi,hi])
			v_max = np.max(vertical[:,wi,hi])
			v_max_index = np.argmax(vertical[:,wi,hi])
			index = h_max_index if (h_max > v_max) else v_max_index
			Sbest[wi][hi] = np.power(2,index)

	fcrs = np.mean(Sbest)

	return fcrs

#THIS IS A COMMENT
def sigmoid(x):
    return 1 / (1 + math.exp(-0.65 * (x - 16.5)))

#CONTRAST CALCULATION FROM - https://stackoverflow.com/questions/58821130/how-to-calculate-the-contrast-of-an-image
def contrast(temp):
    Y = cv2.cvtColor(temp, cv2.COLOR_BGR2YUV)[:,:,0]
    # compute min and max of Y
    min = np.min(Y)
    max = np.max(Y)
    contrast = (max-min)/(max+min)
    return(contrast)

#SMOOTHNESS CALCULATION FROM - https://stackoverflow.com/questions/24671901/does-there-exist-a-way-to-directly-figure-out-the-smoothness-of-a-digital-imag
def smoothness(temp):
    return(np.average(np.absolute(ndi.laplace(temp).astype(float) / 255.0)))

def createCMV(filename, dir_in, dir_out, d, rtype):
    print(f'Tiling Image and Generating Texture Magnitude Values...')
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    #print(w,h)
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    
    max = 0
    min = 9999
    coordinateMapValues = None
    coordinateMapValues = []
    e = math.e
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)
        temp = cv2.imread(os.path.join(dir_out, f'{name}_{i}_{j}{ext}'))
        contrastValue = float(contrast(temp)/255.0)
        if contrastValue == np.inf:
            contrastValue = 0
        if rtype == 1:
            #BLUR TAKEN FROM https://stackoverflow.com/questions/28717054/calculating-sharpness-of-an-image
            canny = cv2.Canny(temp, 50, 250)
            blur = np.mean(canny)/100
            textureMagnitude = (contrastValue  + blur)
            coordinateMapValues.append([i, i+d, j, j+d, textureMagnitude, contrastValue, blur])
        if rtype == 0:
            coarsenessValue = coarseness(temp, 5)
            ncv = sigmoid(coarsenessValue)*100
            smoothnessValue = smoothness(temp) 
            nsv = 1 - smoothnessValue
            textureMagnitude = (nsv+contrastValue+ncv)
            coordinateMapValues.append([i, i+d, j, j+d, textureMagnitude, ncv, contrastValue, nsv])
        
    print(f'Coordinate Map and Texture Magnitude Values Completed.')
    return(coordinateMapValues, w, h)
    
    
def computeHomography(pairs):
    """Solves for the homography given any number of pairs of points. Visit
    http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf
    slide 9 for more details.

    Args:
        pairs (List[List[List]]): List of pairs of (x, y) points.

    Returns:
        np.ndarray: The computed homography.
    """
    A = []
    for x1, y1, x2, y2 in pairs:
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    A = np.array(A)
    # Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(A)
    # V has shape (9, 9) for any number of input pairs. V[-1] is the eigenvector
    # of (A^T)A with the smalles eigenvalue. Reshape into 3x3 matrix.
    H = np.reshape(V[-1], (3, 3))
    # Normalization
    H = (1 / H.item(8)) * H
    return H


def dist(pair, H):
    """Returns the geometric distance between a pair of points given the
    homography H.

    Args:
        pair (List[List]): List of two (x, y) points.
        H (np.ndarray): The homography.

    Returns:
        float: The geometric distance.
    """
    # points in homogeneous coordinates
    p1 = np.array([pair[0], pair[1], 1])
    p2 = np.array([pair[2], pair[3], 1])

    p2_estimate = np.dot(H, np.transpose(p1))
    p2_estimate = (1 / p2_estimate[2]) * p2_estimate

    return np.linalg.norm(np.transpose(p2) - p2_estimate)

def BCRANSAC(directory, point_map, threshold=THRESHOLD, verbose=True):
    """Runs the BCRANSAC algorithm.

    BLUR-CONTRAST RANSAC - sift.detectandcompute found most key points within areas
    of the image that contained low blur values and high contrast values. For that reason,
    we create an algorithm that functions like PSSC-RANSAC but is evaluating texture magnitude 
    based on BLUR and CONTRAST values only.

    Args:
        point_map (List[List[List]]): Map of (x, y) points from one image to the
            another image.
        threshold (float, optional): The minimum portion of points that should
            be inliers before the algorithm terminates. Defaults to THRESHOLD.
        verbose (bool, optional): True if additional information should be
            printed. Defaults to True.

    Returns:
        (np.ndarray, set(List[List])): The homography and set of inliers.
    """
    
    #SPLIT IMAGES INTO BLOCKS
    #TILE IMAGES AT 100 PIXELS AS DONE IN THE PAPER 'ACCELERATED RANSAC FOR ACCURATE IMAGE REGISTRATION IN AERIAL VIDEO SURVEILLANCE'
    #TILES WILL BE EVALUATED 
    print(f'\n\n ---BC-RANSAC---')
    print(f'Pre Sampling for BC-RANSAC')
    
    print(f'Creating Coordinate Map Values...')
    coordinateMapValues = None

    coordinateMapValues, w, h = createCMV('1.png', f'input/images/{args.directory}/', util.OUTPUT_PATH_TILES, 100, 1)

    for rowcmv in coordinateMapValues:
        rowcmv.append(0)

    print(f'Sorting Point Map By Quality...')
    #SORT POINT MAP BY QUALITY BASED ON TEXTURE MAGNITUDE
    sortedPointMap = None
    sortedPointMap = []
    
    for row in point_map:
        for rowcmv in coordinateMapValues:
            if row[0] > rowcmv[2] and row[0] < rowcmv[3] and row[1] > rowcmv[0] and row[1] < rowcmv[1]:
                sortedPointMap.append([row[0], row[1], row[2], row[3], rowcmv[4]])
                rowcmv[7] += 1
                break
    
    with open(f'{util.POINT_MAPS_PATH}/{directory}-BCcoordinateMapValues.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['y1', 'y2', 'x1', 'x2', 'quality','contrast', 'blur', 'count'])
        for line in coordinateMapValues:
            writer.writerow(line)
        
    sortedPointMap.sort(key=lambda row: (row[4]), reverse=True)
    
    temp = None
    temp = []
    for row in sortedPointMap:
        temp.append([row[0], row[1], row[2], row[3]])
    
    sortedPointMap = temp
    print(f'Point Map Sorted by Texture Magnitude Values')
    
    if verbose:
        print(f'Running BCRANSAC with {len(sortedPointMap)} points...')
    bestInliers = set()
    homography = None

    for i in range(0, NUM_ITERS, 1):
        pairs = [sortedPointMap[x] for x in range(i, i+4)]
        H = computeHomography(pairs)
        inliers = {(c[0], c[1], c[2], c[3])
                   for c in sortedPointMap if dist(c, H) < 500}

        if verbose:
            print(f'\x1b[2K\r└──> iteration {i+1}/{NUM_ITERS} ' +
                  f'\t{len(inliers)} inlier' + ('s ' if len(inliers) != 1 else ' ') +
                  f'\tbest: {len(bestInliers)}' , end='')

        if len(inliers) > len(bestInliers):
            bestInliers = inliers
            homography = H
            if len(bestInliers) > (len(sortedPointMap) * threshold):
                break

        
    inlierRatio = round(((len(bestInliers)/len(sortedPointMap))*100),2)
        
    if verbose:
        print(f'\nNum matches: {len(sortedPointMap)}')
        print(f'Num inliers: {len(bestInliers)}')
        print(f'Min inliers: {round(len(sortedPointMap) * threshold)}')
        print(f'Inlier Ratio: {inlierRatio}%')
    
    return homography, bestInliers, inlierRatio

def PSSCRANSAC(directory, point_map, threshold=THRESHOLD, verbose=True):
    """Runs the PSSCRANSAC algorithm.

    Args:
        point_map (List[List[List]]): Map of (x, y) points from one image to the
            another image.
        threshold (float, optional): The minimum portion of points that should
            be inliers before the algorithm terminates. Defaults to THRESHOLD.
        verbose (bool, optional): True if additional information should be
            printed. Defaults to True.

    Returns:
        (np.ndarray, set(List[List])): The homography and set of inliers.
    """
    
    #SPLIT IMAGES INTO BLOCKS
    #TILE IMAGES AT 100 PIXELS AS DONE IN THE PAPER 'ACCELERATED RANSAC FOR ACCURATE IMAGE REGISTRATION IN AERIAL VIDEO SURVEILLANCE'
    #TILES WILL BE EVALUATED 

    print(f'\n\n ---PSSC-RANSAC---')
    print(f'Pre Sampling for PSSC-RANSAC')
    
    print(f'Creating Coordinate Map Values...')

    coordinateMapValues2, w, h = createCMV('1.png', f'input/images/{args.directory}/', util.OUTPUT_PATH_TILES, 100, 0)

    for rowcmv in coordinateMapValues2:
        rowcmv.append(0)

    print(f'Sorting Point Map By Quality...')
    #SORT POINT MAP BY QUALITY BASED ON TEXTURE MAGNITUDE
    sortedPointMap2 = []
    
    for row in point_map:
        for rowcmv in coordinateMapValues2:
            if row[0] > rowcmv[2] and row[0] < rowcmv[3] and row[1] > rowcmv[0] and row[1] < rowcmv[1]:
                sortedPointMap2.append([row[0], row[1], row[2], row[3], rowcmv[4]])
                rowcmv[8] += 1
                break
    
    with open(f'{util.POINT_MAPS_PATH}/{directory}-coordinateMapValues.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['y1', 'y2', 'x1', 'x2', 'quality','coarseness','contrast', 'smoothness', 'count'])
        for line in coordinateMapValues2:
            writer.writerow(line)
        
    sortedPointMap2.sort(key=lambda row: (row[4]), reverse=True)
    
    temp = []
    for row in sortedPointMap2:
        temp.append([row[0], row[1], row[2], row[3]])
    
    sortedPointMap2 = temp
    print(f'Point Map Sorted by Texture Magnitude Values')
    
    if verbose:
        print(f'Running PSSC-RANSAC with {len(sortedPointMap2)} points...')
    bestInliers = set()
    homography = None

    for i in range(0, NUM_ITERS, 1):
        pairs = [sortedPointMap2[x] for x in range(i, i+4)]
        H = computeHomography(pairs)
        inliers = {(c[0], c[1], c[2], c[3])
                   for c in sortedPointMap2 if dist(c, H) < 500}

        if verbose:
            print(f'\x1b[2K\r└──> iteration {i+1}/{NUM_ITERS} ' +
                  f'\t{len(inliers)} inlier' + ('s ' if len(inliers) != 1 else ' ') +
                  f'\tbest: {len(bestInliers)}' , end='')

        if len(inliers) > len(bestInliers):
            bestInliers = inliers
            homography = H
            if len(bestInliers) > (len(sortedPointMap2) * threshold):
                break
        

        
    inlierRatio = round(((len(bestInliers)/len(sortedPointMap2))*100),2)
        
    if verbose:
        print(f'\nNum matches: {len(sortedPointMap2)}')
        print(f'Num inliers: {len(bestInliers)}')
        print(f'Min inliers: {round(len(sortedPointMap2) * threshold)}')
        print(f'Inlier Ratio: {inlierRatio}%')
    
    return homography, bestInliers, inlierRatio

def PROSAC(point_map, threshold=THRESHOLD, verbose=True):
    """Runs the PROSAC algorithm.

    Args:
        point_map (List[List[List]]): Map of (x, y) points from one image to the
            another image.
        threshold (float, optional): The minimum portion of points that should
            be inliers before the algorithm terminates. Defaults to THRESHOLD.
        verbose (bool, optional): True if additional information should be
            printed. Defaults to True.

    Returns:
        (np.ndarray, set(List[List])): The homography and set of inliers.
    """
    print(f'\n\n ---PROSAC---')
    if verbose:
        print(f'Running PROSAC with {len(point_map)} points...')
    bestInliers = set()
    homography = None
    pre = 0.5
    i = 0
    j = 0
    MIN_INLIERS = 0
    # two terminations: max number iterations or min number of inliers
    while (i < NUM_ITERS): 
        #y = 30% * total matches
        subsetPre = pre * len(point_map)
        subsetPre = int(subsetPre)
        
        # create subset
        j= 0
        subset = [point_map[j] for j in range(j, int(subsetPre-1))]
        # choose 4 random points from the subset matrix to compute the homography
        pairs = [subset[i] for i in np.random.choice(len(subset), 4)]
        #homography
        H = computeHomography(pairs)
        inliers = {(c[0], c[1], c[2], c[3])
                    for c in subset if dist(c, H) < 500}
        #print information
        if verbose:
            print(f'\x1b[2K\r└──> iteration {i + 1}/{NUM_ITERS} ' +
                    f'\t{len(inliers)} inlier' + ('s ' if len(inliers) != 1 else ' ') +
                    f'\tbest: {len(bestInliers)}', end=' ')

        # check is length of inliers increased
        if len(inliers) > len(bestInliers):
            bestInliers = inliers
            homography = H
            inlierRatio = round((len(bestInliers)/len(point_map))*100,2)
            # check if inliers is greater than threshold
            if len(bestInliers) > (len(point_map) * threshold):
                break

        # if inliers is less than, increase subset dataset using precentage
        if len(inliers) < (len(point_map) * threshold) and len(point_map) < (pre+0.1)*len(point_map):
            pre = pre + 0.1
        i = i + 1
        
    if verbose:
        print(f'\nTotal Num matches: {len(point_map)}')
        print(f'Num inliers: {len(bestInliers)}')
        print(f'Min inliers: {round(len(point_map) * threshold)}')
        print(f'Subset matches: {len(subset)}')
        print(f'Inlier Ratio: {inlierRatio}')

    return homography, bestInliers, inlierRatio

def lowes_distance(featureMatches):
    good = []
    for m,n in featureMatches :
        if m.distance < 0.7*n.distance :
            good.append(m)
    return good

def RANSAC(point_map, threshold=THRESHOLD, verbose=True):
    """Runs the RANSAC algorithm.

    Args:
        point_map (List[List[List]]): Map of (x, y) points from one image to the
            another image.
        threshold (float, optional): The minimum portion of points that should
            be inliers before the algorithm terminates. Defaults to THRESHOLD.
        verbose (bool, optional): True if additional information should be
            printed. Defaults to True.

    Returns:
        (np.ndarray, set(List[List])): The homography and set of inliers.
    """
    print(f'\n\n ---RANSAC---')
    if verbose:
        print(f'Running RANSAC with {len(point_map)} points...')
    bestInliers = set()
    homography = None
    
    for i in range(NUM_ITERS):
        # randomly choose 4 points from the matrix to compute the homography
        pairs = [point_map[i] for i in np.random.choice(len(point_map), 4)]
        H = computeHomography(pairs)
        inliers = {(c[0], c[1], c[2], c[3])
                   for c in point_map if dist(c, H) < 500}

        if verbose:
            print(f'\x1b[2K\r└──> iteration {i + 1}/{NUM_ITERS} ' +
                  f'\t{len(inliers)} inlier' + ('s ' if len(inliers) != 1 else ' ') +
                  f'\tbest: {len(bestInliers)}' , end='')

        if len(inliers) > len(bestInliers):
            bestInliers = inliers
            homography = H
            if len(bestInliers) > (len(point_map) * threshold):
                break
    
    inlierRatio = round((len(bestInliers)/len(point_map))*100,2)
    
    if verbose:
        print(f'\nNum matches: {len(point_map)}')
        print(f'Num inliers: {len(bestInliers)}')
        print(f'Min inliers: {round(len(point_map) * threshold)}')
        print(f'Inlier Ratio: {inlierRatio}%')


    
    return homography, bestInliers, inlierRatio


def createPointMap(image1, image2, directory, rtype, verbose=True):
    """Creates a point map of shape (n, 4) where n is the number of matches
    between the two images. Each row contains (x1, y1, x2, y2), where (x1, y1)
    in image1 maps to (x2, y2) in image2.

    sift.detectAndCompute returns
        keypoints: a list of keypoints
        descriptors: a numpy array of shape (num keypoints, 128)

    Args:
        image1 (cv2.Mat): The first image.
        image2 (cv2.Mat): The second image.
        directory (str): The directory to save a .csv file to.
        verbose (bool, optional): True if additional information should be
            printed. Defaults to True.

    Returns:
        List[List[List]]: The point map of (x, y) points from image1 to image2.
    """
    if verbose:
        print('Finding keypoints and descriptors for both images...')
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)

    cv2.imwrite(util.OUTPUT_PATH + 'keypoints-1.png',
                cv2.drawKeypoints(image1, kp1, image1))
    cv2.imwrite(util.OUTPUT_PATH + 'keypoints-2.png',
                cv2.drawKeypoints(image2, kp2, image2))

    if verbose:
        print('Determining matches...')
    matches = cv2.BFMatcher(cv2.NORM_L2, True).match(desc1, desc2)

    point_map = np.array([
        [kp1[match.queryIdx].pt[0],
         kp1[match.queryIdx].pt[1],
         kp2[match.trainIdx].pt[0],
         kp2[match.trainIdx].pt[1]] for match in matches
    ])

    if rtype == 1:
        #https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        matches = cv2.BFMatcher().knnMatch(desc1, desc2, k=2)
        #print("Len Matches: ", len(matches))
        #apply ratio test quality function
        #lowe's distance algorithm
        #https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
        startPreSampleTime = time.time()
        startCPUPreSampleTime = time.process_time()
        matches = lowes_distance(matches)
        matches = sorted(matches, key = lambda x:x.distance)
        endCPUPreSampleTime = time.process_time()
        endPreSampleTime = time.time()
        exPreSampleTime = endPreSampleTime - startPreSampleTime
        cpuPreSampleTime = endCPUPreSampleTime - startCPUPreSampleTime
        point_map = np.array([
            [kp1[match.queryIdx].pt[0],
            kp1[match.queryIdx].pt[1],
            kp2[match.trainIdx].pt[0],
            kp2[match.trainIdx].pt[1]]for match in matches
        ])

    if rtype == 1:
        cv2.imwrite(util.OUTPUT_PATH + 'matches_pro.png',
                    util.drawMatches(image1, image2, point_map))

        with open(f'{util.POINT_MAPS_PATH}/{directory}-point_map_pro.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x1', 'y1', 'x2', 'y2'])
            for line in point_map:
                writer.writerow(line)
    else:
        cv2.imwrite(util.OUTPUT_PATH + 'matches.png',
                    util.drawMatches(image1, image2, point_map))

        with open(f'{util.POINT_MAPS_PATH}/{directory}-point_map.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x1', 'y1', 'x2', 'y2'])
            for line in point_map:
                writer.writerow(line)

    return point_map


def main(image1, image2, directory, verbose=True):
    """Analyzes the relation between image1 and image2 by computing a point map
    and running the RANSAC algorithm to compute the homography between the two
    images.

    Args:
        image1 (cv2.Mat): The first image.
        image2 (cv2.Mat): The second image.
        directory (str): The directory to to read from a saved .csv file or
            write to a new one.
        verbose (bool, optional): True if additional information should be
            printed. Defaults to True.

    Returns:
        (List[List[List]], set(List[List]), np.ndarray): The computed point map,
            the set of inlier points, and the computed homography
    """

    point_map = None
    if f'{directory}-point_map.csv' in os.listdir(util.POINT_MAPS_PATH):
        if verbose:
            print('Loading saved point map...')
        point_map = np.loadtxt(
            util.POINT_MAPS_PATH + f'{directory}-point_map.csv',
            delimiter=',', skiprows=1)
    else:
        if verbose:
            print('Creating point map...')
        point_map = createPointMap(image1, image2, directory, 0)

    point_map_pro = None
    if f'{directory}-point_map_pro.csv' in os.listdir(util.POINT_MAPS_PATH):
        if verbose:
            print('Loading saved point map...')
        point_map_pro = np.loadtxt(
            util.POINT_MAPS_PATH + f'{directory}-point_map_pro.csv',
            delimiter=',', skiprows=1)
    else:
        if verbose:
            print('Creating point map...')
        point_map_pro = createPointMap(image1, image2, directory, 1)
    

    #RANSAC
    tic = time.perf_counter()
    homography, inliers, inlierRatio = RANSAC(point_map, verbose=verbose)
    toc = time.perf_counter()
    ransactimer = toc-tic
    print(f'RANSAC completed in {toc-tic:0.4f} seconds')
    
    #PROSAC
    tic = time.perf_counter()
    homographyPRO, inliersPRO, inlierRatioPRO = PROSAC(point_map_pro, verbose=verbose)
    toc = time.perf_counter()
    prosactimer = toc-tic
    print(f'PROSAC completed in {toc-tic:0.4f} seconds')

    #PSSC-RANSAC
    tic = time.perf_counter()
    homographyPSSC, inliersPSSC, inlierRatioPSSC = PSSCRANSAC(directory, point_map, verbose=verbose)
    toc = time.perf_counter()
    pssctimer = toc-tic
    print(f'PSSC-RANSAC completed in {toc-tic:0.4f} seconds')

    #BCSAC
    tic = time.perf_counter()
    homographyBC, inliersBC, inlierRatioBC = BCRANSAC(directory, point_map, verbose=verbose)
    toc = time.perf_counter()
    bctimer = toc-tic
    print(f'BC-RANSAC completed in {toc-tic:0.4f} seconds')

    cv2.imwrite(util.OUTPUT_PATH + 'RANSAC_inlier_matches.png',
                util.drawMatches(image1, image2, point_map, inliers))
    
    cv2.imwrite(util.OUTPUT_PATH + 'PROSAC_inlier_matches.png',
                util.drawMatches(image1, image2, point_map_pro, inliersPRO))

    cv2.imwrite(util.OUTPUT_PATH + 'PSSC_RANSAC_inlier_matches.png',
                util.drawMatches(image1, image2, point_map, inliersPSSC))

    cv2.imwrite(util.OUTPUT_PATH + 'BC_RANSAC_inlier_matches.png',
                util.drawMatches(image1, image2, point_map, inliersBC))

    with open(util.OUTPUT_PATH + 'info.txt', 'w') as f:
        f.write(f'RANSAC Homography:\n{str(homography)}\n\n')
        f.write(f'\nRANSAC Num inliers: {len(inliers)}')
        f.write(f'\nRANSAC InlierRatio: {inlierRatio}')
        f.write(f'\nRANSAC Time: {ransactimer}')
        f.write(f'\nPROSAC Homography:\n{str(homographyPRO)}\n\n')
        f.write(f'\nPROSAC Num inliers: {len(inliersPRO)}')
        f.write(f'\nPROSAC InlierRatio: {inlierRatioPRO}')
        f.write(f'\nRANSAC Time: {prosactimer}')
        f.write(f'\nPSSC-RANSAC Homography:\n{str(homographyPSSC)}\n\n')
        f.write(f'\nPSSC-RANSAC Num inliers: {len(inliersPSSC)}')
        f.write(f'\nPSSC-RANSAC Num inliers: {inlierRatioPSSC}')
        f.write(f'\nRANSAC Time: {pssctimer}')
        f.write(f'\nBC-RANSAC Homography:\n{str(homographyBC)}\n\n')
        f.write(f'\nBC-RANSAC Num inliers: {len(inliersBC)}')
        f.write(f'\nBC-RANSAC Num inliers: {inlierRatioBC}')
        f.write(f'\nRANSAC Time: {bctimer}')

    return point_map, inliers, homography


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--directory', help='image directory id',
                            default='00')
    arg_parser.add_argument('-v', '--verbose',
                            help='increase output verbosity', action='store_true')
    args = arg_parser.parse_args()

    util.INPUT_PATH += f'images/{args.directory}/'
    util.OUTPUT_PATH += f'images/{args.directory}/{datetime.now().strftime("%Y-%m-%d-%H%M")}/'
    util.OUTPUT_PATH_TILES += f'images/{args.directory}/{datetime.now().strftime("%Y-%m-%d-%H%M")}/tiles/'
    util.POINT_MAPS_PATH += f'images/'

    image1 = cv2.imread(util.INPUT_PATH + '1.png', 0)
#    image1 = cv2.imread(util.INPUT_PATH + '1.png', 1)
    assert image1 is not None, f'Invalid first image: {util.INPUT_PATH}1.png'
    image2 = cv2.imread(util.INPUT_PATH + '2.png', 0)
#    image2 = cv2.imread(util.INPUT_PATH + '2.png', 1)
    assert image2 is not None, f'Invalid second image: {util.INPUT_PATH}2.png'

    os.makedirs(util.OUTPUT_PATH, exist_ok=True)
    os.makedirs(util.OUTPUT_PATH_TILES, exist_ok=True)
    os.makedirs(util.POINT_MAPS_PATH, exist_ok=True)

    main(image1, image2, args.directory)
