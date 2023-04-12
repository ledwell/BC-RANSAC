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
import util
from matplotlib import pyplot as plt
from PIL import Image
from itertools import product

THRESHOLD = 0.6
NUM_ITERS = 1000
    
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

def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    
    print(f'Range Height: 0,{h-h%d}  - Width: 0, {w-w%d}')
    
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        #print(f'For Coordinates included within {i}, {j}, {i+d}, {j+d}')
        #SAVE TILE IMAGE INTO TEMP
        img.crop(box).save(out)
        #print(os.path.join(dir_out, f'{name}_{i}_{j}{ext}'))
        temp = cv2.imread(os.path.join(dir_out, f'{name}_{i}_{j}{ext}'))
        #print(f'Coarseness: {coarseness(temp, 5)}')

        #CONTRAST CALCULATION FROM - https://stackoverflow.com/questions/58821130/how-to-calculate-the-contrast-of-an-image
        Y = cv2.cvtColor(temp, cv2.COLOR_BGR2YUV)[:,:,0]
        # compute min and max of Y
        min = np.min(Y)
        max = np.max(Y)
        # compute contrast
        contrast = (max-min)/(max+min)
        #print(f'Contrast: {contrast}')
        
        #SMOOTHNESS CALCULATION FROM - https://stackoverflow.com/questions/24671901/does-there-exist-a-way-to-directly-figure-out-the-smoothness-of-a-digital-imag
        roughness = np.average(np.absolute(ndi.filters.laplace(cv2.imread(os.path.join(dir_out, f'{name}_{i}_{j}{ext}')).astype(float) / 255.0)))
        #print(f'Roughness:{roughness}')
        
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
    #print('\nPAIRS')
    #print(pairs)
    #print('\n')
    for x1, y1, x2, y2 in pairs:
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    #print('\n A BEFORE np.array')
    #print(A)
    A = np.array(A)
    #print('\n A AFTER np.array')
    #print(A)
    #print('\n')
    # Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(A)

    # V has shape (9, 9) for any number of input pairs. V[-1] is the eigenvector
    # of (A^T)A with the smalles eigenvalue. Reshape into 3x3 matrix.
    #print("V[-1]:")
    #print(V[-1])
    H = np.reshape(V[-1], (3, 3))
    #print("\nH:")
    #print(H)
    #print('\n')
    # Normalization
    H = (1 / H.item(8)) * H
    #print(H)
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

def PSSCRANSAC(point_map, threshold=THRESHOLD, verbose=True):
    """Runs the SAC algorithm.

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
    
    tile('1.png', f'input/images/{args.directory}/', util.OUTPUT_PATH_TILES, 100)

    tiles1 = []
    
    #ITERATE THROUGH IMAGES IN FOLDER
    for images in os.listdir(util.OUTPUT_PATH_TILES):
        # check if the image ends with png
        if (images.endswith(".png")):
            print(images)
    
    
    
    
    
    
    #EVALUATE TEXTURE MAGNITUDE FOR THE IMAGE BLOCKS
    
    #SORT POINT MAP BY QUALITY BASED ON TEXTURE MAGNITUDE

    if verbose:
        print(f'Running SAC with {len(point_map)} points...')
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

    if verbose:
        print(f'\nNum matches: {len(point_map)}')
        print(f'Num inliers: {len(bestInliers)}')
        print(f'Min inliers: {len(point_map) * threshold}')

    return homography, bestInliers


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

    if verbose:
        print(f'\nNum matches: {len(point_map)}')
        print(f'Num inliers: {len(bestInliers)}')
        print(f'Min inliers: {len(point_map) * threshold}')

    return homography, bestInliers

def PROSAC(point_map, threshold=THRESHOLD, verbose=True):
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
    if verbose:
        print(f'Running PROSAC with {len(point_map)} points...')
    bestInliers = set()
    homography = None
    pre = 0.5
    i = 0
    j = 0
    MIN_INLIERS = 0
    # two terminations: max number iterations or min number of inliers
    while (i < NUM_ITERS) or (MIN_INLIERS == len(point_map) * threshold) :
        #y = 30% * total matches
        subsetPre = pre * len(point_map)
        subsetPre = int(subsetPre)
        if subsetPre < len(point_map):
            # create subset
            j= 0
            subset = [point_map[j] for j in range(j, int(subsetPre-1))]
            # choose 4 points from the subset matrix to compute the homography
            pairs = [subset[i] for i in range(i, i+4)]
            #homography
            H = computeHomography(pairs)
            inliers = {(c[0], c[1], c[2], c[3])
                       for c in subset if dist(c, H) < 500}
            #print information
            if verbose:
                print(f'\x1b[2K\r└──> iteration {i + 1}/{NUM_ITERS} ' +
                      f'\t{len(inliers)} inlier' + ('s ' if len(inliers) != 1 else ' ') +
                      f'\tbest: {len(bestInliers)}', end='')
            MIN_INLIERS = len(inliers)
            # check is lenght of inliers increased
            if len(inliers) > len(bestInliers):
                bestInliers = inliers
                MIN_INLIERS = len(bestInliers)
                homography = H
                # check if inliers is greater than threshold
                if len(bestInliers) > (len(point_map) * threshold):
                    break
            # if inliers is less than, increase subset dataset using precentage
            if MIN_INLIERS < (len(point_map) * threshold):
                pre = pre + 0.1
            i = i + 1
        else:
            break

    if verbose:
        print(f'\nTotal Num matches: {len(point_map)}')
        print(f'Num inliers: {len(bestInliers)}')
        print(f'Min inliers: {len(point_map) * threshold}')
        print(f'Subset matches: {len(subset)}')

    return homography, bestInliers

def lowes_distance(featureMatches):
    good = []
    for m,n in featureMatches :
        if m.distance < 0.7*n.distance :
            good.append(m)
    return good

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
    #print("Len Matches: ", len(matches))
    point_map = np.array([
        [kp1[match.queryIdx].pt[0],
         kp1[match.queryIdx].pt[1],
         kp2[match.trainIdx].pt[0],
         kp2[match.trainIdx].pt[1]] for match in matches
    ])
    if rtype == 'PROSAC':
        #https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        matches = cv2.BFMatcher().knnMatch(desc1, desc2, k=2)
        #print("Len Matches: ", len(matches))
        #apply ratio test quality function
        #lowe's distance algorithm
        #https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
        matches = lowes_distance(matches)
        matches = sorted(matches, key = lambda x:x.distance)
        point_map = np.array([
            [kp1[match.queryIdx].pt[0],
            kp1[match.queryIdx].pt[1],
            kp2[match.trainIdx].pt[0],
            kp2[match.trainIdx].pt[1]]for match in matches
        ])

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
    rtype = 'PROSAC'
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
        point_map = createPointMap(image1, image2, directory, rtype, verbose=verbose)
    if rtype == 'PSSC':
        #TILE ATTEMPT 1
        tile('1.png', f'input/images/{args.directory}/', f'output/images/{args.directory}/{datetime.now().strftime("%Y-%m-%d-%H%M")}/tiles', 100)
        #CHECK IMAGES
        #for images in os.listdir(util.OUTPUT_PATH_TILES):
        # check if the image ends with png
            #if (images.endswith(".png")):
                #print(images)
    if rtype == 'RANSAC':
        homography, inliers = RANSAC(point_map, verbose=verbose)
    if rtype == 'PROSAC':
        #https://willguimont.github.io/cs/2019/12/26/prosac-algorithm.html
        homography, inliers = PROSAC(point_map, verbose=verbose)

    cv2.imwrite(util.OUTPUT_PATH + 'inlier_matches.png',
                util.drawMatches(image1, image2, point_map, inliers))

    with open(util.OUTPUT_PATH + 'info.txt', 'w') as f:
        f.write(f'Homography:\n{str(homography)}\n\n')
        f.write(f'Num inliers: {len(inliers)}')

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
