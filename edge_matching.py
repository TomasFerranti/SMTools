import cv2 as cv
import numpy as np
from improve_edge import improveEdges

# Merge everything to test with the available images
def getLogDist(array):
    L = array.shape[0]
    X = np.arange(L)
    return array * (np.log(1 + L - X) / np.log(1 + L))

def findBestAdds(img, c_x, c_y, r_x, r_y, image_left, ds = 5):
    # X axis padding necessary for right image
    x_padding = int(0.5*img.shape[1])
    if image_left:
        img = img[:, :-x_padding, :]
    else:
        x_padding -= 100
        img = img[:, x_padding:, :]
        c_x -= x_padding

    # Finding best adds for each rectangle side
    best_adds = []
    for coord in ['x', 'y']:
        for ind in [0, 1]:
            # Segs to test the best add based on for loop variables
            if coord == 'x':
                if ind == 0:
                    segs = img[(c_y - r_y[0]) : (c_y + r_y[1]), 
                                            0 : (c_x - r_x[0])]
                else:
                    segs = img[(c_y - r_y[0]) : (c_y + r_y[1]), 
                               (c_x + r_x[1]) : img.shape[1]]
            else:
                if ind == 0:
                    segs = img[             0 : (c_y - r_y[0]), 
                               (c_x - r_x[0]) : (c_x + r_x[1])]  
                else:
                    segs = img[(c_y + r_y[1]) : img.shape[0], 
                               (c_x - r_x[0]) : (c_x + r_x[1])]
            if ind == 0:
                segs = np.flip(segs)
            
            # Get best add
            segs = np.mean(segs, axis = int(coord == 'y'))
            segs_diff = np.sum(np.abs(segs[ds:] - segs[:-ds]), axis = 1)
            segs_diff = getLogDist(segs_diff)
            best_add = ds + np.argmax(segs_diff)
            best_adds.append(best_add)
    return best_adds

def getSplit(img):
    # Initial centers and radius of images left and right
    imgL_c_x = int(1/4 * img.shape[1])
    imgR_c_x = int(3/4 * img.shape[1])
    imgL_c_y = int(1/2 * img.shape[0])
    imgR_c_y = int(1/2 * img.shape[0])
    imgL_r_x = 2*[int((1/2 * 0.25) * img.shape[1])] 
    imgR_r_x = 2*[int((1/2 * 0.25) * img.shape[1])]
    imgL_r_y = 2*[int((1/2 * 0.7) * img.shape[0])]
    imgR_r_y = 2*[int((1/2 * 0.7) * img.shape[0])]
    
    # Find the amount to add to borders
    imgL_adds = findBestAdds(img, imgL_c_x, imgL_c_y, imgL_r_x, imgL_r_y, image_left = True, ds = 20)
    imgL_add_x, imgL_add_y = imgL_adds[0:2], imgL_adds[2:]
    imgR_adds = findBestAdds(img, imgR_c_x, imgR_c_y, imgR_r_x, imgR_r_y, image_left = False, ds = 20)
    imgR_add_x, imgR_add_y = imgR_adds[0:2], imgR_adds[2:]

    # Update centers to the new rectangle
    imgL_c_x += int((imgL_add_x[1] - imgL_add_x[0]) / 2)
    imgL_c_y += int((imgL_add_y[1] - imgL_add_y[0]) / 2)
    imgR_c_x += int((imgR_add_x[1] - imgR_add_x[0]) / 2)
    imgR_c_y += int((imgR_add_y[1] - imgR_add_y[0]) / 2)
    
    # Update radius to the new value with add
    imgL_r_x = int(((imgL_add_x[1] + imgL_r_x[1]) + (imgL_add_x[0] + imgL_r_x[0])) / 2)
    imgL_r_y = int(((imgL_add_y[1] + imgL_r_y[1]) + (imgL_add_y[0] + imgL_r_y[0])) / 2)
    imgR_r_x = int(((imgR_add_x[1] + imgR_r_x[1]) + (imgR_add_x[0] + imgR_r_x[0])) / 2)
    imgR_r_y = int(((imgR_add_y[1] + imgR_r_y[1]) + (imgR_add_y[0] + imgR_r_y[0])) / 2)
    
    r_x = min(imgL_r_x, imgR_r_x)
    r_y = min(imgL_r_y, imgR_r_y)
    
    # Get our image through its boundaries
    imgL = img[(imgL_c_y - r_y) : (imgL_c_y + r_y), (imgL_c_x - r_x) : (imgL_c_x + r_x), :]
    imgR = img[(imgR_c_y - r_y) : (imgR_c_y + r_y), (imgR_c_x - r_x) : (imgR_c_x + r_x), :]
    return imgL, imgR

def rgbToLuv(rgb_array):
    pre_luv = (rgb_array / 255).astype(np.float32)
    pre_luv = pre_luv.reshape(1, -1, 3)
    luv = cv.cvtColor(pre_luv, cv.COLOR_RGB2Luv)
    return luv[0, :, 0]

def getErrorSeries(s1, s2, dx):
    width = s1.shape[0]
    if dx > 0:
        s1 = s1[dx:]
        s2 = s2[:(-dx)]
    elif dx < 0:
        s1 = s1[:dx]
        s2 = s2[(-dx):]
    error = np.mean(np.abs(s2 - s1))
    error = error * ((width / 2 + abs(dx)) / width)
    return error

def resizeWithAspectRatio(image, width = None, height = None, inter = cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv.resize(image, dim, interpolation=inter)

def findEquivalentEdge(imgL, imgR, improved_lineL):
    height, width = imgL.shape[0:2]

    s1_l = rgbToLuv(imgL[improved_lineL[0, 1], :])
    s1_r = rgbToLuv(imgR[improved_lineL[0, 1], :])
    s2_l = rgbToLuv(imgL[improved_lineL[1, 1], :])
    s2_r = rgbToLuv(imgR[improved_lineL[1, 1], :])

    best_dx_s1 = 0
    best_error_s1 = 255
    best_dx_s2 = 0
    best_error_s2 = 255
    for dx in range(- int(width / 2), int(width / 2) + 1):
        error_s1 = getErrorSeries(s1_l, s1_r, dx)
        error_s2 = getErrorSeries(s2_l, s2_r, dx)
        if error_s1 < best_error_s1:
            best_dx_s1 = dx
            best_error_s1 = error_s1
        if error_s2 < best_error_s2:
            best_dx_s2 = dx
            best_error_s2 = error_s2

    lineR = np.array([[improved_lineL[0, 0] + best_dx_s1, improved_lineL[0, 1]],
                    [improved_lineL[1, 0] + best_dx_s2, improved_lineL[1, 1]]])
    improved_lineR = improveEdges(imgR, [lineR.tolist()], plot=False)
    improved_lineR = np.array(improved_lineR[0])
    return lineR, improved_lineR

class EdgeMatchingWidget(object):
    def __init__(self, imageWidth, imagePath):
        self.original_image = cv.imread(imagePath)
        self.original_left, self.original_right = getSplit(self.original_image)
        self.clone_left = self.original_left.copy()
        self.clone_right = self.original_right.copy()
        self.imageWidth = imageWidth

        cv.namedWindow('image')
        cv.setMouseCallback('image', self.extractCoordinates)

        # List to store start/end points
        self.image_coordinates = []

        # Get mean translation between two images
        self.dx = 0
        self.dy = 0
        self.getTranslation()

    def getTranslation(self):
        img1 = cv.cvtColor(self.original_left, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(self.original_right, cv.COLOR_BGR2GRAY)

        # SOURCE: https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html

        # So first we need to find as many possible matches between two images to find the best translation. 
        # For this, we use SIFT descriptors with FLANN based matcher and ratio test.
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        pts1 = []
        pts2 = []
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        # Now we have the list of best matches from both the images. Let's find the mean translation
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        total_pts = len(pts1)
        total_dx, total_dy = 0, 0
        for pointIndex in range(total_pts):
            pt1 = pts1[pointIndex]
            pt2 = pts2[pointIndex]
            total_dx += pt2[0] - pt1[0]
            total_dy += pt2[1] - pt1[1]
        
        self.dx = total_dx / total_pts
        self.dy = total_dy / total_pts

    def extractCoordinates(self, event, x, y, flags, parameters):
        # Adjust (x,y) coordinates
        ratio = self.clone_left.shape[1] / self.imageWidth
        x, y = int(ratio * x), int(ratio * y)

        # Record starting (x,y) coordinates on left mouse button click
        if event == cv.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            lineL = np.array([[self.image_coordinates[0][0], self.image_coordinates[0][1]], 
                              [self.image_coordinates[1][0], self.image_coordinates[1][1]]])
            #lineR, improved_lineR = find_equivalent_edge(self.original_left, self.original_right , lineL)

            # Draw line
            improved_lineL = improveEdges(self.original_left, [lineL.tolist()], plot=False)
            improved_lineL = np.array(improved_lineL[0])
            lineR = improved_lineL.copy()
            lineR[:, 0] = lineR[:, 0] + self.dx
            lineR[:, 1] = lineR[:, 1] + self.dy
            improved_lineR = improveEdges(self.original_right, [lineR.tolist()], plot=False)
            improved_lineR = np.array(improved_lineR[0])

            cv.line(self.clone_left, lineL[0, :], lineL[1, :], (10,10,255), 2)
            cv.line(self.clone_left, improved_lineL[0, :], improved_lineL[1, :], (10,255,10), 2)

            cv.line(self.clone_right, lineR[0, :], lineR[1, :], (10,10,255), 2)
            cv.line(self.clone_right, improved_lineR[0, :], improved_lineR[1, :], (10,255,10), 2)

        # Clear drawing boxes on right mouse button click
        elif event == cv.EVENT_RBUTTONDOWN:
            self.clone_left = self.original_left.copy()
            self.clone_right = self.original_right.copy()

    def returnImages(self):
        return self.clone_left, self.clone_right

#fig, ax = plt.subplots(1, 2, figsize = (20,10))

imageWidth = 700
whiteSpaceWidth = 200
imagePath = 'imagens_stereo_IMS/002080Vol05Cx0317.jpg'
edge_matching_widget = EdgeMatchingWidget(imageWidth, imagePath)
while True:
    imgL, imgR = edge_matching_widget.returnImages()
    plotImgL = resizeWithAspectRatio(imgL, width = imageWidth)
    plotImgR = resizeWithAspectRatio(imgR, width = imageWidth)
    whiteRectangle = np.zeros(shape = (plotImgL.shape[0], whiteSpaceWidth, 3)).astype(np.uint8)

    plotImg = np.concatenate([plotImgL, whiteRectangle, plotImgR], axis = 1)
    cv.imshow('image', plotImg)

    key = cv.waitKey(1)

    # Close program with keyboard 'q'
    if key == ord('q'):
        cv.destroyAllWindows()
        exit(1)