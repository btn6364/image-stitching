import numpy as np 
import cv2 as cv
import glob 

# Detech and match features in the input images. 
def detectAndMatchFeatures(img1, img2): 
    # Use ORB feature detector
    orb = cv.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Match the features from 2 descriptors. 
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    return keypoints1, keypoints2, matches

# Compute the Homography matrix that is used to project features from one image to another image's coordinate. 
def estimateHomography(keypoints1, keypoints2, matches, threshold=5):
    src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, threshold)
    return H, mask

# Warp the images. Transform image2 to image1's plane. 
def warpImages(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners2 = cv.perspectiveTransform(corners2, H)

    corners = np.concatenate((corners1, warped_corners2), axis=0)
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)

    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    warped_img2 = cv.warpPerspective(img2, Ht @ H, (xmax - xmin, ymax - ymin))
    warped_img2[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1

    return warped_img2

# Blend the 2 images together.
def blendImages(img1, img2):
    resize_scale = (img1.shape[1], img1.shape[0])
    img2 = cv.resize(img2, resize_scale, interpolation= cv.INTER_LINEAR)
    mask = np.where(img1 != 0, 1, 0).astype(np.float32)
    blended_img = img1 * mask + img2 * (1 - mask)
    return blended_img.astype(np.uint8)

# Perform image stitching 
def imageStitching(left_img, right_img): 
    keypoints1, keypoints2, matches = detectAndMatchFeatures(left_img, right_img)
    H, _ = estimateHomography(keypoints1, keypoints2, matches)
    warped_img = warpImages(right_img, left_img, H)
    output_img = blendImages(warped_img, left_img)
    return output_img

# Stitch multiple images at once
def imageStitchingMultiple(): 
    image_paths = sorted(glob.glob("images/*.jpg"))
    first = cv.imread(image_paths[0])
    second = cv.imread(image_paths[1])
    cur_stitched = imageStitching(first, second)
    for image_path in image_paths[2:]: 
        cur_img = cv.imread(image_path)
        cur_stitched = imageStitching(cur_stitched, cur_img)
    return cur_stitched

if __name__=="__main__":
    stitched_img = imageStitchingMultiple()
    cv.imshow("Stitched image", stitched_img)
    cv.waitKey(0)
    cv.destroyAllWindows()




