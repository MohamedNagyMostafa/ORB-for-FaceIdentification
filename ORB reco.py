import cv2
import numpy as np
import matplotlib.pyplot as plt

face_image = cv2.imread('images/face.jpeg')
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)

f, (ax1, ax2) = plt.subplots(1,2)
plt.rcParams['figure.figsize'] = [20,10]
ax1.set_title('Colored image')
ax1.imshow(face_image)
ax2.set_title('Grayed image')
ax2.imshow(face_image_gray, cmap='gray')
plt.show()

# Key points
n_features = 1000
factor = 2.0
orb = cv2.ORB_create(n_features, factor)

# K.P located and ORB descriptor
keypoints, descriptor = orb.detectAndCompute(face_image_gray, None) # No Mask

keyp_with_size = np.copy(face_image)
keyp_without_size = np.copy(face_image)

# Draw keypoints on the image
cv2.drawKeypoints(face_image, keypoints, keyp_without_size, color=(0,255,0))
cv2.drawKeypoints(face_image, keypoints, keyp_with_size, flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



f, (ax1, ax2) = plt.subplots(1,2)
plt.rcParams['figure.figsize'] = [20,10]
ax1.set_title('Key points')
ax1.imshow(keyp_without_size)
ax2.set_title('Key ponits with size')
ax2.imshow(keyp_with_size)
plt.show()

print(len(keypoints))


# Feature Matching

image_query = cv2.imread('images/face.jpeg')
image_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY)

keypoints_query, descriptor_query = orb.detectAndCompute(image_query, None) # No Mask

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(descriptor, descriptor_query)
#get short lengths first
matches = sorted(matches, key=lambda x: x.distance)

result = cv2.drawMatches(face_image_gray, keypoints, image_query, keypoints_query, matches[:300], image_query, flags=2)

plt.title('Best Matching Points')
plt.imshow(result)
plt.show()

# Face Recognition rotation & team group

image_team = cv2.imread('images/Team.jpeg')
image_filt = cv2.imread('images/faceRI.png')
image_team = cv2.cvtColor(image_team, cv2.COLOR_BGR2GRAY)
image_filt = cv2.cvtColor(image_filt, cv2.COLOR_BGR2GRAY)

# Keypoints Extraction
im_team_orb = cv2.ORB_create(3000, 2.0)

keypoints_team, descriptro_team = im_team_orb.detectAndCompute(image_team, None)
keypoints_filt, descriptor_filt = orb.detectAndCompute(image_filt, None)

# Matching
matches_team_real = bf.match(descriptor, descriptro_team)
matches_filt_real = bf.match(descriptor, descriptor_filt)

# Sorting
matches_team_real = sorted(matches_team_real, key=lambda x:x.distance)
matches_filt_real = sorted(matches_filt_real, key=lambda x:x.distance)


#drawing

result_team_real = cv2.drawMatches(face_image_gray, keypoints, image_team, keypoints_team, matches_team_real[:58], image_team, flags=2)
result_filt_real = cv2.drawMatches(face_image_gray, keypoints, image_filt, keypoints_filt, matches_filt_real[:100], image_filt,flags=2)


#ax1.set_title('Recognize over orientation + filter')
#ax1.imshow(result_filt_real)
plt.title('Recognize over orientation + filter')
plt.imshow(result_filt_real)
plt.show()