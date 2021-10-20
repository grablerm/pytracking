import os
import cv2
arr = os.listdir("/gdrive/My Drive/video")
print(arr)
video = arr[0]
print(video)
vidcap = cv2.VideoCapture(arr[0])
success,image = vidcap.read()
count = 0

os.mkdir("/gdrive/My Drive/images/")
os.mkdir("/gdrive/My Drive/tracking_results/")
os.mkdir("/gdrive/My Drive/data")
while success:
  cv2.imwrite("/gdrive/My Drive/images/%s.jpg" % str(count).zfill(4), image)     # save frame as JPEG file      
  success,image = vidcap.read()
  count += 1

os.remove("/gdrive/My Drive/video/%s" % video)