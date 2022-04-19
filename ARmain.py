import cv2
import numpy as np

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
imgTar=cv2.imread("img\\book.jpeg")
myVid = cv2.VideoCapture("img\Home.mp4")

imgTar=cv2.resize(imgTar,(700,500))

# detect special points
orb=cv2.ORB_create(nfeatures=10000)
kp1,des1 = orb.detectAndCompute(imgTar, None)
imgTar=cv2.drawKeypoints(imgTar,kp1,None)


while(myVid.isOpened()):
      
  # Capture frame-by-frame
  ret1,webc=cap.read()
  imgAug=webc.copy()

  orb=cv2.ORB_create(nfeatures=10000)
  kp2,des2 = orb.detectAndCompute(webc ,None)
  webc=cv2.drawKeypoints(webc,kp2,None)

  ret2, frame = myVid.read()
  hT, wT, cT = imgTar.shape
  frame = cv2.resize(frame,(wT,hT))
  
  try:
   bf=cv2.BFMatcher()
   matches = bf.knnMatch(des1,des2,k=2)
   good = []
   for m,n in matches:
      if m.distance < 0.75 *n.distance:
       good.append(m)
#    print("this is len of good : "+str(len(good)))
   imgfeatures=cv2.drawMatches(imgTar,kp1,webc,kp2,good,None,flags=2)

   if(True):
     srcpts= np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
     despts= np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
     matrix,mask =cv2.findHomography(srcpts,despts,cv2.RANSAC,5)
     print("this is matrix : "+str(matrix))

     pts=np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1,1,2)
     dst=cv2.perspectiveTransform(pts,matrix)

     cv2.polylines(webc,[np.int32(dst)],True,(2,255,2),5)

     frame1=cv2.warpPerspective(frame,matrix,(webc.shape[1],webc.shape[0]))

     masknew=np.zeros(webc.shape[0],webc.shape[1],np.uint8)
     cv2.fillPoly(masknew,[np.int32(dst)],(0,0,0))
     maskInv = cv2.bitwise_not(masknew)

     imgAug = cv2.bitwise_and(imgAug,imgAug,mask = maskInv)

  except:
      print("Points not detected")
      break

 
  if ret1 == True & ret2==True:
   
    # Display the resulting frame
    # cv2.imshow('Frame', frame)
    # cv2.imshow("image",imgTar)
    cv2.imshow("WebCam",webc)
    cv2.imshow("neural",imgfeatures)
    # cv2.imshow("wrap",frame1)
    cv2.imshow("mask",imgAug)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == 27:
      break
   
  # Break the loop
  else: 
    break
cap.release()   
myVid.release()
cv2.destroyAllWindows()













