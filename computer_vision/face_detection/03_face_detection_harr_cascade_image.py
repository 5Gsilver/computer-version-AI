import cv2 

# 분류기 
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

# 이미지 불러오기(BGR) 
image = cv2.imread('sample.jpg') 

# 흑백처리(BGR -> GRAY) 
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

# 얼굴 찾기 
faces = detector.detectMultiScale(
    image_gray, 
    scaleFactor=1.1, 
    minNeighbors=5, 
    minSize=(100, 100)
) 
print(f'faces count: {len(faces)}') 

# 찾은 얼굴에 사각형 그리기
for (x, y, w, h) in faces: 
	print(f'x: {x}, y: {y}, w: {w}, h: {h}') 
	cv2.rectangle(
		image, 
  		(x, y), (x + w, y + h), 
    	(0, 0, 255), 1
	) 

# 이미지 보여주기
cv2.imshow('image', image) 

# 키 입력 대기 
cv2.waitKey(0) 

# 윈도우 삭제 
cv2.destroyAllWindows()
