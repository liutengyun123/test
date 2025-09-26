import cv2

# 读取PPM文件
image = cv2.imread('D:\googleDownload\hpatches-sequences-release\hpatches-sequences-release\\v_boat\\1.ppm', cv2.IMREAD_COLOR)
 
# 显示图像
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()