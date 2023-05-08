from ocrVisual import ocr_image, ocr_frame, ocr_detect
import cv2

# img_path = '/home/qiaoyj/Documents/PaddleOCR/ppocr_img/ppocr_img/ch/ch.jpg'
img_ori = 'word_1.jpg'
img_num = cv2.imread(img_ori)

ocr_detect(flag=1, frame=img_num)