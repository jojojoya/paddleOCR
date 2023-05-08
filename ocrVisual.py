from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import cv2


def ocr_frame(img_num, lang="ch", font_path="doc/fonts/simfang.ttf"):
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang=lang)
    result = ocr.ocr(img_num, cls=True)

    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)


def ocr_image(img_result, lang="ch", font_path="doc/fonts/simfang.ttf"):
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang=lang)
    result = ocr.ocr(img_result, cls=True)

    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

    result = result[0]
    image = Image.open(img_result).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.save('result.jpg')


def ocr_detect(flag, frame):
    if flag == 1:
        img_result = cv2.imwrite('result.jpg', frame)
        ocr_image('result.jpg')
    else:
        ocr_frame(frame)