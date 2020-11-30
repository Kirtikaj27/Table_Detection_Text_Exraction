import cv2
import numpy as np

def impose_mask(org_img_path, mask_img_path, output_masked_img_path):
    INPUT_IMAGE1 = org_img_path
    INPUT_IMAGE2 = mask_img_path

    src1 = cv2.imread(INPUT_IMAGE1)
    src2 = cv2.imread(INPUT_IMAGE2)

    gray1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

    gray2 = cv2.resize(gray2, gray1.shape[1::-1])

    dst = cv2.bitwise_and(gray1, gray2)
    cv2.imwrite(output_masked_img_path,dst)