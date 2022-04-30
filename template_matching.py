import cv2
import numpy as np


import cv2
img = cv2.imread("img.png")
crop_img = img[0:100]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)


# img = cv2.imread("img.png", 0)
# template = cv2.imread("template.png", 0)
# img2 = img.copy()
# h, w = template.shape
#
# methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
#            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
# for method in methods:
#     img2 = img.copy()
#
#     result = cv2.matchTemplate(img2, template, method)
#
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#     print(f"min_val: {min_val}, max_val: {max_val}, min_loc: {min_loc}, max_loc: {max_loc}")
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         location = min_loc
#
#     else:
#         location = max_loc
#
#     print(location)
#
#     bottom_right = (location[0] + w, location[1] + h)
#     cv2.rectangle(img2, location, bottom_right, 255, 5)
#     cv2.imshow('Match', img2)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#
templates = []
templ_shapes = []
threshold =0.7
img = cv2.imread("img.png")

for i in range(8):
    templates.append(cv2.imread("img{}.png".format(i+2),0))
    templ_shapes.append(templates[i].shape[:: -1])
    # print(templates[i].shape[:: -1])
    print(templ_shapes[i])


def doTemplateMatch(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i, template in enumerate(templates):
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print(f"min_val: {min_val}, max_val: {max_val}, min_loc: {min_loc}, max_loc: {max_loc}")
        location = max_loc
        if max_val > threshold:
            bottom_right = (location[0] + templ_shapes[i][0], location[1] + templ_shapes[i][1])
            cv2.rectangle(img, location, bottom_right, 255, 5)
            cv2.imshow('Match', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # loc = np.where(res >= threshold)
        # if len(loc) != 0:
        #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(loc)
        #     location = min_loc
        #     print("loc: ", loc)
        #     print("location: ", location)
        #     bottom_right = (location[0] + templ_shapes[i][0], location[1] + templ_shapes[i][1])
        #     cv2.rectangle(img, location, bottom_right, 255, 5)
        #     cv2.imshow('Match', img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()



doTemplateMatch(img)

