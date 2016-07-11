#2D image segmenting
import cv2
import numpy as np
import openpyxl

def contour_sort(l):
    """Sort contours by area largest to smallest."""
    length = len(l)
    if length <= 1:
        return l
    else:
        pivot = l.pop(int(length / 2))
        less, more = [], []
        for x in l:
            if cv2.contourArea(x) >= cv2.contourArea(pivot):
                less.append(x)
            else:
                more.append(x)
        return contour_sort(less) + [pivot] + contour_sort(more)

wb = openpyxl.load_workbook('july_10_2016-data/characterized_data_dump.xlsx')
debris_id = []
shape = []
ws = wb.get_sheet_by_name('Sheet1')
skip_first_row = True
for col in ws.columns[2]:
    if skip_first_row == False:
        debris_id.append(col.value)
    else: skip_first_row = False
for col in ws.columns[22]:
    if col.value == 'Flat Plate':
        shape.append(0)
    if col.value == 'Strt. Rod/Ndl/Cyl':
        shape.append(1)
y_vec = np.asarray(shape)
des_img_size = 20
wrk_dir = 'path'
X = np.empty(shape=(312,1,20,20))
i=0
print'y', len(y_vec)
print 'debrid',len(debris_id)
for name in debris_id:
    img_file = '/DS'+str(name)
    print wrk_dir+img_file+'/revision_3'+img_file+'_backlit_capture.jpg'
    img = cv2.imread(wrk_dir+img_file+'/revision_3/'+img_file+'_backlit_capture.jpg',0)
    ret,thresh = cv2.threshold(img,70,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contour_sort(contours)
    x, y, w, h = cv2.boundingRect(contours[3])
    roi = img[y: y + h, x: x + w]
    if w > h:
        max_dem = w
    else:
        max_dem = h
    ratio = (max_dem / des_img_size)
    height, width = img.shape[:2]
    small_image = cv2.resize(img,(width/ratio,height/ratio),interpolation=cv2.INTER_AREA)
    ret,thresh = cv2.threshold(small_image,70,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contour_sort(contours)
    x, y, w, h = cv2.boundingRect(contours[3])
    x_cent=x+w/2
    y_cent=y+h/2
    roi = small_image[y_cent-des_img_size/2: y_cent + des_img_size/2, x_cent-des_img_size/2: x_cent + des_img_size/2]
    debug_image = cv2.resize(roi,(200,200),interpolation=cv2.INTER_LINEAR)
    X[i,:,:,:]=roi
    #cv2.imshow(str(y_vec[i]),debug_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    i=i+1
X_train = X[0:285,:]
X_test = X[285:311,:]
y_train = y_vec[0:285]
y_test = y_vec[285:311]
np.savez('july_10_2016-data/data_set',X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)