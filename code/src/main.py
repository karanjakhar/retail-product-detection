import numpy as np 
import cv2
from tqdm import tqdm
import os
from glob import glob

THRESHOLD_MATCH_COUNT = 30


def resize_image(img,scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return (width,height)




product_dir = '/home/karan/kj_workspace/kj_random_work/infilect/data/product_images'
shelf_dir = '/home/karan/kj_workspace/kj_random_work/infilect/data/shelf_images'       
encircle_width = 35
sift_dict = {}
sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

for i in tqdm(range(1,301)):
    if i%3 != 1:
        continue
    product_image_path = os.path.join(product_dir,f'qr{i}.jpg')
    total_match = 0
    product_img = cv2.imread(product_image_path,0)
    dim_product_img = resize_image(product_img,50)
    product_img = cv2.resize(product_img, dim_product_img, interpolation = cv2.INTER_AREA)
    p_shape = product_img.shape
    product_kp, product_des = sift.detectAndCompute(product_img,None)
    sift_dict[product_image_path] = (product_kp, product_des)
print(p_shape)        

for i in tqdm(range(1,len(glob(os.path.join(shelf_dir,'*'))))):
    shelf_image_path = os.path.join(shelf_dir,f'db{i}.jpg')
    shelf_img = cv2.imread(shelf_image_path,0)
    shelf_kp, shelf_des = sift.detectAndCompute(shelf_img,None)
    sift_dict[shelf_image_path] = (shelf_kp, shelf_des)



def matching(product_id):
    print('Started:',product_id)
    product_path = os.path.join(product_dir,f'qr{product_id}.jpg')
    product_kp,product_des = sift_dict[product_path]
    total_match = 0
    for i in tqdm(range(1,len(glob(os.path.join(shelf_dir,'*'))))):
            shelf_image_path = os.path.join(shelf_dir,f'db{i}.jpg')
            shelf_kp, shelf_des = sift_dict[shelf_image_path]        
            
            matches = flann.knnMatch(product_des,shelf_des,k=2)


            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
            #print('Good:',len(good))

            if len(good)>THRESHOLD_MATCH_COUNT:
                shelf_id = i
                try:
                    src_pts = np.float32([ product_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                    dst_pts = np.float32([ shelf_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                    

                    h,w = 972, 1296
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts,M)
                    print(dst)
                    x1,y1,x2,y2 = int(dst[0,0,0]), int(dst[0,0,1]),int(dst[3,0,0]), int(dst[3,0,1])
                    x1 = max(x1,0)
                    y1 = max(y1,0)
                    x2 = max(x2,0)
                    y2 = max(y2,0)
                    if (x2-x1) * (y2-y1) > 30:
                        solution1.append([product_id,shelf_id,x1,y1,x2,y2])
                        solution2.append([shelf_id,product_id,x1,y1,x2,y2])
                        total_match += 1
                except:
                    pass

solution1 = []
solution2 = []
for i in range(1,52):
    if i%3 != 1:
        continue
    matching(i)
    with open('solutions_1.txt','a') as file:
        for r in solution1:
            for ri in r:
                file.write(str(ri))
                file.write(',')
            file.write('\n')
        solution1 = []


    with open('solutions_2.txt','a') as file:
        for r in solution2:
            for ri in r:
                file.write(str(ri))
                file.write(',')
            file.write('\n')
        solution2 = []
    
print("All Done")
