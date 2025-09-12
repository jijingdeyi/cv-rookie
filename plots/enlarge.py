import cv2
import matplotlib.pyplot as plt



def Partial_magnification(pic, target, location='lower_right', ratio=1):
    
    '''
    :param pic: input pic
    :param target: Intercept area, for example [target_x, target_y, target_w, target_h]
    :param location: lower_right,lower_left,top_right,top_left,center
    :param ratio: gain
    :return: oringal pic, pic
    '''

    w, h = pic.shape[1], pic.shape[0],
    target_x, target_y = target[0], target[1]
    target_w, target_h = target[2]-target[0], target[3]-target[1]
    cv2.rectangle(pic, (target_x, target_y), (target_x + target_w, target_y + target_h), (255, 255, 0), 2)
    new_pic = pic[target_y:target_y + target_h, target_x:target_x + target_w]
    new_pic = cv2.resize(new_pic, (target_w*ratio, target_h*ratio), interpolation=cv2.INTER_CUBIC)


    if location == 'lower_right':
        pic[h-1-target_h*ratio:h-1, w-1-target_w*ratio:w-1] = new_pic
        cv2.line(pic, (target_x + target_w, target_y + target_h), (w-1-target_w*ratio, h-1-target_h*ratio), (255, 0, 0),2)
    elif location == 'lower_left':
        pic[h-1-target_h*ratio:h-1, 0:target_w*ratio] = new_pic
        cv2.line(pic, (target_x, target_y + target_h), (0 + target_w*ratio, h-1-target_h*ratio), (255, 0, 0), 2)
    elif location == 'top_right':
        pic[0:target_h*ratio, w-1-target_w*ratio:w-1] = new_pic
        cv2.line(pic, (target_x + target_w, target_y), (w-1-target_w*ratio, 0 + target_h*ratio), (255, 0, 0), 2)
    elif location == 'top_left':
        pic[0:target_h*ratio, 0:target_w*ratio] = new_pic
        cv2.line(pic, (target_x, target_y), (0 + target_w*ratio, 0 + target_h*ratio), (255, 0, 0), 2)
    elif location == 'center':
        pic[int(h/2-target_h*ratio/2):int(h/2+target_h*ratio/2),
            int(w/2-target_w*ratio/2):int(w/2+target_w*ratio/2)] = new_pic
        
    return pic



if __name__ == '__main__':
    
    picture_index = ['2','9','10','15','19']
    position = [[284,4,318,39], [354,5,410,28], [304,288,382,424], [125,359,328,447],[470,267,617,353]]
    location=['lower_right','lower_right','lower_right','top_right','top_left']
    ratio = [ 4, 4, 3, 2, 2]

    for i in range(5):

        readpath = r'D:\SOTA\SDNet\IR_VIS\result\epoch10\{}.png'.format(picture_index[i])
        img = cv2.imread(readpath)  
        result = Partial_magnification(img, position[i], location=location[i], ratio=ratio[i])
        cv2.imwrite(r'D:/SOTA/visual_test/{}/sdnet.png'.format(picture_index[i]), result)


        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize=(20, 10))  # figsize 尺寸
        plt.imshow(result)
        plt.show()
