import os
def rn_imgs(root_path):
    for i in range(360, 2881):
        img_name = 'frame_'+str(i).zfill(4) + '.png'
        path = os.path.join(root_path, img_name)
        print(path)
        # if os.path.isfile(path):
        os.remove(path)

if __name__ == '__main__':
    root_path = '/home/baizu/hyb/ProPainter/temp_file/inputs/imgs_resize'
    rn_imgs(root_path)