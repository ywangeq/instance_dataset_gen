import parseTrackletXML as xmlParser
import os
import numpy as np

root = os.getcwd()

raw_data = root+'/Kitti/2011_09_26'
def method(dir_path,label_path,way):
    data = xmlParser.parseXML(dir_path)
    for IT,it in enumerate(data):
        h,w,l = it.size
        label = it.objectType
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in it:
            id = absoluteFrameNumber
            print( id)
            x,y,z =translation
            id= '%03d'% id
            print( id)

            path = label_path+'/0000000'+str(id)+'.txt'
            #print path
            if way=='dir':
                if os.path.exists(path):
                    print ('pass')
                else:
                    os.mknod(path)
            else:
                    file=open(path,'a')
                    print (label)
                    file.write(label)
                    file.write(' ')

                    file.write('1')
                    file.write(' ')

                    file.write('2')
                    file.write(' ')

                    file.write('3')
                    file.write(' ')

                    file.write('4')
                    file.write(' ')

                    file.write('5')
                    file.write(' ')

                    file.write('6')
                    file.write(' ')

                    file.write('7')
                    file.write(' ')

                    file.write(str(h))
                    file.write(' ')
                    file.write(str(w))
                    file.write(' ')

                    file.write(str(l))
                    file.write(' ')

                    file.write(str(x))
                    file.write(' ')

                    file.write(str(y))
                    file.write(' ')

                    file.write(str(z))
                    file.write(' ')

                    file.write(str(rotation[2]))
                    file.write('\n')
                    file.close()
def make_dir(path,way):
    for path in os.listdir(path):
        if path[-4:] == 'sync':
            Data_path = raw_data+'/'+path
            #for dir_ in os.listdir(Data_path):
            #    if dir_ == 'velodyne_points':
            #lidar = Data_path +'/'+velodyne_points+'/data'
            laber = raw_data+'/'+path+'/label'
            xml_path = Data_path +'/tracklet_labels.xml'
            print('data',xml_path,laber)

            if os.path.exists(laber):
                print('the file has been made')
            else:
                os.makedirs(laber)
                #for bin_lidar in os.listdir(lidar):
            method(xml_path,laber,way)
make_dir(raw_data,'a')