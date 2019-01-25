import os

root = os.getcwd()

raw_data = root+'/Kitti/2011_09_26'
type = 'label'
root_txt ='test.txt'
file_object = open(root_txt,'wb')
a =0
for path in os.listdir(raw_data):
    if path[-4:] == 'sync':
        Data_file = raw_data+'/'+path
        print(Data_file)
        for file in os.listdir(Data_file):
            if file ==type:
                lidar_data = Data_file + '/'+ file
                for data in os.listdir(lidar_data):
                    print('data',data)
                    a= a+1
                    #print(a)
                    root_name = lidar_data[:-6] +'/'+'velodyne_points/data/'+data[:-4]+'.bin'
                    lidar_name = Data_file+'/'+'label/'+data[:-4]+'.txt'
                    if os.path.exists(root_name) and os.path.exists(lidar_name):
                        print('root',root_name)
                        print('lidar',lidar_name)
                        file_object.write(root_name)
                        file_object.write(' ')
                        file_object.write(lidar_name)
                        file_object.write('\n')
                #file_object.close()