## write for the generator of instance segmentation

### for Kitti
**generator.ipynb** is used to create kitti label(txt) from raw_data(bin) 
    its path was set to 'Kitti/2011_09_26'

**path_generator.py** is to get the 'test.txt' which inclund the data_path and label_path
**data_get.ipynb** will read the path in 'text.txt' and produce the raw_data and instance_label

    raw_data: x, y, z, itensity, distance  it can be set to save from[:,:,:5]
    
    instance_label is saved as npy file with [height,width,num_object]# instance_dataset_gen
