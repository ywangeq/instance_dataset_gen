# Write a new version with parameter input.

## for the generator of instance segmentation

### for Kitti
**generator.ipynb** is used to create kitti label(txt) from raw_data(bin) 
    its path was set to 'Kitti/2011_09_26'

**path_generator.py** is to get the 'test.txt' which inclund the data_path and label_path
**data_get.ipynb** will read the path in 'text.txt' and produce the raw_data and instance_label


####
If you want to generate the semantic segmentation label, it is very easy to modified this data_get.ipynb file  to get.
The most path in the file are hard, so please check if there are some small issues.