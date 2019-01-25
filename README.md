## A python version is under the Test_data file

### Step1
    download the raw data (2011_09_26_drive_0002_sync....) in the Test_data/2011_09_26
    
## Step2
    run 'python generator.py' to get the each label
## Step3
    run ' python path_genertor.py' to get the test.txt which contains the prepared data
## Step4
    run 'python gene_data.py'  the Kitti/npydata will store all the dataset,
    I concat the data[:,:,:4] and the mask [:,:,5] in the one file. So each file in the Killt/npydata will have the shape 64 * 512 *6

### Attention
When you try to use the Step, you need to change the index in the line 400:
            id_name = para.name[46:67] # 2011_09_26_drive_000
            id_root = para.name[79:89] # 00000001
To get the right name depend on you root 