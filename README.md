#Download training, validation and testing data for ShapeNet dataset.
Run prepareDataset.sh

#BaiduDisk is supported as this moment. Download and unzip to the ./ directory of this project. You should keep the
directories like ./Dataset/ShapeNet/hdf5_data/xxx

#### Active Training
Run training script at ./RandomSamp/train_DGCNN_SP_AL.py
Parameters like annotation budget need to be set

#### Active Strategies
Different active learning strategies in ./Strategies
