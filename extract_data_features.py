import os
import glob
from lesson_function import extract_features
import pickle
import time

def main():
    # Images are divided into vehicles and non-vehicles folders with subfolders
    # Read in image file names for all vehicle data
    basedir = 'data/vehicles/'
    # Different folders represent different sources for images e.g. GTI, KITTI
    image_dirs = os.listdir(basedir)
    cars = []
    for img_dir in image_dirs:
        cars.extend(glob.glob(basedir+img_dir+'/*'))

    print('Number of vehicle images found:', len(cars))
    # Save all vehicle file names to cars.txt
    with open('data/cars.txt', 'w') as f:
        for fname in cars:
            f.write(fname+'\n')

    # Do the same for non-vehicle images
    basedir = 'data/non-vehicles/'
    image_dirs = os.listdir(basedir)
    non_cars = []
    for img_dir in image_dirs:
        non_cars.extend(glob.glob(basedir+img_dir+'/*'))

    print('Number of non-vehicle images found:', len(non_cars))
    # Save all non-vehicle file names to non_cars.txt
    with open('data/non_cars.txt', 'w') as f:
        for fname in non_cars:
            f.write(fname+'\n')
    
    #Extract the features now 
    print('Extracing features from the datasets.')

        
    # Define feature parameters
    color_space = 'HLS'
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'
    spatial_size = (16, 16)
    hist_bins = 32
    spatial_feat = False
    hist_feat = True
    hog_feat = True

    # Log time to extract features
    t = time.time()

    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    noncar_features = extract_features(non_cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)

    # Print time for extracting features
    print(time.time()-t, 'Seconds to compute features...')

    # Save features for training classifier
    pickle.dump(car_features, open('car_features.p', 'wb'))
    pickle.dump(noncar_features, open('noncar_features.p', 'wb'))


if __name__=='__main__':
    main()