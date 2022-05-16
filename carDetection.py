#do it only once to extract features and save it in pickle file as it takes times
def train_svc():
 colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
 orient = 9
 pix_per_cell = 8
 cell_per_block = 2
 hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
 spatial_size=(32, 32)
 hist_bins=32

 t=time.time()

 car_features = extract_features(cars, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel, hist_bins=hist_bins)
 notcar_features = extract_features(notcars, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel, hist_bins=hist_bins)
 t2 = time.time()
 print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
 X = np.vstack((car_features, notcar_features)).astype(np.float64)
 print(X.shape)
# Fit a per-column scaler
 X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
 scaled_X = X_scaler.transform(X)

# Define the labels vector
 y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

 print(len(y))

 # Split up data into randomized training and test sets
 rand_state = np.random.randint(0, 100)
 X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.15, random_state=rand_state)

 print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
 print('Feature vector length:', len(X_train[0]))

# Use a linear SVC X_scaler
 svc = LinearSVC()
# Check the training time for the SVC
 t=time.time()
 svc.fit(X_train, y_train)
 t2 = time.time()
 print(round(t2-t, 2), 'Seconds to train SVC...')

# Pickle the data as it takes a lot of time to generate it

 data_file = 'svc_pickle.p'

 if not os.path.isfile(data_file):
    with open(data_file, 'wb') as pfile:
        pickle.dump(
            {
                'svc': svc,
                'scaler': X_scaler,
                'orient': orient,
                'pix_per_cell': pix_per_cell,
                'cell_per_block': cell_per_block,
                'spatial_size': spatial_size,
                'hist_bins': hist_bins

            },
            pfile, pickle.HIGHEST_PROTOCOL)

 print('Data saved in pickle file')

####Load the training model from pickle file
dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
print(dist_pickle)
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

#Read cars and not-cars images

#test images folder
test_images_dir = 'test_images/'
#Read cars and not-cars images


# images are divided up into vehicles and non-vehicles
test_images = []

images = glob.glob(test_images_dir + '*.jpg')

for image in images:
        test_images.append(mpimg.imread(image))

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
              vis_bboxes=False):
    draw_img = np.copy(img)
    xstart = int(img.shape[1] / 5)
    xstop = img.shape[1]
    img_tosearch = img[ystart:ystop, xstart:xstop, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YUV')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    rectangles = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3)).reshape(1, -1)
            #             hog_features = np.hstack((hog_feat1))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            #             subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            #             spatial_features = bin_spatial(subimg, size=spatial_size)
            #             hist_features = color_hist(subimg, nbins=hist_bins)
            # Scale features and make a prediction
            #             stacked = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(hog_features)
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1 or vis_bboxes == True:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                rectangles.append(((xbox_left + xstart, ytop_draw + ystart),
                                   (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart)))

    return rectangles