import numpy as np



state =  "TEST"  #"TEST"  #'TRAIN'        #only chose one
test_is_carry_label = True

test_root = './test'


data_root = "./data"         		      # train data root 
label_txt = "./label.txt"                 # subdir name map to number ,each subdir is category 


train_val_root = "./train_val_imgs"       # copy original dataset to it(backup)
train_val_ratio = 0.75

data_augment_root = "./data_augment"      # train set to augment


classifier_save_path = "./model_save/classifier_model.file"
PCA_save_path = "./model_save/pca_model.file"
codebook_save_path  = "./model_save/codebook.file"

img_size = (128,128)


feature_meathod = [                       #only chose one

						#"hog_meathod",
						#"LBP_meathod",
						"bag_of_visual_word_category_partion_cluster"    
						#"bag_of_visual_word_category_partion"
		
						  ]

# bag of words param

weights_matrix = np.array([  [0.1,0.4,0.1],
						     [0.4,0.4,0.4],
						     [0.1,0.4,0.1] ],dtype=np.float)


num_points_each_image = 10

num_category_cluster =  2000



select_classifiers = [
							# 'NB', 
							# 'KNN', 
							# 'LR', 
							# 'RF', 
							# 'DT', 
							'SVM',
							# 'SVMCV',
							#  'GBDT'
							 ]