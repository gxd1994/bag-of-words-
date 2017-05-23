import numpy as np
import scipy.cluster.vq as vq

import cv2
import os,random,shutil
import cPickle

from skimage.feature import hog,local_binary_pattern

from matplotlib import pyplot as plt

from sklearn import svm
from sklearn.decomposition import PCA
# from mahotas.features import surf
from sklearn import preprocessing,metrics
#from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans


from data_augment import Data_Augementaion
from data_augment import config_file as Data_Aug_Cfg
from anmial_classifier import anmial_classifier
import config as cfg




class object_classfication():

	def __init__(self,config):
		self.cfg = config

		self.train_imgs = None
		self.train_labels = None
		self.train_features  = None

		self.test_imgs = None
		self.test_labels  = None
		self.test_features = None

		self.label_dict = None
		self.label_dict_inverse = None


	def train_val_make(self,data_root,dst_root,trian_test_ratio = 0.75):
		if not os.path.exists(data_root):
			print "data_root is not correct"
			return 

		if os.path.exists(dst_root):
			shutil.rmtree(dst_root)

		train_root = os.path.join(dst_root,"train_imgs")
		test_root  = os.path.join(dst_root,"test_imgs")

		shutil.copytree(data_root,train_root)

		for root,dirs,files in os.walk(train_root):
			for subdir in dirs:
				files = os.listdir(os.path.join(root,subdir))
				examples  = random.sample(files,int((1-trian_test_ratio)*len(files)))
				if os.path.exists(os.path.join(test_root,subdir)) == False:
					os.makedirs(os.path.join(test_root,subdir))
				for example in examples:
					shutil.move(os.path.join(root,subdir,example),os.path.join(test_root,subdir,example))

	def imgs_preprocess(self):


		data_root = self.cfg.data_root
		train_val_root = self.cfg.train_val_root
		trian_val_ratio = self.cfg.train_val_ratio 
		self.train_val_make(data_root,train_val_root,trian_val_ratio)


		train_root =  os.path.join(self.cfg.train_val_root,"train_imgs")
		test_root  = os.path.join(self.cfg.train_val_root,"test_imgs")

		label_txt = self.cfg.label_txt

		data_augment_root = self.cfg.data_augment_root

		img_size = self.cfg.img_size


		label_list = np.loadtxt(label_txt,dtype=str,delimiter='\n')

		self.label_dict = dict(zip(label_list,[i for i in range(len(label_list))]))   #get label_dict 
		self.label_dict_inverse = dict(zip([i for i in range(len(label_list))],label_list))   #get label_dict 


		# data_file = "./data_processed.npz" 

		# if os.path.exists(data_file):
		# 	data = np.load(data_file)

		# 	self.train_imgs = data['train_imgs']
		# 	self.train_labels = data['train_labels']
		# 	self.test_imgs = data['test_imgs']
		# 	self.test_labels = data['test_labels']

		# 	return 


		train_data = self.get_data(train_root,self.label_dict,img_size)
		test_data = self.get_data(test_root,self.label_dict,img_size)



		self.train_imgs = train_data[:,0:-1]
		self.train_labels = train_data[:,-1].astype(int)

		self.test_imgs = test_data[:,0:-1]
		self.test_labels = test_data[:,-1].astype(int)

		data_aug_cfg_obj = Data_Aug_Cfg()

		data_aug_cfg_obj.src_root = train_root
		data_aug_cfg_obj.dst_root = data_augment_root

		data_aug_cfg_obj.TARGET_HEIGHT = img_size[0]
		data_aug_cfg_obj.TARGET_WIDTH = img_size[1]

		data_aug_obj = Data_Augementaion(data_aug_cfg_obj)

		data_aug_obj.creat_data()

		print "creat_augment_data_done........."


		train_data = self.get_data(data_augment_root,self.label_dict,img_size)

		np.random.shuffle(train_data)

		self.train_imgs = train_data[:,0:-1]
		self.train_labels = train_data[:,-1].astype(int)

		# np.savez(data_file,train_imgs = self.train_imgs,train_labels = self.train_labels,
		# 				  test_imgs = self.test_imgs,test_labels = self.test_labels)



		return self.train_imgs,self.train_labels,self.test_imgs,self.test_labels,self.label_dict



	def get_data(self,file_root,label_dict,size):

		if not os.path.exists(file_root):
			print "fille not exists,please check path"

		data = None
		for root,dirs,files in os.walk(file_root):
			for subdir in dirs:
				files = os.listdir(os.path.join(root,subdir))
				for file in files:
					img = cv2.imread(os.path.join(root,subdir,file),0)
					# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

					# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2HSV_FULL)
					# img_gray = img_gray[:,:,0]+1
					# print img_gray.dtype,img_gray.min(),img_gray.max()

					img_gray_resize = cv2.resize(img,size)

					# cv2.imshow("%s"%file,img_gray_resize)
					# print img_gray_resize.shape

					img_gray_resize = img_gray_resize.reshape(1,-1)

					# print img_gray_resize.shape

					if data is not None:
						img_gray_resize = np.concatenate((img_gray_resize,np.array([label_dict[subdir]],dtype=np.uint8).reshape(1,1)),axis = 1)
						data = np.concatenate((data,img_gray_resize),axis = 0)	
					else:
						img_gray_resize = np.concatenate((img_gray_resize,np.array([label_dict[subdir]],dtype=np.uint8).reshape(1,1)),axis = 1)
						data = np.copy(img_gray_resize)					

		return data


	def compute_mean(self,data):
		mean = np.mean(data)
		print "imgs_mean:%5f"%mean
		return mean


	def imgs_normalization(self,train_data,test_data):
		# mean =133.368
		# # mean = compute_mean(data)
		# train_data = train_data - mean
		# for i in range(train_data.shape[0]):
		# 	min = np.min(train_data[i])
		# 	max = np.max(train_data[i])
		# 	train_data[i] = (train_data[i]-min)/(max-min)

		# test_data = test_data - mean
		# for i in range(test_data.shape[0]):
		# 	min = np.min(test_data[i])
		# 	max = np.max(test_data[i])
		# 	test_data[i] = (test_data[i]-min)/(max-min)


		# mean = np.mean(train_data,axis = 0)
		# train_data = train_data - mean
		# test_data = test_data - mean

		mean = np.mean(train_data,axis = 1)
		for i in range(train_data.shape[0]):
			train_data[i] = train_data[i] - mean[i]


		# mean = np.mean(test_data,axis = 1)
		for i in range(test_data.shape[0]):
			test_data[i] = test_data[i] - mean[i]


	 # 	train_data = preprocessing.minmax_scale(train_data,axis = 1)
	 # 	test_data = preprocessing.minmax_scale(test_data,axis = 1)

		return train_data,test_data



	def img_Standardization(self,train_data,test_data):

		train_data = train_data.T
		scalar = preprocessing.StandardScaler()
		train_data = scalar.fit_transform(train_data)

		test_data = test_data.T
		test_data = scalar.fit_transform(test_data)

		return train_data.T,test_data.T

	def data_Standardization(self,train_data,test_data):

		train_data = preprocessing.scale(train_data)

		test_data = preprocessing.scale(test_data)

		return train_data,test_data

	def data_normalizer(self,train_data,test_data):


		train_data = preprocessing.normalize(train_data)

		test_data = preprocessing.normalize(test_data)

		return train_data,test_data

	def hog_meathod(self,data,size,orientations,pixels_per_cell,cells_per_block,block_norm,visualise,transform_sqrt):
		num_examples = data.shape[0]
		feat = None
		for i in range(num_examples):
			ft = hog(data[i,:].reshape(size),orientations= orientations, pixels_per_cell= pixels_per_cell,cells_per_block=cells_per_block, 
									block_norm = block_norm,visualise = visualise, transform_sqrt = transform_sqrt)
			ft = ft.reshape(1,-1)
			# cv2.imshow("src",data[i,:].reshape(size))
			# cv2.imshow("hog",vis_img)
			# cv2.waitKey()

			if feat is not None:
				feat = np.concatenate((feat,ft),axis = 0)
			else :
				feat = np.copy(ft)
		return feat

	def surf_meathod(self,data,size):
		num_examples = data.shape[0]
		feat = None
		for i in range(num_examples):
			ft = surf.surf(data[i,:].reshape(size))
			print "orgin",ft.shape
			ft = ft.reshape(1,-1)
			print ft.shape
			if feat != None:
				feat = np.concatenate((feat,ft),axis = 0)
			else:
				feat = np.copy(ft)

		return feat

	def LBP_meathod(self,data,size):
		radius = 2
		n_point = radius * 8
		feat = None
		num_examples = data.shape[0]
		for i in range(num_examples):
			lbp = local_binary_pattern(data[i,:].reshape(size),n_point,radius,'ror')
			# max_bins = int(lbp.max()+1)
			max_bins = 256
			ft,_ = np.histogram(lbp,bins = max_bins,normed = True,range =(0,max_bins))
			ft = ft.reshape(1,-1)
			if feat is not None:
				feat = np.concatenate((feat,ft),axis = 0)
			else:
				feat = np.copy(ft)


		return feat

	def sift_meathod(self,data,size,name):

		# data_file = "./%s_feature_data.npz"%name

		# if os.path.exists(data_file):
		# 	data = np.load(data_file)
		# 	feat = data["feat"]
		# 	img_descriptor_index = data["img_descriptor_index"]

		# 	return feat,img_descriptor_index

		num_examples = data.shape[0]
		feat = None
		img_descriptor_index = []

		sift = cv2.SIFT()
		for i in range(num_examples):
			img  = data[i,:].reshape(size)
			kp,ft= sift.detectAndCompute(img,None)

			img_descriptor_index.append(ft.shape[0])
			# print "orgin",ft.shape[0]
			if feat is not None:
				feat = np.concatenate((feat,ft),axis = 0)
			else:
				feat = np.copy(ft)

		img_descriptor_index = np.array(img_descriptor_index,dtype = np.int)
		img_descriptor_index = np.cumsum(img_descriptor_index)

		print "all sift feature",feat.shape,num_examples
		# print "img_descriptor_index.shape",img_descriptor_index.shape
		# print "img_descriptor_index",img_descriptor_index

		if img_descriptor_index[-1] != feat.shape[0]:
			print "img_descriptor_index is fail"


		# data = np.savez(data_file,feat = feat,img_descriptor_index = img_descriptor_index)


		return feat,img_descriptor_index

	def get_codebook(self,feat,val):

		codebook_path = "./codebook.file"

		if val != True:
			n_clusters = 100
			K_THRESH = 1e-5

			# whitened = vq.whiten(feat)

			# codebook,distortion = vq.kmeans(whitened,n_clusters,thresh = K_THRESH)

			kmeans = KMeans(n_clusters = n_clusters).fit(feat)

			# codebook = kmenas.cluster_centers_

			# print codebook.shape

			if os.path.exists(codebook_path):
				os.remove(codebook_path)

			with open(codebook_path,'wb') as f:
				cPickle.dump(kmeans,f)

		else:
			with open(codebook_path,'rb') as f:
				kmeans = cPickle.load(f)

		return kmeans


	def get_img_represent(self,feat,img_descriptor_index,codebook):

		bow_feat = None

		# whitened = vq.whiten(feat)

		# code,dist = vq.vq(whitened,codebook)

		code = codebook.predict(feat)
		len_cluster_centers = codebook.cluster_centers_.shape[0]


		for i in range(img_descriptor_index.shape[0]):
			if i == 0:
				img_code = code[0:img_descriptor_index[i]]
			else:
				img_code = code[img_descriptor_index[i-1]:img_descriptor_index[i]]

			histogram_of_words, bin_edges = np.histogram(img_code,bins = range(len_cluster_centers+1),normed = True)

			# print "histogram_of_words shape",histogram_of_words.shape,histogram_of_words[10]

			histogram_of_words = histogram_of_words.reshape(1,-1)

			if bow_feat is not None:
				bow_feat = np.concatenate((bow_feat,histogram_of_words),axis= 0)
			else:
				bow_feat = np.copy(histogram_of_words)

		return bow_feat


	def get_img_represent_vq(self,feat,img_descriptor_index,codebook):

		bow_feat = None

		# whitened = vq.whiten(feat)

		code,dist = vq.vq(feat,codebook)


		len_cluster_centers = codebook.shape[0]


		for i in range(img_descriptor_index.shape[0]):
			if i == 0:
				img_code = code[0:img_descriptor_index[i]]
			else:
				img_code = code[img_descriptor_index[i-1]:img_descriptor_index[i]]

			histogram_of_words, bin_edges = np.histogram(img_code,bins = range(len_cluster_centers+1),normed = True)

			# print "histogram_of_words shape",histogram_of_words.shape,histogram_of_words[10]

			histogram_of_words = histogram_of_words.reshape(1,-1)

			if bow_feat is not None:
				bow_feat = np.concatenate((bow_feat,histogram_of_words),axis= 0)
			else:
				bow_feat = np.copy(histogram_of_words)

		return bow_feat


	def bag_of_visual_word(self,data,size,val,name):

		sift_feat,img_descriptor_index = self.sift_meathod(data,size,name)

		codebook = self.get_codebook(sift_feat,val)


		feat = self.get_img_represent(sift_feat,img_descriptor_index,codebook)

		return feat
		# print kmeans.labels_ ,kmeans.cluster_centers_

	def bag_of_visual_word_category(self,data,size,val,name):

		category_num = len(self.label_dict)
		print "category num ",category_num

		codebook = None

		n_clusters = 100

		codebook_path = self.cfg.codebook_save_path

		if val != True:

			for i in range(category_num):
				category_imgs = self.train_imgs[np.where(self.train_labels == i)]

				sift_feat,img_descriptor_index = self.sift_meathod(category_imgs,size,name)

				# kmeans = KMeans(n_clusters = n_clusters).fit(sift_feat)

				codebook_part,distortion = vq.kmeans(sift_feat,n_clusters ,thresh = 0.001)

				if codebook is not None:
					codebook = np.concatenate((codebook,codebook_part),axis = 0)
				else :
					codebook = np.copy(codebook_part)

			print "codebook.shape",codebook.shape

			if os.path.exists(codebook_path):
				os.remove(codebook_path)

			with open(codebook_path,'wb') as f:
				cPickle.dump(codebook,f)

		else:

			if not os.path.exists(codebook_path):
				print "do not exists codebook file ..please train data first"
				return

			with open(codebook_path,'rb') as f:
				codebook = cPickle.load(f)
				print "codebook.shape",codebook.shape


		sift_feat,img_descriptor_index = self.sift_meathod(data,size,name)


		feat = self.get_img_represent_vq(sift_feat,img_descriptor_index,codebook)

	
		return feat

	def point_distribute(self,point,cord_list):
		x = point[0]
		y = point[1]

		for i in range(len(cord_list)):
			if x >= cord_list[i][0] and x < cord_list[i][1] \
				 and y >= cord_list[i][2] and y < cord_list[i][3]:
				 return i

	def points_distribute(self,points,ft,cord_list,num_partion_axis):
		num_point = ft.shape[0]
		ft_list = [[] for i in range(num_partion_axis*num_partion_axis)]

		for i in range(num_point):
			index = self.point_distribute((points[i].pt[0],points[i].pt[1]),cord_list)
			ft_list[index].append(ft[i])

		return ft_list

	def get_codebook_partion(self,imgs,size):


		partion_weight = self.cfg.weights_matrix


		num_points = self.cfg.num_points_each_image

		num_partion_axis = partion_weight.shape[0]

		h,w = size

		h_list = list(np.linspace(0,h,num_partion_axis+1))
		w_list = list(np.linspace(0,w,num_partion_axis+1))

		cord_list = []
		for i in range(num_partion_axis):
			for j in range(num_partion_axis):
				x1 = w_list[j]
				x2 = w_list[j+1]

				y1 = h_list[i]
				y2 = h_list[i+1]

				cord_list.append([x1,x2,y1,y2])

		# print cord_list



		num_imgs = imgs.shape[0]
		sift = cv2.SIFT()

		img_feat  = None
		for i in range(num_imgs):

			img = imgs[i].reshape(size)

			kp,ft = sift.detectAndCompute(img,None)

			#one image gets kpoints and features , select kpoints 
			ft_list = self.points_distribute(kp,ft,cord_list,num_partion_axis)

			num_points_partion = partion_weight * num_points
			num_points_partion = num_points_partion.flatten().astype(np.int)

			# one image to get fearute,use regular grid (weights)
			for i in range(num_partion_axis*num_partion_axis):
				num_ft_list_element = len(ft_list[i])
				num_sample_part = num_points_partion[i] if num_ft_list_element >= num_points_partion[i] else num_ft_list_element
				if  num_ft_list_element != 0 and num_sample_part != 0:
					img_feat_part_list = random.sample(ft_list[i],num_sample_part)
					img_feat_part = np.vstack(img_feat_part_list)
					if img_feat is not None:
						img_feat = np.concatenate((img_feat,img_feat_part),axis = 0)
					else:
						img_feat = np.copy(img_feat_part)

		codebook  = img_feat

		print "category_codebook_shape:",codebook.shape

		return codebook




	def bag_of_visual_word_category_partion(self,data,size,val,name):

		category_num = len(self.label_dict)
		print "category num",category_num

		codebook = None

		codebook_path = self.cfg.codebook_save_path


		if val != True:

			for i in range(category_num):
				category_imgs = self.train_imgs[np.where(self.train_labels == i)]

				codebook_part = self.get_codebook_partion(category_imgs,size)


				if codebook is not None:
					codebook = np.concatenate((codebook,codebook_part),axis = 0)
				else :
					codebook = np.copy(codebook_part)

			print "created........ all_codebook_shape",codebook.shape

			if os.path.exists(codebook_path):
				os.remove(codebook_path)

			with open(codebook_path,'wb') as f:
				cPickle.dump(codebook,f)

		else:

			if not os.path.exists(codebook_path):
				print "do not exists codebook file ..please train data first"
				return

			with open(codebook_path,'rb') as f:
				codebook = cPickle.load(f)
				print "load codebook.......all_codebook_shape",codebook.shape



		sift_feat,img_descriptor_index = self.sift_meathod(data,size,name)


		feat = self.get_img_represent_vq(sift_feat,img_descriptor_index,codebook)

		return feat



	def bag_of_visual_word_category_partion_cluster(self,data,size,val,name):

		category_num = len(self.label_dict)
		print "category num",category_num

		codebook = None

		codebook_path = self.cfg.codebook_save_path 

		n_clusters = self.cfg.num_category_cluster

		if val != True:

			for i in range(category_num):
				category_imgs = self.train_imgs[np.where(self.train_labels == i)]

				pre_codebook_part = self.get_codebook_partion(category_imgs,size)


				codebook_part,distortion = vq.kmeans(pre_codebook_part,n_clusters ,thresh = 0.001)



				if codebook is not None:
					codebook = np.concatenate((codebook,codebook_part),axis = 0)
				else :
					codebook = np.copy(codebook_part)

			print "created........ all_codebook_shape",codebook.shape

			if os.path.exists(codebook_path):
				os.remove(codebook_path)

			codebook_path_dirname = os.path.dirname(codebook_path)
			if not os.path.exists(codebook_path_dirname):
				os.makedirs(codebook_path_dirname)


			with open(codebook_path,'wb') as f:
				cPickle.dump(codebook,f)

		else:
			if not os.path.exists(codebook_path):
				print "do not exists codebook file ..please train data first"
				return

			with open(codebook_path,'rb') as f:
				codebook = cPickle.load(f)
				print "load codebook.......all_codebook_shape",codebook.shape


		sift_feat,img_descriptor_index = self.sift_meathod(data,size,name)


		feat = self.get_img_represent_vq(sift_feat,img_descriptor_index,codebook)

		return feat


	def extract_features(self,data,size,val,name):


		fearute_meathod = self.cfg.feature_meathod[0]

		if fearute_meathod == "hog_meathod":

			feat = self.hog_meathod(data,size,orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),
									block_norm='L2-Hys', visualise=False, transform_sqrt=True)
		elif fearute_meathod == "LBP_meathod":

			feat = LBP_meathod(data,size)

		elif fearute_meathod == "bag_of_visual_word_category_partion_cluster":

			feat = self.bag_of_visual_word_category_partion_cluster(data,size,val = val,name = name)

		elif fearute_meathod == "bag_of_visual_word_category_partion":

			feat = self.bag_of_visual_word_category_partion(data,size,val = val,name = name)

		else :
			print "please chose correct feature meathod"
			return 


		return feat


	def knn_classifier(self,train_x, train_y,test_x,test_y,label_dict,val = False): 
		from sklearn.neighbors import KNeighborsClassifier    
		model = KNeighborsClassifier()    
		model.fit(train_x, train_y)    

		result = model.predict(test_x)

		label_dict_inverse = dict([(v,k) for k,v in label_dict.iteritems()])
		target_names = [label_dict_inverse[i]  for i in xrange(len(label_dict_inverse))]

		print metrics.classification_report(test_y,result,target_names = target_names) 

		print "(lab,pre):",zip([label_dict_inverse[i] for i in list(test_y)],[label_dict_inverse[i] for i in list(result)])



	# SVM Classifier    
	def svm_classifier(self,train_x, train_y,test_x,test_y,label_dict,val = False): 

		model_save_path = self.cfg.classifier_save_path



		#model = svm.SVC(C=10,gamma=0.0001)
		model = svm.SVC()

		model.fit(train_x, train_y)


		label_dict_inverse = dict([(v,k) for k,v in label_dict.iteritems()])
		target_names = [label_dict_inverse[i]  for i in xrange(len(label_dict_inverse))]


		if val == True:

			result = model.predict(train_x)


			print metrics.classification_report(train_y,result,target_names = target_names) 



		result = model.predict(test_x)

		print metrics.classification_report(test_y,result,target_names = target_names) 

		print "(lab,pre):",zip([label_dict_inverse[i] for i in list(test_y)],[label_dict_inverse[i] for i in list(result)])



	    # # prob = model.predict_proba(test_x)
	    # test_score = model.score(test_x,test_y)

	    # # print test_y
	    # # print "prob:\n",prob

	    # if val:
	    # 	train_score = model.score(train_x,train_y)
	    # 	print "train_accuracy:",train_score

	    # print "test_accuracy:",test_score


	    # model = svm.LinearSVC()#kernel="linear",probability = True)

	    # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],  
	    #          	 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}  

	    # model = GridSearchCV(svm.SVC(kernel='rbf'), param_grid)


	    # model = svm.SVC(probability = False)
	    # model.fit(train_x, train_y)
	    # result = model.predict(test_x)

	    # print "%s"%(metrics.classification_report(test_y,result))

	    # correct_prediction = np.equal(test_y,result)
	    # accuracy = np.mean(correct_prediction.astype(float))
	    # print correct_prediction
	    # print accuracy


	def PCA_meathod(self,test):

		if test == False:

			PCA_save_path = self.cfg.PCA_save_path

			#n_components = 500
			# pca = PCA(n_components = 420)

			pca = PCA()

			self.train_features = pca.fit_transform(self.train_features)
			self.test_features = pca.transform(self.test_features)

			explained_variance_ratio_ = np.cumsum(pca.explained_variance_ratio_)
			print explained_variance_ratio_
			print np.where(abs(explained_variance_ratio_-1.0) < 1e-5)

			
			if not os.path.exists(os.path.dirname(PCA_save_path)):
				os.makedirs(os.path.dirname(PCA_save_path))

			cPickle.dump(pca,open(PCA_save_path,'wb'))


		else:

			PCA_save_path = self.cfg.PCA_save_path
			
			if not os.path.exists(PCA_save_path):
				print "does not exists pca model,please train data first"

			pca = cPickle.load(open(PCA_save_path,'rb'))

			self.test_features = pca.transform(self.test_features)


	def train(self):

		print "start img preprocess......"

		self.imgs_preprocess()

		print "start extact features......"

		print "start extact train set features......"
		self.train_features = self.extract_features(self.train_imgs,self.cfg.img_size,val = False,name = "train")
		print "start extact val set features......"
		self.test_features = self.extract_features(self.test_imgs,self.cfg.img_size,val = True,name = "val")


		# self.train_features = self.bag_of_visual_word(self.train_imgs,self.cfg.img_size,val = False,name = "train")
		# self.test_features = self.bag_of_visual_word(self.test_imgs,self.cfg.img_size,val = True,name = "val")

		# self.train_features = self.bag_of_visual_word_category(self.train_imgs,self.cfg.img_size,val = False,name = "train")
		# self.test_features = self.bag_of_visual_word_category(self.test_imgs,self.cfg.img_size,val = True,name = "val")

		# self.train_features = self.bag_of_visual_word_category_partion(self.train_imgs,self.cfg.img_size,val = False,name = "train")
		# self.test_features = self.bag_of_visual_word_category_partion(self.test_imgs,self.cfg.img_size,val = True,name = "val")

		# self.train_features = self.bag_of_visual_word_category_partion_cluster(self.train_imgs,self.cfg.img_size,val = False,name = "train")
		# self.test_features = self.bag_of_visual_word_category_partion_cluster(self.test_imgs,self.cfg.img_size,val = True,name = "val")



		# self.train_features, self.test_features = self.data_Standardization(self.train_features,self.test_features)

		self.PCA_meathod(test = False)


		self.train_features, self.test_features = self.data_Standardization(self.train_features,self.test_features)

		print "train feature and labels:",self.train_features.shape,self.train_labels.shape
		print "test feature and labels",self.test_features.shape,self.test_labels.shape	


		print "start classifiy features......."

		# # # select_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM','SVMCV', 'GBDT']

		# select_classifiers = ['SVM']

		select_classifiers = self.cfg.select_classifiers

		anmial_classifier(self.cfg,select_classifiers,self.train_features,self.train_labels,self.test_features,self.test_labels,self.label_dict,val = True,test = False)


		# self.svm_classifier(self.train_features,self.train_labels,self.test_features,self.test_labels,self.label_dict,val = True)
		# self.knn_classifier(self.train_features,self.train_labels,self.test_features,self.test_labels,self.label_dict,val = True)


		# pipe = make_pipeline(PCA(),svm.SVC())
		# param_grid = {'pca__n_components':[10,20,50,80,100,200,300,400,500,600,700,800,900,1000,1500],
		# 'svc__C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000,10000], 'svc__gamma': [1,0.1,0.01,0.001, 0.0001]}

		# grid_search = GridSearchCV(pipe, param_grid, n_jobs = 1, verbose=1)  
		
		# grid_search.fit(train_features, train_labels) 

		# best_parameters = grid_search.best_estimator_.get_params() 
		# for para, val in list(best_parameters.items()):    
		# 	print(para, val) 


		# predit = grid_search.predict(test_features,test_labels)

		# accuracy = metrics.accuracy_score(test_labels,predict)

		# print('accuracy: %.2f%%' % (100 * accuracy))


	def test(self):

		# get label_dict
		label_txt = self.cfg.label_txt
		label_list = np.loadtxt(label_txt,dtype=str,delimiter='\n')
		self.label_dict = dict(zip(label_list,[i for i in range(len(label_list))]))   #get label_dict 
		self.label_dict_inverse = dict(zip([i for i in range(len(label_list))],label_list))   #get label_dict 


		#get test_data and label

		test_data = self.get_data(self.cfg.test_root,self.label_dict,self.cfg.img_size)

		self.test_imgs = test_data[:,0:-1]
		self.test_labels = test_data[:,-1].astype(int)


		#get feature
		self.test_features = self.extract_features(self.test_imgs,self.cfg.img_size,val = True,name = "val")

		self.PCA_meathod(test = True)

		self.test_features = preprocessing.scale(self.test_features)


		print "test feature and labels",self.test_features.shape,self.test_labels.shape	


		# classifiy
		select_classifiers = self.cfg.select_classifiers
		anmial_classifier(self.cfg,select_classifiers,None,None,self.test_features,self.test_labels,self.label_dict,val = True,test = True)



def main():

	# cfg = config_file()
	
	if cfg.state == 'TRAIN':

		anmial_clf = object_classfication(cfg)
		anmial_clf.train()

	elif cfg.state == "TEST":
		if not os.path.exists(cfg.test_root):
			print "test_root is not exists,or rename the test_root"
			return

		anmial_clf = object_classfication(cfg)
		anmial_clf.test()

	else:

		print "config state must be 'TRAIN' or 'TEST'"


if __name__ == "__main__":
    main()

   