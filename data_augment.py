import cv2
import numpy as np 
import shutil,random,os
from skimage import exposure,img_as_float,img_as_ubyte

class config_file():
	def __init__(self):

		self.src_root= "./data"
		self.dst_root = "./data_augment"

		self.img_aug_meathod=[
							     # 'crop_img',
							     'rotate_img',
							     'flip_img',
							     # 'shift_img',
							     # 'blur_img',
							   #'reverse_gray_img',
							   # 'exposure_gamma_img',
							   # 'exposure_log_img',
							 ]


		self.TARGET_WIDTH = 128
		self.TARGET_HEIGHT = 128


		self.RANDOM_ITER = 6


		# crop
		self.RANDOM_CROP_LB = 0.7
		self.RANDOM_CROP_UB = 1.0

		# shift
		self.RANDOM_SHIFT_LB = 0.1
		self.RANDOM_SHIFT_UB = 0.3

		# gamma
		self.RANDOM_GAMMA_LB = 0.5
		self.RANDOM_GAMMA_UB = 1.5



		# rotate
		self.RANDOM_ROTATE_LB = -30
		self.RANDOM_ROTATE_UB = 30

		# blur
		self.RANDOM_BLUR_RADIUS_LB = 1
		self.RANDOM_BLUR_RADIUS_UB = 3




class Data_Augementaion():
	def __init__(self,cfg):
		self.cfg = cfg
	def creat_data(self):
		dst_root = self.cfg.dst_root
		src_root = self.cfg.src_root
		if not os.path.exists(src_root):
			print "src_root is not exists"
		if os.path.exists(dst_root):
			shutil.rmtree(dst_root)

		for root,dirs,files in os.walk(src_root):
			for subdir in dirs:
				if not os.path.exists(os.path.join(dst_root,subdir)):
					os.makedirs(os.path.join(dst_root,subdir))
				abs_subdir = os.path.join(root,subdir)
				count =0
				for file in os.listdir(abs_subdir):

					img = cv2.imread(os.path.join(root,subdir,file),0)

					for i in range(self.cfg.RANDOM_ITER):
						img_aug,name = self.random_augment(img)

						if self.cfg.TARGET_WIDTH:
							img_aug  = cv2.resize(img_aug,(self.cfg.TARGET_WIDTH,self.cfg.TARGET_HEIGHT))

						count += 1
						cv2.imwrite(os.path.join(dst_root,subdir,"%d%s.jpg"%(count,name)),img_aug)



	def random_augment(self,img):
		img_aug_meathod = self.cfg.img_aug_meathod
		aug_num  = len(img_aug_meathod)
		rand_aug_list = random.sample(img_aug_meathod,random.randint(1,aug_num))
		all_name = ''
		fun_dict = {
					'crop_img':self.crop_img,
					'rotate_img':self.rotate_img,
					'flip_img':self.flip_img,
					'shift_img':self.shift_img,
					'blur_img':self.blur_img,
					'reverse_gray_img':self.reverse_gray_img,
					'exposure_gamma_img':self.exposure_gamma_img,
					'exposure_log_img':self.exposure_log_img,

		}
		image_aug=img
		for rand_augment in rand_aug_list:
			# print rand_augment
			image_aug,name = fun_dict[rand_augment](image_aug)
			all_name += name 

		return image_aug,all_name

	def crop_img(self,image):

		ratio_lb = self.cfg.RANDOM_CROP_LB
		ratio_ub = self.cfg.RANDOM_CROP_UB
		height,width = image.shape
		ratio_h = random.uniform(ratio_lb, ratio_ub)
		ratio_w = random.uniform(ratio_lb, ratio_ub)
		crop_h = int(height * ratio_h)
		crop_w = int(width * ratio_w)

		x_start = random.randint(0, width-crop_w-1)
		y_start = random.randint(0, height-crop_h-1)

		image_crop = image[y_start:y_start+crop_h,x_start:x_start+crop_w]

		name = "_crop_h%dw%d"%(crop_h,crop_w)

		return image_crop,name



	def rotate_img(self,image):

		angle_lb = self.cfg.RANDOM_ROTATE_LB 
		angle_ub = self.cfg.RANDOM_ROTATE_UB

		rotate_angle = random.randint(angle_lb,angle_ub)
		
		h,w = image.shape

		M = cv2.getRotationMatrix2D((w//2,h//2),rotate_angle,1.0)
		image_rotate = cv2.warpAffine(image,M,(w,h),borderMode = cv2.BORDER_REPLICATE)#borderValue = 128)   

		name = "_rotate_%d"%rotate_angle

		return image_rotate,name


	def shift_img(self,image):
		h,w = image.shape

		y = h*random.uniform(self.cfg.RANDOM_SHIFT_LB,self.cfg.RANDOM_SHIFT_UB)
		x = w*random.uniform(self.cfg.RANDOM_SHIFT_LB,self.cfg.RANDOM_SHIFT_UB)

		# shift_choise = [(x,0),(-x,0),(0,y),(0,-y)]

		# tx,ty = shift_choise[random.randint(0,3)]
		tx = x
		ty = y

		M = np.array([[1,0,tx],[0,1,ty]],dtype=np.float32)

		image_shift = cv2.warpAffine(image,M,(w,h),borderMode = cv2.BORDER_REPLICATE)#borderValue = 128)

		name = "_shift_x%dy%d"%(tx,ty)

		return image_shift,name

	def blur_img(self,image):

		ksize = random.randint(self.cfg.RANDOM_BLUR_RADIUS_LB,self.cfg.RANDOM_BLUR_RADIUS_UB)
		if ksize%2 == 0:
			ksize += 1

		b_img = cv2.GaussianBlur(image,(ksize,ksize),0)

		name = "_gauss_k%d.jpg"%ksize

		return b_img,name


	def reverse_gray_img(self,image):

		r_img = 255-image

		name = "_reverse"

		return r_img,name


	def exposure_gamma_img(self,image):

		gam = random.uniform(self.cfg.RANDOM_GAMMA_LB,self.cfg.RANDOM_GAMMA_UB)

		src = img_as_float(image)

		gam_img = exposure.adjust_gamma(src,gam)

		gam_img = img_as_ubyte(gam_img)

		name = "_gamma_%.1f"%gam

		return gam_img,name

	def exposure_log_img(self,image):

		src = img_as_float(image)
		log_img= exposure.adjust_log(src) 
		log_img = img_as_ubyte(log_img)

		name = "_log"

		return log_img,name
	


	def flip_img(self,image):

		image = cv2.flip(image,random.randint(0,1))

		name = "_flip"

		return image,name

		# xy_img = cv2.flip(img,-1)
		# post_name = name+"xyflip.jpg"
		# cv2.imwrite(post_name,xy_img)


def main():
	cfg = config_file()
	data_aug=Data_Augementaion(cfg)
	data_aug.creat_data()

if __name__ == "__main__":
	pass
	# main()