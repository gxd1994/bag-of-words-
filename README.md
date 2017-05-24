bag-of-words(main)

1.project-main (final).py 是主文件
2.config.py 是配置文件
3.data_augment.py 是数据扩展文件
4.anmial_classifier.py 是分类器文件


运行条件: sklearn 0.18.1
          skimage 0.13.0
          cv2 2.4.13
          cPickle 1.7.1
          numpy
          scipy
          matplotlib
                 
 

首先需阅读 config.py 文件:

    state 配置训练还是测试阶段 (默认是 "TEST"(测试),使用已训练好的模型参数)
    test_root 新的测试数据集的路径 (默认是  "./test",子文件夹为各类图片)
    label_txt 类别名映射到数字 (子文件名映射到数字)
    
    data_root 训练集的路径(默认是 "./data")
    train_val_root 将训练集分为训练集和验证集
    data_augment_root 训练集扩展的放置路径
    
    classifier_save_path 分类器模型保存路径
    PCA_save_path pca pca模型保存路径
    codebook_save_path codebook保存路径
    
    img_size  所有的图片resize到 统一大小(默认 128X128)
    
    
    feature_meathod 特征提取的方法:hog,lbp,bag of words(默认bag_of_visual_word_category_partion_cluster)
    select_classifiers 分类器方法:KNN,LR,RF,DT,SVM,GBDT(默认SVM)
    
    # bag of words 参数
        weights_matrix 
        num_points_each_image
        num_category_cluster


运行方式:


   mode 1.测试新的数据集
        1.1 创建test文件夹,子文件夹为各类图片,文件夹名为类别名称(也可以修改config文件,任意命名都可,同时修改label.txt文件)
        1.2 python project-main (final).py  运行测试,给出结果
        
        
        
    mode 2.训练课设所给数据集,并给出验证集的结果
        1.1 修改config.py 文件的 state = "TRAIN"
        1.2 python project-main (final).py  进行训练以及验证,给出结果
