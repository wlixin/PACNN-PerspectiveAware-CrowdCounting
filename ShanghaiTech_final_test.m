function [MAE, MSE] = ShanghaiTech_final_test(phase, iter)
    lib = 'LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/lib64 ';
    caffe_path = '$CAFFE_ROOT$/build/tools/extract_features';
    lmdb2txt = '$CAFFE_ROOT$/build/tools/lmdb2txt';
    %% load image_list and frame_full
    % test set
    if strcmp(phase, 'test') == 1
        root_dir = '/ShanghaiTech/part_B/test_data/';
        image_list = dir([root_dir 'images/' '*.jpg']);
    end
    
    % train set
    if strcmp(phase, 'train') == 1
        root_dir = '/ShanghaiTech/partB/train_data/';
        image_list = dir([root_dir 'images/' '*.jpg']);
    end
    
    caffe_model = ['./pretrain_PACNN_ShanghaiTechPartB.caffemodel'];
    deploy_proto = ['./phase_2/deploy.prototxt'];
    %% test image prepare
    test_dir = ['/ShanghaiTech/part_B/test_data/test_image/'];
    if exist(test_dir)
        rmdir(test_dir, 's');
    end
    if exist('./estdmap.db')
        system(['sudo rm -r estdmap.db']);
        delete('./estdmap.txt')
    end
    if exist(['estdmap_' phase '_iter_' num2str(iter) '.txt'])
        system(['rm estdmap_' phase '_iter_' num2str(iter) '.txt']);
    end
    mkdir(test_dir);
    gpu_id = 3;
    type = 'resize';
    fid = fopen([test_dir 'list.txt'], 'w');
    
    nImg = length(image_list);
    gt_cc = zeros(nImg, 1);
    r = 1;
    for i = 1:nImg
        i
        test_image = imread([image_list(i).folder '/' image_list(i).name]);
        GT = load([root_dir '/ground-truth/GT_' strrep(image_list(i).name, '.jpg', '.mat')]);
        s = size(test_image);
        h = s(1) - mod(s(1), 16);
        w = s(2) - mod(s(2), 16);
        test_image = imresize(test_image, [h, w]);
        imwrite(test_image, [test_dir, image_list(i).name]);
        fprintf(fid, '%s\n', [test_dir, image_list(i).name]);
        gt_cc(i) = GT.image_info{1}.number;
    end
    fclose(fid);
    
    %% caffe test
    disp('caffe test');
    system(['sudo ' lib caffe_path ' ' caffe_model ' ' deploy_proto ' estdmap estdmap.db ' num2str(nImg) ' lmdb GPU ' num2str(gpu_id)]);
    system(['sudo ' lib lmdb2txt ' estdmap.db >> ' 'estdmap_' phase '_iter_' num2str(iter) '.txt']);
    cc = dlmread(['estdmap_' phase '_iter_' num2str(iter) '.txt']);
    cc = sum(cc(:,:), 2);
    cc_gt_cc = [cc gt_cc abs(cc-gt_cc)];
    MAE = mean(abs(cc-gt_cc))
    MSE = mean((cc-gt_cc).^2)^(0.5)
end

