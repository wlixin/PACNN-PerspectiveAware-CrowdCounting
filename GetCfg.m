function cfg = GetCfg()
    cfg.datasetName = 'ShanghaiTech_part_A';
    % Train images and ground truth folder
    cfg.trainImageFolder = 'data/ShanghaiTech/part_A/train_data/images';
    cfg.trainGroundTruthFolder = 'data/ShanghaiTech/part_A/train_data/ground-truth';
    cfg.trainGroundTruthPrefix = 'data/ShanghaiTech/part_A/train_data/ground-truth/GT_';
    % Train image gaussian distribution K and baseSigma
    cfg.k = 25;
    cfg.baseSigma = 1.5;
    % Downsample time means 2^() downsample
    cfg.downsampleTimes = 3;
    % How many patches sampled from each image
    cfg.patchNum = 4;
    % Ground Truth of each person position is ranged as [x, y], first index
    % is width, second index is height, [0, 0] is at top left of the image
    cfg.xFirst = true;
end