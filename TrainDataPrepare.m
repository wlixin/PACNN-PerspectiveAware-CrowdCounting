function TrainDataPrepare()
    % Get crowd counting config
    cfg = GetCfg()

    % Images names
    trainImageFolder = cfg.trainImageFolder;
    trainFilesStruct = dir(fullfile(trainImageFolder, '*.jpg'));
    trainFilesName = {trainFilesStruct.name};
    
    %% Generate Patch
    imagePatchDir = fullfile('.', 'data', 'samplePatch', [cfg.datasetName, '_', num2str(cfg.k) '_' num2str(cfg.baseSigma)], 'image_patch')
    densityMapPatchDir = fullfile('.', 'data', 'samplePatch', [cfg.datasetName, '_', num2str(cfg.k) '_' num2str(cfg.baseSigma)], 'densitymap_patch')
    caffeDensityMapPatchDir = fullfile('.', 'data', 'samplePatch', [cfg.datasetName, '_', num2str(cfg.k) '_' num2str(cfg.baseSigma)], 'densitymap_caffe')
    mkdir(imagePatchDir);
    mkdir(densityMapPatchDir);
    mkdir(caffeDensityMapPatchDir);
    
    for i = 1:length(trainFilesName)
        fileName = trainFilesName{i};
        [~, fileName, ext] = fileparts(fileName);
        fprintf('%d / %d : %s\n', i, length(trainFilesName), fileName);
        imageGroundTruth = load([cfg.trainGroundTruthPrefix, fileName, '.mat']);
        image = double(imread(fullfile(cfg.trainImageFolder, [fileName '.jpg']))) / 255.0;
        [imageH, imageW, ~] = size(image);
        imageLoc = imageGroundTruth.image_info{1, 1}.location;
        % Density map
        imageDownsampleH = ceil(imageH / 2^(cfg.downsampleTimes));
        imageDownsampleW = ceil(imageW / 2^(cfg.downsampleTimes));
        if imageGroundTruth.image_info{1, 1}.number == 0
            densityMapDownsample = zeros([imageDownsampleH, imageDownsampleW]);
        else
            imageDownsampleLoc = bsxfun(@rdivide, imageLoc, 2^cfg.downsampleTimes);
            densityMapDownsample = GetDensityMap(imageDownsampleLoc, cfg, [imageDownsampleH, imageDownsampleW]);
        end
        %% Sample patch and density map
        % Anchor mod 32 == 0
        anchorX = round(imageW / 2);
        anchorY = round(imageH / 2);
        anchorX = anchorX - mod(anchorX, 32);
        anchorY = anchorY - mod(anchorY, 32);
        % Generate start index and filter them
        xIdx = randperm(imageW - ceil(imageW / 2));
        yIdx = randperm(imageH - ceil(imageH / 2));
        validIdx = 0;
        for j = min(length(xIdx), length(yIdx)): -1: 1
            xv = [xIdx(j), xIdx(j), xIdx(j) + anchorX -1, xIdx(j) + anchorX -1];
            yv = [yIdx(j), yIdx(j) + anchorY -1, yIdx(j) + anchorY -1, yIdx(j)];
            locInAnchor = 0;
            for k = 1: length(imageLoc)
                if inpolygon(imageLoc(k, 1), imageLoc(k, 2), xv, yv)
                    locInAnchor = locInAnchor + 1;
                end
            end
            if locInAnchor == 0
                xIdx(j) = [];
                yIdx(j) = [];
            else
                validIdx = validIdx + 1;
            end
            if validIdx == cfg.patchNum
                break;
            end
        end
        xIdx(1:length(xIdx)-cfg.patchNum) = [];
        yIdx(1:length(yIdx)-cfg.patchNum) = [];
        %% Save patch and its densitimap
        for j = 1:cfg.patchNum
            xStart = xIdx(j);
            yStart = yIdx(j);
            imagePatch = image(yStart:yStart+anchorY-1, xStart:xStart+anchorX-1, :);
            [patchH, patchW, ~] = size(imagePatch);
            tmpH = patchH; tmpW = patchW;
            for k = 1:cfg.downsampleTimes
                tmpH = ceil(tmpH / 2);
                tmpW = ceil(tmpW / 2);
            end
            patchDownsampleH = tmpH;
            patchDownsampleW = tmpW;
            xDensityMapStart = floor(xStart / (2^cfg.downsampleTimes));
            yDensityMapStart = floor(yStart / (2^cfg.downsampleTimes));
            patchDensityMapDownsample = densityMapDownsample(max(1,yDensityMapStart):max(1,yDensityMapStart)+patchDownsampleH-1, max(1,xDensityMapStart):max(1,xDensityMapStart)+patchDownsampleW-1);
%             subplot(1, 3, 1);
%             imshow(image);
%             subplot(1, 3, 2);
%             imshow(imagePatch)
%             subplot(1, 3, 3);
%             imagesc(patchDensityMapDownsample)
%             pause(1)
            imwrite(imagePatch, fullfile(imagePatchDir, [fileName '_' num2str(j) '_' num2str(patchH) '_' num2str(patchW) '.png']));
            imwrite(patchDensityMapDownsample, fullfile(densityMapPatchDir, [fileName '_' num2str(j) '_' num2str(patchH) '_' num2str(patchW) '.png']));
            patchDensityMapDownsample = patchDensityMapDownsample';
            fid = fopen(fullfile(caffeDensityMapPatchDir, [fileName '_' num2str(j) '_' num2str(patchH) '_' num2str(patchW) '.bin']), 'wb');
            fwrite(fid, patchDensityMapDownsample, 'single');
            fclose(fid);
        end
    end
end

function [DensityMapDownsample] = GetDensityMap(imageDownsampleLoc, cfg, imageDownsampleSize)
    imageDownsampleH = imageDownsampleSize(1);
    imageDownsampleW = imageDownsampleSize(2);
    padding = floor((cfg.k - 1) / 2);
    gaussianKernel = fspecial('gaussian', cfg.k, cfg.baseSigma);
    paddingDensityMapDownsample = zeros(imageDownsampleH + cfg.k - 1, imageDownsampleW + cfg.k - 1);
    for i = 1:length(imageDownsampleLoc)
        locX = round(imageDownsampleLoc(i, 1));
        locY = round(imageDownsampleLoc(i, 2));
        locX(locX < 1) = 1;
        locY(locY < 1) = 1;
        locX(locX > imageDownsampleW) = imageDownsampleW;
        locY(locY > imageDownsampleH) = imageDownsampleH;
        paddingDensityMapDownsample(locY : locY + cfg.k -1, locX : locX + cfg.k - 1) = paddingDensityMapDownsample(locY : locY + cfg.k -1, locX : locX + cfg.k - 1) + gaussianKernel;
    end
    DensityMapDownsample = paddingDensityMapDownsample(1 + padding : imageDownsampleH + padding, 1 + padding : imageDownsampleW + padding);
end