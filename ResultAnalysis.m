clear
close all

saveFolder = 'W:\04_Segmentation\Outputs\Axial';
origFolder = 'W:\04_Segmentation\Data';
mkdir(saveFolder)
dataFolder = 'Z:\Advisory_Folder\Placenta_Project\Outputs\10_14_22_Mats\Axial';

toSave = false;

Files = dir(fullfile(dataFolder, '*.mat'));
patientNums = [];
DSCs = [];
HDs = [];
VDs = [];
VDPs = [];

for imNum = 1:numel(Files)
    ptNum = Files(imNum).name(8:10); % 14:16 for sagittal data, 8:10 for axial, alter based on data file names
    if ismember(imNum, excludeNums)
        continue
    end
    
    fprintf('(%03d/%03d) P%s\n',imNum,numel(Files),ptNum)
    patientNums = [patientNums; ptNum];
    
    load(fullfile(dataFolder, Files(imNum).name));
    load(fullfile(origFolder, 'DL_data_axial\Mat_Files\Images', strcat(patientnum, '.mat')), 'pixDim'); 
    
    uLabel_true = imfill(((uLabel_true + pLabel_true) > 0),'holes');
    uLabel = (uLabel_pred + pLabel_pred) > 0;
    pLabel = logical(pLabel_pred);
    pLabel_true = imfill(logical(pLabel_true),'holes');
    clear pLabel_pred uLabel_pred
    
%     figure, isosurface(pLabel), title('Placenta label before post-processing')
%     figure, isosurface(uLabel), title('Uterus label before post-processing')
    
%     % Erode labels with 3x3x3 cube
    se = strel('cube',3);
    pLabel = imerode(pLabel, se);
    uLabel = imerode(uLabel, se);
     
    % Keep only largest 26-connected component of placenta label
    CC = bwconncomp(pLabel);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [~,idx] = max(numPixels);
    pLabel = false(size(pLabel));
    pLabel(CC.PixelIdxList{idx}) = true;
    
    % Keep only largest 26-connected component of uterus label
    CC = bwconncomp(uLabel);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [~,idx] = max(numPixels);
    uLabel = false(size(uLabel));
    uLabel(CC.PixelIdxList{idx}) = true;
     
    % Dilate labels with 3x3x3 cube
    pLabel = imdilate(pLabel,se);
    uLabel = imdilate(uLabel,se);
     
    % Fill holes
    pLabel = imfill(pLabel,'holes');
    uLabel = imfill(uLabel,'holes');
    
    % Smooth via blur with 3x3x3 mean filter and keep only values > 0.5
    pLabel = double(pLabel);
    pLabel = imboxfilt3(pLabel);
    pLabel = pLabel > 0.5;
    uLabel = double(uLabel);
    uLabel = imboxfilt3(uLabel);
    uLabel = uLabel > 0.5;
    
%     figure, isosurface(pLabel), title('Final processed placenta label')
%     figure, isosurface(uLabel), title('Final processed uterus label')
    
    % Calculate evaluation metrics (Dice score, Hausdorff distance, Volume
    % difference)
    % Need to scale to pixel dimensions per dimension
    [pHaus,~] = imhausdorff(pLabel,pLabel_true);
    [uHaus,~] = imhausdorff(uLabel,uLabel_true);
    
    DSCs = [DSCs; dice(pLabel,pLabel_true), dice(uLabel,uLabel_true)];
    HDs = [HDs; pHaus, uHaus];
    VDs = [VDs; (sum(pLabel(:))-sum(pLabel_true(:))) * prod(pixDim)/1000, (sum(uLabel(:))-sum(uLabel_true(:))) * prod(pixDim)/1000];
    VDPs = [VDPs; 100*(sum(pLabel(:))-sum(pLabel_true(:)))/sum(pLabel_true(:)), 100*(sum(uLabel(:))-sum(uLabel_true(:)))/sum(uLabel_true(:))];
    
    % Display overlaid ground truth and prediction
%     plotSurfaces(pLabel_true, pLabel)
%     plotSurfaces(uLabel_true, uLabel)
    
    if toSave
        save(fullfile('Z:\Advisory_Folder\Placenta_Project\Outputs\10_14_22_Mats_Processed\Sagittal', strcat(patientnum,'.mat')), 'mrImage', 'pLabel_true',...
        'pLabel', 'uLabel_true', 'uLabel');
    end
end

mean(DSCs)
std(DSCs)
mean(HDs)
std(HDs)
mean(VDs)
std(VDs)
mean(VDPs)
std(VDPs)

figure, boxplot(VDPs)
grid on
xticklabels({'Placenta', 'Uterine Cavity'})
ylabel('Volume Difference (%)')
ylim([-70,70])

figure, boxplot(DSCs)
grid on
xticklabels({'Placenta', 'Uterine Cavity'})
ylabel('Dice Similarity Coefficient')
ylim([0.25 1])