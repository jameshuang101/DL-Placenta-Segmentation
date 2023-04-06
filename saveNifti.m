clear
close all

dataFolder = 'Y:\Advisory_Folder\Transition\01_Projects\01_Placenta_Project\Segmentation\01_Raw_Data\Axial\Mat_Files\Labels';
saveFolder = 'Y:\Advisory_Folder\Transition\01_Projects\01_Placenta_Project\Segmentation\02_Manual_Segmentation_Data\Axial';
%mkdir(saveFolder)
toSave = true;

imFiles = dir(fullfile(dataFolder,'*.mat'));
fprintf('\n')
for imNum = 1:numel(imFiles)
    fprintf('%s\n',imFiles(imNum).name)
    load(fullfile(dataFolder, imFiles(imNum).name));
    if exist('mrLabel','var')
        label = single(mrLabel); 
    elseif exist('plLabel','var')
        label = single(plLabel);  
    else
        label = single(utLabel);
    end
    clear mrLabel plLabel utLabel
    niftiwrite(label,fullfile(saveFolder, imFiles(imNum).name));
end