% Car License Plate Detection and Recognition in MATLAB

% Close all open figures and clear the workspace
close all;
clear all;
%% 0. Get Image Input from User
[fileName, filePath] = uigetfile(...
    {'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', 'Image Files (*.jpg, *.jpeg, *.png, *.bmp, *.tif)'; ...
     '*.*', 'All Files (*.*)'}, ...
    'Select an Image File');

% Check if the user cancelled the dialog
if fileName == 0
    disp('User cancelled the operation.');
    return; % Exit the script gracefully
end

% Construct the full path to the image
fullImagePath = fullfile(filePath, fileName);

%% 1. Read and Preprocess
try
    fprintf('Loading image: %s\n', fullImagePath);
    I = imread(fullImagePath); % Load the selected image
catch ME
    fprintf('Error reading image file: %s\n', fullImagePath);
    fprintf('Error message: %s\n', ME.message);
    return; % Exit if image reading fails
end
%% Step 1: Read and preprocess the image
grayI = rgb2gray(I); % Convert the image to grayscale
grayI = adapthisteq(grayI, 'NumTiles', [8 8], 'ClipLimit', 0.02); % Enhance contrast
filteredI = imgaussfilt(grayI, 2.6); % Apply Gaussian smoothing to the grayscale image

%% Step 2: Perform plate localization using morphology-based techniques
edgesV = edge(filteredI, 'Sobel', [], 'vertical'); % Detect vertical edges using the Sobel operator
bw1 = imdilate(edgesV, strel('rectangle', [9 1])); % Highlight vertical features like the edges of the plate
bw2 = imdilate(bw1, strel('rectangle', [1 15])); % Further dilate to connect components
bw = imclose(bw2, strel('rectangle', [5 17])); % Close gaps between connected components
bw = bwareaopen(bw, 500); % Remove small objects based on area threshold

%% Step 3: Extract and filter bounding boxes for potential license plates
stats = regionprops(bw, 'BoundingBox', 'Area'); % Get properties of connected components
areas = [stats.Area]; % Extract areas of the components
ratios = arrayfun(@(s) s.BoundingBox(3) / s.BoundingBox(4), stats)'; % Compute aspect ratios
mask = (areas > 2000) & (ratios > 2) & (ratios < 6); % Logical mask for filtering

% Ensure at least one candidate exists; select the first valid candidate
if ~any(mask)
    error('No license plate candidate found.');
end
plateBBox = stats(find(mask, 1)).BoundingBox; % Get the bounding box of the first valid candidate

%% Step 4: Perform OCR on the detected license plate region
plateRegion = imcrop(grayI, plateBBox); % Crop the region of interest (ROI) containing the plate
plateRegion = imgaussfilt(plateRegion, 0.8); % Apply slight Gaussian smoothing
plateRegion(plateRegion < 130) = 0; % Threshold low-intensity pixels
plateRegion(plateRegion > 165) = 255; % Threshold high-intensity pixels
plateRegion = localcontrast(plateRegion, 0.1, 0.9); % Enhance local contrast
plateRegion = imbinarize(plateRegion); % Binarize the image
plateRegion = ~plateRegion; % Invert the binary image
plateRegion = imclearborder(plateRegion); % Remove connected components at image borders
plateRegion = bwareaopen(plateRegion, 200); % Remove small objects

% Display the plate region
figure;
subplot(3, 1, 1); imshow(plateRegion); 
title('Plate Region');

% --- START: Template Matching OCR (Custom) ---

load NewTemplates % Load your templates created with template_creation.m

Iprops = regionprops(plateRegion, 'BoundingBox', 'Area', 'Image'); % Get letter components
count = numel(Iprops);
plateText = ''; % Initialize detected text

[h, w] = size(plateRegion); % Get size of plate region

for i = 1:count
    % Get the bounding box for the current character
    bbox = Iprops(i).BoundingBox; % [x, y, width, height]
    
    
    % Crop the character region using the padded bounding box
    charROI = imcrop(plateRegion, bbox);
    
    % Filter to avoid picking noise
    ow = size(charROI, 2); % Width of the character
    oh = size(charROI, 1); % Height of the character
    if ow < (h / 2) && oh > (h / 3.7)
        letter = Letter_detection(charROI); % Read letter using template matching
        plateText = [plateText letter]; % Append to final plate text
        % Display the character with padding
        subplot(3, count, count + i); imshow(charROI);
        title(['Character ' num2str(i) ': ' letter])
    end

end

% --- END: Template Matching OCR (Custom) ---

plateText = strtrim(plateText); % Clean up any leading/trailing spaces



%% Step 5: Annotate the original image with the detected plate
 output = insertObjectAnnotation(I, 'rectangle', plateBBox, plateText);
subplot(3, count, [count+i+1, count+i+2, count+i+3, count+i+4, count+i+5]);  imshow(output);
 title(['Detected Plate: ', plateText]);