import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from image_processing.processing import denormalize_img
from scipy import stats
from scipy.signal import convolve2d

def get_number_of_colors(img):
    n_colors = len(np.unique(img.reshape(-1, img.shape[-1]), axis=0))
    return n_colors

def get_most_frequent_color(data, return_count=False):
    '''
    Find the most frequent color in the arrays given. The color dimension is assumed to be the last one.

    Parameters
    ----------
    data : numpy.array
        Array to find the most common color in.
    return_count : Boolean, optional
        Boolean that decides whether to return times the most common color occured in the data. The default is False.

    Returns
    -------
    tuple
        The most common color (tuple of length 3).

    '''
    colors = {}
    greatest_frequency = 0
    color_picked = (255, 255, 255)
    
    arr = np.array([data[..., i].ravel() for i in range(data.shape[-1])])

    arr = np.reshape(arr, (-1, data.shape[-1]))
    size = arr.shape[0]
    
    for i in range(size):
        color = tuple(arr[i])
        if color in colors:
            colors[color] += 1
            if colors[color] > greatest_frequency:
                greatest_frequency = colors[color]
                color_picked = color
        else:
            colors[color] = 1

    if not return_count:
        return color_picked
    else:
        return greatest_frequency, color_picked

def get_heatmap(images, method='mean', quantile=0.75):
    '''
    Get heatmap of the given images. 

    Parameters
    ----------
    images : numpy.array
        Images to use for creating heatmap.
    method : one of ['mean', 'median', 'quantile'], optional
        Method to calculate heatmap with. The default is 'mean'.
    quantile : float, between [0.0 and 1.0], optional
        Only used when method chosen is quantile. The default is 0.75. Method quantile with quantile=0.5 is the same as method='median'.

    Returns
    -------
    numpy.array
        The heatmap array of the images given.

    '''
    if method == 'mean':
        return np.mean(images, axis=0)
    if method == 'median':
        return np.median(images, axis=0)
    if method == 'quntile':
        return np.quantile(images, quantile, axis=0)
    
def estimate_character_mask(images, most_frequent_color=None, conv_correction_threshold=8):
    '''
    Estimate masks for the provided images. Estimated masks should True where characters are and False where background is. On the original images it achieved 99.65% accuracy
    
    Parameters
    ----------
    images : numpy.array
        Images to estimate mask on.
    most_frequent_color : typle, optional
        The color that is assumed to be background. The default is None. If not provided it will be estimated in the function (costly operation)
    conv_correction_threshold : int, optional
        The threshold used in correcting the estimated mask so it doesn't have any obvious holes in it. The default is 8 - the best value for estimating mask on originals.

    Returns
    -------
    numpy.array (with dtype boolean)
        Boolean mask estimated for the provided images.

    '''
    if images.shape[-1] == 4:
        # the 4th channel is usally the alpha channel
        if images.max() == 1.0:
            return images[..., -1] == 1.0
        else:
            return images[..., -1] == 255

    # Estimate mask from pixel values
    if not most_frequent_color:
        most_frequent_color = get_most_frequent_color(images)
    mask = (images[:, :, :, 0] != most_frequent_color[0]) |\
           (images[:, :, :, 1] != most_frequent_color[1]) |\
           (images[:, :, :, 2] != most_frequent_color[2])
           
    if conv_correction_threshold > 0:
        lst = [np.pad(mask[i], 1, 'constant', constant_values=False) for i in range(mask.shape[0])]
        mask_padded = np.array(lst)
        filter = np.ones((3, 3))
        
        # Use convolution operations to "repair holes" in the mask
        for i in range(mask_padded.shape[0]):
            conv_out = convolve2d(mask_padded[i], filter, mode='valid')
            m = conv_out >= conv_correction_threshold # if more than threshold neighbours were True then this one also should be True
            conv_out = conv_out.astype(bool)
            conv_out[m] = True
            conv_out[~m] = False
            mask[i] = mask[i] | conv_out # the logical sum of the initially estimated mask and the correction got from convolution
    return mask
    
def get_left_offsets(masks):
    offsets = []
    for arr in masks:
        offset = -1 # -1 is used where there is no character at all
        for i in range(arr.shape[1]):
            if arr[:, i].any():
                offset = i
                break
        offsets.append(offset)
    return offsets

def get_right_offsets(masks):
    offsets = []
    for arr in masks:
        offset = -1 # -1 is used where there is no character at all
        for i in range(arr.shape[1]):
            if arr[:, arr.shape[1] - 1 - i].any():
                offset = i
                break
        offsets.append(offset)
    return offsets

def get_top_offsets(masks):
    offsets = []
    for arr in masks:
        offset = -1 # -1 is used where there is no character at all
        for i in range(arr.shape[0]):
            if arr[i, :].any():
                offset = i
                break
        offsets.append(offset)
    return offsets

def get_bottom_offsets(masks):
    offsets = []
    for arr in masks:
        offset = -1 # -1 is used where there is no character at all
        for i in range(arr.shape[0]):
            if arr[arr.shape[0] - 1 - i, :].any():
                offset = i
                break
        offsets.append(offset)
    return offsets

def calculate_colors_std(images):
    if images.shape[-1] == 4:
        images = images[..., :-1]
    return np.std(images, axis=(1, 2))

def calculate_pixels_number(mask):
    return mask.sum(axis=(1, 2))

def calculate_colors_histograms(data, mask=None, bins=10):
    data = data.astype(np.float32) #makes a copy
    if data.shape[-1] == 4:
        data = data[..., :-1]
    if mask is not None:
        data[~mask] = -1
    min_val = data.min()
    max_val = data.max()
    step = (max_val - min_val) / bins
    l = []
    for channel in range(data.shape[-1]):
        channel = data[..., channel]
        bins_stats = []
        for i in range(bins):
            if i != bins - 1:
                bin_count = ((channel >= i * step) & (channel < (i + 1) * step)).sum(axis=(1, 2))
            else:
                bin_count = ((channel >= i * step) & (channel <= (i + 1) * step)).sum(axis=(1, 2))
            bins_stats.append(bin_count)
        l.append(bins_stats)
    return l

def calculate_brightness(data, mask=None):
    if data.shape[-1] == 4: 
        if mask is None:
            mask = data[..., -1]
        data = data[..., :-1]
        
    if len(data.shape) == 4:
        brightness_list = []
        for i in range(data.shape[0]):
            img = data[i]
            if mask is not None:
                character_pixels = img[mask[i]]
                if character_pixels.size:
                    brightness = np.mean(character_pixels)
                else:
                    brightness = -1.0
                brightness_list.append(brightness)
                
    else:
        brightness_list = [np.mean(data[mask])]
    return brightness_list

def vertical_direction_change_score(data, mask=None):
    data = data.copy()
    if data.shape[-1] == 4:
        data = data[..., :-1]
    if mask is not None:
        data[~mask] = 0
    
    if len(data.shape) == 4:
        ax = (1, 2, 3)
        score = np.abs(data[:, :-1, ...] - data[:, 1:, ...]).sum(axis=ax)
    else:
        ax = (0, 1, 2)
        score = np.abs(data[:-1, ...] - data[1:, ...]).sum(axis=ax)

    if mask is not None:
        score = score / (mask.sum(axis=ax[:-1]) + data.shape[ax[1]] * 2) / data.shape[-1]
    else:
        height = data.shape[1]
        width = data.shape[2]
        n_channels = data.shape[-1]
        score = score / (height - 1) / (width - 1) / n_channels
    return score

def horizontal_direction_change_score(data, mask=None):
    data = data.copy()
    if data.shape[-1] == 4:
        data = data[..., :-1]
    if mask is not None:
        data[~mask] = 0
    
    if len(data.shape) == 4:
        ax = (1, 2, 3)
        score = np.abs(data[..., :-1, :] - data[..., 1:, :]).sum(axis=ax)
    else:
        ax = (0, 1, 2)
        score = np.abs(data[:, :-1, :] - data[:, 1:, :]).sum(axis=ax)
    
    if mask is not None:
        score = score / (mask.sum(axis=ax[:-1]) + data.shape[ax[0]] * 2) / data.shape[-1]
    else:
        height = data.shape[1]
        width = data.shape[2]
        n_channels = data.shape[-1]
        score = score / (height - 1) / (width - 1) / n_channels
    return score

if __name__ == '__main__':
    images = np.load('dataset.npy')
    images = denormalize_img(images)
    im_size = (images.shape[1] * images.shape[2])
    df = pd.DataFrame(index=range(len(images)))
    colors = ['Red', 'Green', 'Blue']
    max_value = images.max()
    
    masks = estimate_character_mask(images)
    assert(all(masks.sum(axis=(1, 2)) >= 0) and all(masks.sum(axis=(1, 2)) <= im_size))
    
    df['LeftOffset'] = get_left_offsets(masks)
    assert(all((df['LeftOffset'] >= -1) | (df['LeftOffset'] < masks.shape[2]))) 
    df['RightOffset'] = get_right_offsets(masks)
    assert(all((df['RightOffset'] >= -1) | (df['RightOffset'] < masks.shape[2])))
    df['TopOffset'] = get_top_offsets(masks)
    assert(all((df['TopOffset'] >= -1) | (df['TopOffset'] < masks.shape[1])))
    df['BottomOffset'] = get_bottom_offsets(masks)
    assert(all((df['BottomOffset'] >= -1) | (df['BottomOffset'] < masks.shape[1])))
    assert(((df['LeftOffset'] == -1) & (df['RightOffset'] == -1) &\
           (df['TopOffset'] == -1) & (df['BottomOffset'] == -1)).sum() == \
           ((df['LeftOffset'] == -1) | (df['RightOffset'] == -1) |\
           (df['TopOffset'] == -1) | (df['BottomOffset'] == -1)).sum())
    
    df['Width'] = masks.shape[2] - df['RightOffset'] - df['LeftOffset']
    df.loc[df['RightOffset'] == -1, 'Width'] = 0
    assert(all((df['Width'] <= images.shape[2]) & (df['Width'] >= 0)))
    df['Height'] = masks.shape[1] - df['TopOffset'] - df['BottomOffset']
    df.loc[df['RightOffset'] == -1, 'Height'] = 0
    assert(all((df['Height'] <= images.shape[1]) & (df['Height'] >= 0)))
    
    df['BoundingBoxArea'] = df['Width'] * df['Height']
    assert(all(df['BoundingBoxArea'] <= images.shape[1] * images.shape[2]))
    df['CharacterSize'] = calculate_pixels_number(masks)
    assert(all(df['CharacterSize'] <= df['BoundingBoxArea']))
    df['SizeToImageRatio'] = df['CharacterSize'] / im_size
    assert(df['SizeToImageRatio'].max() <= 1.0)
    df['SizeToBoundingBoxRatio'] = df['CharacterSize'] / df['BoundingBoxArea']
    df['SizeToBoundingBoxRatio'].fillna(0.0, inplace=True)
    assert(df['SizeToBoundingBoxRatio'].max() <= 1.0)
    df['BoundingBoxNonCharacterPixels'] = df['BoundingBoxArea'] - df['CharacterSize']
    assert(df['BoundingBoxNonCharacterPixels'].min() >= 0)
    assert(all((df['BoundingBoxNonCharacterPixels'] < df['BoundingBoxArea']) | (df['BoundingBoxArea'] == 0)))
    
    variety = calculate_colors_std(images)
    assert(np.all((variety >= 0) & (variety <= max_value)))
    for i in range(variety.shape[1]):
        df['Variety' + colors[i]] = variety[:, i]
    del variety 
    
    df['VerticalDirectionChangeScoreMasked'] = vertical_direction_change_score(images, masks)
    assert(all(df['VerticalDirectionChangeScoreMasked'] < images[..., :-1].max(axis=(1, 2, 3))))
    #df['VerticalDirectionChangeScore']  = vertical_direction_change_score(images)
    df['HorizontalDirectionChangeScoreMasked'] = horizontal_direction_change_score(images, masks)
    #df['HorizontalDirectionChangeScore']  = horizontal_direction_change_score(images)
    assert(all(df['HorizontalDirectionChangeScoreMasked'] < images[..., :-1].max(axis=(1, 2, 3))))
    
    histograms = np.array(calculate_colors_histograms(images, mask=masks))
    #histograms2 = np.array(calculate_colors_histograms(images))
    for i in range(histograms.shape[0]):
        for j in range(histograms.shape[1]):
            df['HistogramMasked' + colors[i] + f'Bin{j+1}'] = histograms[i, j]
            assert(all((histograms[i, j] >= 0) & (histograms[i, j] <= images.shape[1] * images.shape[2])))
            #df['Histogram' + colors[i] + f'Bin{j+1}'] = histograms2[i, j]
    #del histograms
    #del histograms2        
    
    df['BrightnessMasked'] = calculate_brightness(images, masks)
    assert(all((df['BrightnessMasked'] >= -1) & (df['BrightnessMasked'] <= max_value)))
    #brightness2 = calculate_brightness(images)
    