def get_downsampling_scheme(img_size, min_size=4):
    img_size = tuple(img_size)
    sizes = [img_size]
    current_size = img_size
    while current_size[0] % 2 == 0 and current_size[1] % 2 == 0 and\
        current_size[0] // 2 >= min_size and current_size[1] >= min_size:
        print(f'DOWNSAMPLE {current_size[0]}x{current_size[1]} -> {current_size[0] // 2}x{current_size[1] // 2}')
        current_size = (current_size[0] // 2, current_size[1] // 2)
        sizes.append(current_size)
    #print(f'FLATTEN {current_size[0]}x{current_size[1]} -> {current_size[0]*current_size[1]}')
    return sizes