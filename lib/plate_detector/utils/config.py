im_w_1080_2 = 480 * 2
im_h_1080_2 = 270 * 2
f_m_1080_2 = [[34, 60], [17, 30], [9, 15], [5, 8], [3, 4], [2, 2]]

min_size_list = [0.04, 0.14, 0.25, 0.5, 0.65, 0.80]
max_size_list = min_size_list[1:] + [min_size_list[-1] + 0.15]
COCO_mobile_1080_2 = {
    'feature_maps': f_m_1080_2,
    'min_dim': -300,
    'img_w': im_w_1080_2,
    'img_h': im_h_1080_2,
    'steps_x': [im_w_1080_2 / a[1] for a in f_m_1080_2],         # [16, 32, 64, 120, 240, 480],
    'steps_y': [im_h_1080_2 / a[0] for a in f_m_1080_2],         # [15.88, 31.76, 60, 108, 180, 270],
    'min_sizes_x': [i * im_w_1080_2 for i in min_size_list],     # [45, 90, 135, 180, 225, 864],
    'min_sizes_y': [i * im_h_1080_2 for i in min_size_list],
    'max_sizes_x': [i * im_w_1080_2 for i in max_size_list],     # [45, 90, 135, 180, 225, 864],
    'max_sizes_y': [i * im_h_1080_2 for i in max_size_list],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'BGR_MEANS': [103.94, 116.78, 123.68]
}
