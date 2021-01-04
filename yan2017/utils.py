import numpy as np

MAX_IMAGE_ID = 2**31 - 1

def array_to_blob(array):
    return array.tostring()

def blob_to_array(blob, dtype, shape=(-1,)):
    if blob is None:
        return None
    return np.fromstring(blob, dtype=dtype).reshape(*shape)

def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return image_id1, image_id2

def binary_search_matches(keypoint_id, matches):
    # assuming the id is in the first column
    left = 0
    right = len(matches) - 1
    mid = (left + right) // 2
    while left <= right and keypoint_id != matches[mid, 0]:
        if keypoint_id < matches[mid, 0]:
            right = mid - 1
        if keypoint_id > matches[mid, 0]:
            left = mid + 1

        mid = (left + right) // 2
    
    if left <= right:
        assert(keypoint_id == matches[mid, 0])
        return matches[mid, 1]
    else:
        # there might be no keypoint asscoiated with `keypoint_id` in this match
        return -1
    
    # assert(mid >= 0 and mid < len(matches) and keypoint_id == matches[mid, 0])


def IntersectList(v1, v2, need_diff=True):
    """
        intersect: intersection of v1 and v2
        non_intersect: v2 - v1
    """
    intersect = []

    seen = set(v1)
    if need_diff:
        for element in v2:
            if element in seen:
                intersect.append(element)
            else:
                non_intersect.append(element)
        return intersect, non_intersect
    else:
        non_intersect = []
        for element in v2:
            if element in seen:
                intersect.append(element)
        return intersect