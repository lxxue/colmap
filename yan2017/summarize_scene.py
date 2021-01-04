
import numpy as np
from utils import IntersectList

# class SceneScore:
#     def __init__(self, alpha):
#         # weight between scene coverage score and scene conflict score
#         self.alpha = alpha
# 
#     def compute(self):
#         return 0


def SummarizeScene(tracks, visible_tracks, visible_keypoints, coverage_thres, alpha):
    assert(len(visible_tracks) == len(visible_keypoints))
    num_images = len(visible_tracks)
    img_included = np.zeros((num_images,), dtype=np.bool)

    delta = -10000000
    pre = 0
    coverage = 0

    covered_tracks = []
    confusing_tracks = []
    unique_tracks = []

    # while delta > 0.001:
    while coverage < coverage_thres:
        delta = -10000000
        chosen_image = -1
        chosen_intersected = None
        chosen_non_intersected = None

        for i in range(num_images):
            if img_included[i]:
                continue
            intersected, non_intersected = IntersectList(covered_tracks, visible_tracks[i])

            cur_delta = len(non_intersected) - alpha * len(intersected) 
            if (cur_delta > delta):
                delta = cur_delta
                chosen_image = i
                chosen_intersected = intersected
                chosen_non_intersected = non_intersected

        if chosen_image == -1:
            assert(np.all(img_included))
            print("all images included! coverage_thres too large!")
            break
        # assert(chosen_image > -1)
        img_included[chosen_image] = True
        covered_tracks.extend(non_intersected)

        confusing_it, confusing_non_it = IntersectList(confusing_tracks, intersected)
        confusing_tracks.extend(confusing_non_it)
        coverage = len(covered_tracks) / len(tracks) 

    _, unique_tracks = IntersectList(confusing_tracks, covered_tracks)
    return confusing_tracks, unique_tracks, img_included

                


