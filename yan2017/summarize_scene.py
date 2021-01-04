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

    # for i in range(num_images):
    #     assert(len(visible_tracks[i]) == len(visible_keypoints[i]))
    #     print("image {} sees {} tracks".format(i+1, len(visible_tracks[i])))

    # prev_coverage = 0
    cur_coverage = 0

    covered_tracks = []
    confusing_tracks = []
    unique_tracks = []

    # greedily add an image to the iconic set using coverage threshold as stopping criterion
    # TODO: try use changing in coverage as stopping criterion, e.g. while delta >= ...
    while cur_coverage < coverage_thres:
        best_delta = -1e7
        chosen_image = -1
        chosen_intersected = None
        chosen_non_intersected = None

        for i in range(num_images):
            if img_included[i]:
                continue
            intersected, non_intersected = IntersectList(covered_tracks, visible_tracks[i])

            cur_delta = len(non_intersected) - alpha * len(intersected) 
            if (cur_delta > best_delta):
                best_delta = cur_delta
                chosen_image = i
                chosen_intersected = intersected
                chosen_non_intersected = non_intersected

        # if chosen_image == -1:
        #     assert(np.all(img_included))
        #     print("all images included! coverage_thres too large!")
        #     break
        assert(chosen_image > -1)
        img_included[chosen_image] = True
        
        covered_tracks.extend(chosen_non_intersected)

        confusing_it, confusing_non_it = IntersectList(confusing_tracks, chosen_intersected)
        confusing_tracks.extend(confusing_non_it)

        cur_coverage = len(covered_tracks) / len(tracks) 
        # print(cur_coverage)

    _, unique_tracks = IntersectList(confusing_tracks, covered_tracks)
    # print(confusing_tracks)
    # print(sorted(unique_tracks))
    print("{} images included in the iconic set".format(np.sum(img_included)))
    print("there are {} confusing tracks and {} unique tracks".format(len(confusing_tracks), len(unique_tracks)))
    print("-----------SummarizeScene Done-----------")
    return unique_tracks, img_included

                


