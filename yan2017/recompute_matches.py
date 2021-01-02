import numpy as np
from utils import IntersectList, image_ids_to_pair_id
from data_structure import MatchesList

def ConstructPathNetwork(num_images, matches_list, img_included, unique_tracks, visible_tracks, minimal_views, score_thres):
    path_graph = [None] * num_images
    # we can make this sparse matrix and save space later on
    is_geo_neighbors = np.zeros((num_images, num_images), dtype=np.bool)
    for i in range(num_images):
        path = []
        if img_included[i]:
            print("---iconic image {} connects:".format(i+1))
            continue
        
        print("---non-iconic image {} connects:".format(i+1))
        unique_i = IntersectList(unique_tracks, visible_tracks[i], need_diff=False)
        for j, matches in matches_list[i]:
            if img_included[j]:
                # skip iconic images
                # was commented in the original code
                # continue
                pass
            common_ij = IntersectList(visible_tracks[i], visible_tracks[j], need_diff=False)
            unique_j = IntersectList(unique_tracks, visible_tracks[j])
            unique_ij = IntersectList(unique_i, unique_j)
            if len(unique_ij) > minimal_views:
                score = len(unique_ij) / max(len(unique_i), len(unique_j))
                if score > score_thres:
                    path.append(j)
                    print("{} {} {} {}".format(j, len(unique_ij), max(len(unique_i), len(unique_j)), score))
                    is_geo_neighbors[i, j] = True
                    is_geo_neighbors[j, i] = True
        path_graph[i] = path
        print("")
    
    return path_graph

def RegenerateMatches(num_images, is_geo_neighbors, matches_list):
    # subject to simplications. For now we mimic the original code
    # Actually I feel like do this in my way is way more convenient and straightforaward
    new_matches_list = MatchesList(num_images) 
    for i in range(num_images):
        for j, matches in matches_list[i]:
            # j is stored as 1-based
            if is_geo_neighbors[i][j-1]:
                new_matches_list.append((j, matches))

    return new_matches_list


    # for i in range(num_images):
    #     for j in range(num_images):
    #         existed = False
    #         for k in path_graph[i]:
    #             if k == j:
    #                 existed = True
    #                 break
    #         if not existed:
    #             for k in path_graph[j]:
    #                 if k == i:
    #                     existed = True
    #                 break

            if existed:
                pair_id = image_ids_to_pair_id(i, j)







