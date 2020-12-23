import numpy as np
import sqlite3
import queue

from utils import binary_search_matches

def ComputeTracks(images_list, keypoints_list, matches_list, track_degree):
    num_images = len(images_list)
    # this two flags are used for avoid inconsistent tracks
    # in which there are more than one feature in some views
    # not sure if this is the better way 
    # TODO: in this way the track construction is not invariant to  permutation
    # try filtering out inconsistent tracks as done in the Bundler's commented code?
    image_marked = np.zeros((num_images,), dtype=np.bool)
    touched = []
    
    tracks = []

    max_num_keypoints = max([len(keypoints) for keypoints in keypoints_list])
    keypoints_visited = np.zeros((num_images, max_num_keypoints), dtype=np.bool)
    for i, image, keypoints in enumerate(zip(images_list, keypoints_list)):
        for j, keypoint in enumerate(keypoints):
            features = []
            features_queue = queue.Queue()
            # BFS on this keypoint
            if keypoints_visited[i][j]:
                continue
            keypoints_visited[i][j] = True

            # Reset flags
            for touched_idx in touched:
                img_marked[touched_idx] = False
            touched = []

            # image_id is 1-based, keypoint_id is 0_based
            features.append((i+1, j))
            features_queue.put((i+1, j))

            img_marked[i] = True
            touched.append(i)

            num_rounds = 0
            while not features_queue.empty():
                num_rounds += 1

                img_id1, keypoint_id1 = features_queue.get()

                for img_id2, matches in matches_list[img_id1]:
                    # skip already visited images to avoid inconsistency
                    if img_marked[img_id2]:
                        continue
                    keypoint_id2 = binary_search_matches(keypoint_id1, matches)
                    if keypoint_id2 < 0:
                        # matches not found
                        continue
                    assert(keypoint_id2 < len(keypoints_list[img_id2]))
                    if keypoints_visited[img_id2, keypoint_id2]:
                        continue
                    img_marked[img_id2] = True
                    keypoints_visited[img_id2, keypoint_id2] = True
                    touched.append(img_id2)
                    features.append((img_id2, keypoint_id2))
                    features_queue.put((img_id2, keypoint_id2))
            
            # found all features corresponding to one 3D point in different views
            # in a consistent way (by construction)
            if len(features) >= track_degree:
                # show up in enough number of images
                tracks.push_back(features)
    
    # All tracks have been computed
    # we check which tracks and keypoints are visible in the image
    # track_idx is 0-based
    visible_tracks = [[]] * num_images
    visible_keypoints = [[]] * num_images
    for track_idx, track in enumerate(tracks):
        for img_id, keypoint_id in track:
            visible_tracks[img_id].append(track_idx)
            visible_keypoints[img_id].append(keypoint_id)

    return tracks, visible_tracks, visible_keypoints










