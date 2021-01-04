import numpy as np
import sqlite3
import queue

from utils import binary_search_matches
from matches_list import MatchesList
from database import COLMAPDatabase

def ComputeTracks(num_images, num_keypoints_list, matches_list, track_degree):
    """
        num_images: number of images in the database
        num_keypoints_list: list of #keypoints in each image (int)
        matches_list: matches_list[i] is a list of tuple (image_id, matches), see class MatchesList
        track_degree: only consider tracks which appear in more than `track_degree` images
    """
    # this two flags are used for avoid inconsistent tracks
    # in which there are more than one feature in some views
    # in this way the track construction is consistent but not invariant to  permutation
    # TODO: try filtering out inconsistent tracks as done in the Bundler's commented code?
    img_marked = np.zeros((num_images,), dtype=np.bool)
    touched = []
    
    # a track is simply a list of (image_id, keypoint_id) which marks the appearance of the 3D point in certain images
    tracks = []

    max_num_keypoints = max(num_keypoints_list)
    keypoints_visited = np.zeros((num_images, max_num_keypoints), dtype=np.bool)

    # image is 1-based indexed, keypoints is 0-based indexed
    # i here is 0-based so it can be used directly as index without being subtracted by 1
    for i, num_keypoints in enumerate(num_keypoints_list):
        for j in range(num_keypoints):
            # print("")
            features = []
            features_queue = queue.Queue()
            # print("track {} consists of: ".format(len(tracks)), end="")
            if keypoints_visited[i][j]:
                continue

            # Reset flags
            for touched_idx in touched:
                img_marked[touched_idx-1] = False
            touched = []

            # BFS on this keypoint
            keypoints_visited[i][j] = True

            # image_id is 1-based, keypoint_id is 0-based
            features.append((i+1, j))
            features_queue.put((i+1, j))
            # print("({}, {}) ".format(i+1, j), end="")

            img_marked[i] = True
            touched.append(i)

            while not features_queue.empty():
                img_id1, keypoint_id1 = features_queue.get()

                for img_id2, matches in matches_list[img_id1-1]:
                    # skip already visited images to avoid inconsistency
                    if img_marked[img_id2-1]:
                        continue
                    keypoint_id2 = binary_search_matches(keypoint_id1, matches)
                    if keypoint_id2 < 0:
                        # match not found
                        # TODO: is this missing correspondence? maybe we can do something about it?
                        continue
                    assert(keypoint_id2 < num_keypoints_list[img_id2-1])
                    if keypoints_visited[img_id2-1, keypoint_id2]:
                        # already picked by other tracks
                        continue
                    img_marked[img_id2-1] = True
                    keypoints_visited[img_id2-1, keypoint_id2] = True
                    touched.append(img_id2)
                    features.append((img_id2, keypoint_id2))
                    features_queue.put((img_id2, keypoint_id2))
                    # print("({}, {}) ".format(img_id2, keypoint_id2), end="")
            
            # found all features corresponding to one 3D point in different views
            # in a consistent way (by construction)
            if len(features) >= track_degree:
                # show up in enough number of images
                # print("Point with {} projections found".format(len(features)))
                tracks.append(features)
                # print("track {} consists of: ".format(len(tracks)), features)
    
    # All tracks have been computed
    # we check which tracks and keypoints are visible in each image
    # track_idx is 0-based
    # `[[]] * N` create a list containing the same list object N times!!!
    # visible_tracks = [[]] * num_images
    # visible_keypoints = [[]] * num_images
    visible_tracks = [[] for _ in range(num_images)]
    visible_keypoints = [[] for _ in range(num_images)]
    for track_idx, track in enumerate(tracks):
        for img_id, keypoint_id in track:
            visible_tracks[img_id-1].append(track_idx)
            visible_keypoints[img_id-1].append(keypoint_id)

    print("[----------ComputeTracks Done----------")
    return tracks, visible_tracks, visible_keypoints


if __name__ == "__main__":
    pass
    # db = COLMAPDatabase.conenct("data/yan2017/colmap_street/match.db")
    # image_results = db.execute("SELECT * FROM images")
    # num_images = len(list(image_results))
    # matches_list = MatchesList(db, num_images)
    # tracks, visible_tracks, visible_keypoints = ComputeTracks()







