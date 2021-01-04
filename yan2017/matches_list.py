import numpy as np
import sqlite3

from utils import image_ids_to_pair_id, pair_id_to_image_ids, blob_to_array, array_to_blob
from database import COLMAPDatabase


class MatchesList:
    def __init__(self, num_images, database=None):
        
        # `[[]] * N` create a list containing the same list object N times!!!
        # self.matches_list = [[]] * num_images
        self.matches_list = [[] for _ in range(num_images)]
        if database is not None:
            matches_results = database.execute("select * FROM matches")
            for matches_result in matches_results:
                pair_id, rows, cols, matches = matches_result
                image_id1, image_id2 = pair_id_to_image_ids(pair_id) 
                # if image_id1 == 1:
                #     print(image_id1, image_id2)
                matches = blob_to_array(matches, np.uint32, (rows, cols))
                # print(image_id1, image_id2, rows)
                if rows > 0:
                    # print(image_id1, image_id2)
                    matches_1 = self._sort_matches(matches) 
                    matches_2 = matches[:, [1,0]]
                    # image_id is 1-based
                    # we keep this convention and only subtract it by 1 when we use it as index
                    # we make this adjacency list symmetric for easier construction of tracks
                    # at the expense of memory and computation(sorting)
                    self.matches_list[image_id1-1].append((image_id2, matches_1)) 
                    self.matches_list[image_id2-1].append((image_id1, matches_2)) 
                    # self.matches_list[image_id1-1].append(image_id2) 
                    # self.matches_list[image_id2-1].append(image_id1) 
        # print(self.matches_list[0])
        # print(self.matches_list[1])
        # print(self.matches_list[2])
        # print(self.matches_list[3])
        # print([nbr[0] for nbr in self.matches_list[2]])
        # print(self.matches_list[1][:3])

    def __getitem__(self, idx):
        return self.matches_list[idx]
    
    def _sort_matches(self, matches):
        # matches (n, 2) array.
        # sort according to first col
        order = np.argsort(matches[:, 0])
        return matches[order, :]

# class Tracks:
#     def __init__(self):
#         self.track_list = []
#         pass

def test_matchlist():
    db = COLMAPDatabase.connect("data/yan2017/colmap_street/match.db")

    image_results = db.execute("SELECT * FROM images")
    num_images = len(list(image_results))
    # print(num_images)
    matches_list = MatchesList(num_images, database=db)


if __name__ == "__main__":
    test_matchlist()
