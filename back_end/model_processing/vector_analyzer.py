
import numpy as np
from scipy.spatial.distance import cdist
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


#TODO: does this create duplicate vectors?
#TODO: does this take in inference results or activation vectors?

class Vector_Analyzer:
    def __init__(self, inference_results: list[dict]): #TODO: inference results are stored in the session, dont need duplicate storage here
        #TODO: do all these need to be computed and stored on initialization, or can they be created as needed?
        #TODO: find way to walk through the code and see how much time/memory every step uses
        self.id_map: np.ndarray = self.get_id_mapping(inference_results)
        self.activation_matrix: np.ndarray = self.get_activation_matrix(inference_results)
        self.distance_matrix: np.ndarray = None #lazy loaded in find_all_similar_pairs
        """
        Shape of vectors
        {
                'id':          record['id'],
                'input':       record['input'].tolist(),
                'label':       record.get('label'),
                'predicted':   int(np.argmax(layer_activations[-1][0])),
                'correct':     int(np.argmax(layer_activations[-1][0])) == record.get('label'),
                'activations': activations,
        }
        """
    

    def get_id_mapping(self, inf_results: list[dict]) -> np.ndarray:
        return [r['id'] for r in inf_results]

    # Stack all activation vectors into a 2D matrix of shape (num_records, total_activation_size)
    def get_activation_matrix(self, inf_results: list[dict]) -> np.ndarray:
        return np.vstack([r['activations'] for r in inf_results])

    # Compute pairwise cosine distances in a single vectorized operation
    # Returns a (num_records, num_records) matrix where [i][j] is the cosine distance
    # between record i and record j. 0.0 = identical, 1.0 = completely different
    def get_distance_matrix(self, act_matrix: np.ndarray) -> np.ndarray:
        return cdist(act_matrix, act_matrix, metric='cosine')
    
    def find_all_similar_pairs(self, inference_results: list[dict], low: float = 0.0, high: float = 0.2) -> list[dict]:

        if self.distance_matrix is None:
            self.distance_matrix = self.get_distance_matrix(self.activation_matrix)

        # Get indices of all pairs below threshold in one vectorized call
        #TODO: need to filter distance matrix to be same size as inference matrix: dist: 535,  535  inf: 526

        rows, cols = np.where(
            (self.distance_matrix >= low) & (self.distance_matrix <= high)
        )

        pairs = []
        # distance_matrix[i][j] is the cosine distance between record i and record j
        for i, j in zip(rows, cols): #TODO: verify the inference_results are in the correct order to support this
            if i >= len(inference_results) or j >= len(inference_results): #filtered inf results will be less than dist matrix
                break
            if i < j: #i < j required to avoid duplicates
                pairs.append({
                    'id_a':     self.id_map[i],
                    'id_b':     self.id_map[j],
                    'distance': float(self.distance_matrix[i][j]),
                    'label_a':  inference_results[i]['label'],
                    'label_b':  inference_results[j]['label'],
                })

        return sorted(pairs, key=lambda x: x['distance'])

    #TODO: create a version of find_all_similar_pairs that uses a percentiles to find similarity, rather than absolute measurements
        
    def get_cluster_plot_data(self, inference_results) -> dict:
        """
        Reduces activation vectors to 2D using UMAP (preferred) or t-SNE fallback,
        then returns plot-ready data points with labels and record IDs.
        """

        # Normalise before dimensionality reduction
        scaled = StandardScaler().fit_transform(self.activation_matrix)
  
        reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
        coords  = reducer.fit_transform(scaled)
        method  = 'UMAP'

        points = []
        for i, record in enumerate(inference_results): #TODO: verify this is based on activation vectors
            points.append({
                'id':        self.id_map[i],
                'x':         float(coords[i, 0]),
                'y':         float(coords[i, 1]),
                'label':     record.get('label'),
                'predicted': record.get('predicted'),
                'correct':   record.get('correct'),
            })

        return {
            'method': method,
            'points': points,
        }
    

        