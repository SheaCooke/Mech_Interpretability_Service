
import numpy as np
from scipy.spatial.distance import cdist
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


#TODO: does this create duplicate vectors?
#TODO: does this take in inference results or activation vectors?

class Vector_Analyzer:
    def __init__(self, inference_results: list[dict]):
        self.inference_results: list[dict] = inference_results
        self.id_map = self.get_id_mapping(inference_results)
        self.activation_matrix = self.get_activation_matrix(inference_results)
        self.distance_matrix = self.get_distance_matrix(self.activation_matrix)
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
    
    def find_all_similar_pairs(self, low: float = 0.0, high: float = 0.2) -> list[dict]:
        # Get indices of all pairs below threshold in one vectorized call
        rows, cols = np.where(
            (self.distance_matrix >= low) & (self.distance_matrix <= high)
        )

        pairs = []
        for i, j in zip(rows, cols):
            if i < j:  # upper triangle only
                pairs.append({
                    'id_a':     self.id_map[i],
                    'id_b':     self.id_map[j],
                    'distance': float(self.distance_matrix[i][j]),
                    'label_a':  self.inference_results[i]['label'],
                    'label_b':  self.inference_results[j]['label'],
                })

        return sorted(pairs, key=lambda x: x['distance'])

    #TODO: create a version of find_all_similar_pairs that uses a percentiles to find similarity, rather than absolute measurements
        
    def get_cluster_plot_data(self) -> dict:
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
        for i, record in enumerate(self.inference_results):
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
    

        