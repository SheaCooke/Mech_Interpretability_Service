
import numpy as np
from scipy.spatial.distance import cdist
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from ..api.types import PredictionFilter


#TODO: does this create duplicate vectors?
#TODO: does this take in inference results or activation vectors?

#TODO: when filtering for incorrect vectors, they dont appear on the cluster or similar pairs, but they do appear when everything
# is displayed. probably an issue with how things are filtered.

class Vector_Analyzer:
    def __init__(self, inference_results: list[dict]): #TODO: inference results are stored in the session, dont need duplicate storage here
        #TODO: do all these need to be computed and stored on initialization, or can they be created as needed?
        #TODO: find way to walk through the code and see how much time/memory every step uses
        self.id_map: np.ndarray = self.get_id_mapping(inference_results)
        self.incorrect_ids = {rec['id'] for rec in inference_results if rec['correct'] == False}

        #holds activation vectors for all records within inference_results. Consolidated here for faster computation 
        self.activation_matrix: np.ndarray = self.get_activation_matrix(inference_results)

        #Matrix of i by j. where matrix[i][j] is the cosine distance betwen the activation vectors for record i and record j
        #both dimensions = the number of records
        self.distance_matrix: np.ndarray = None #lazy loaded in find_all_similar_pairs. 
    

    def get_id_mapping(self, inf_results: list[dict]) -> np.ndarray:
        return [r['id'] for r in inf_results]

    # Stack all activation vectors into a 2D matrix of shape (num_records, total_activation_size)
    def get_activation_matrix(self, inf_results: list[dict]) -> np.ndarray:
        return np.vstack([r['activations'] for r in inf_results]) #preserves the order of the input list

    # Compute pairwise cosine distances in a single vectorized operation
    # Returns a (num_records, num_records) matrix where [i][j] is the cosine distance
    # between record i and record j. 0.0 = identical, 1.0 = completely different
    def get_distance_matrix(self, act_matrix: np.ndarray) -> np.ndarray:
        return cdist(act_matrix, act_matrix, metric='cosine') #preserves the order of the input
    
    def find_all_similar_pairs(self, inference_results: list[dict], low: float = 0.0, high: float = 0.2, filter: PredictionFilter = PredictionFilter.ALL) -> list[dict]:

        if self.distance_matrix is None:
            self.distance_matrix = self.get_distance_matrix(self.activation_matrix)

        # Get indices of all pairs below threshold in one vectorized call
        rows, cols = np.where(
            (self.distance_matrix >= low) & (self.distance_matrix <= high)
        )

        pairs = []
        for i, j in zip(rows, cols):
            if i >= len(inference_results) or j >= len(inference_results): #filtered inf results will be less than dist matrix
                break
            if i < j: #i < j required to avoid duplicates
                pairs.append({
                    'id_a':     self.id_map[i],
                    'id_b':     self.id_map[j],
                    'distance': float(self.distance_matrix[i][j]),
                    'label_a':  inference_results[i]['label'],
                    'label_b':  inference_results[j]['label']
                })
        
        #Distance matrix is broken if the order of the inference records changes, filter later
        if filter == PredictionFilter.INCORRECT:
            pairs = [pair for pair in pairs if pair['id_a'] in self.incorrect_ids or pair['id_b'] in self.incorrect_ids]
        elif filter == PredictionFilter.CORRECT:
            pairs = [pair for pair in pairs if pair['id_a'] not in self.incorrect_ids and pair['id_b'] not in self.incorrect_ids]

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
        for i, record in enumerate(inference_results):
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
    

        