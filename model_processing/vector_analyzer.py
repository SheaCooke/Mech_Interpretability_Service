
class Vector_Analyzer:
    def __init__(self, inference_results: list[dict]):
        self.inference_results: list[dict] = inference_results
        self.record_av_mapping = self.create_record_av_mapping(inference_results)
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
    
    """
    maps the record id to a 2 dimensional list of activation vectors
    """
    def create_record_av_mapping(self, inference_results) -> dict[str,list[list]]:
        mapping = {}
        for result in inference_results:
            mapping[result['id']] = result['activations']
        
        return mapping

    

        