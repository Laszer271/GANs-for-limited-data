import json
import processing_framework 
import sys, inspect
POSSIBLE_OPERATIONS_MAPPING = inspect.getmembers(sys.modules[processing_framework.__name__], inspect.isclass)
POSSIBLE_OPERATIONS_MAPPING = {op[0]: op[1] for op in POSSIBLE_OPERATIONS_MAPPING}

def do_processing(path):
    print('='*50)
    with open(path, "r") as f:
        data = json.load(f)
    
    source_paths = data['SourcePaths']
    
    sequencer = processing_framework.Sequencer(verbose=2)

    for operation in data['Operations']:
        name = operation['OpName']
        del operation['OpName']
        kwargs = operation
        
        operation = POSSIBLE_OPERATIONS_MAPPING[name](**kwargs)
        sequencer.add_operation(operation)
        
    data = sequencer(paths=source_paths)
    return data

if __name__ == '__main__':
    paths = ['configs/processing/pokemon_pixelart.json',
             'configs/processing/pokemon_artwork.json',
             'configs/processing/margonem.json',
             'configs/processing/iconset.json',
             'configs/processing/profantasy.json',
             'configs/processing/dnd.json',]
    
    for path in paths:
        d = do_processing(path)