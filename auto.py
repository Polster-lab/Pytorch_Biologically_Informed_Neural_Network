import yaml

# Define the base paths and cell types
base_path = '/12tb_dsk1/usman/Pytorch_Biologically_Informed_Neural_Network/'
data_path = f'{base_path}Preprocessed_data/excitory_neurons/'
model_path = f'{base_path}model_save/excitory_neurons/'

cell_types = [
    'Exc_L2-3_CBLN2_LINC02306',
    'Exc_L3_4_RORB_CUX2',
    'Exc_L3_5_RORB_PLCH1',
    'Exc_L4_5_RORB_GABRG1',
    'Exc_L4_5_RORB_IL1RAPL2',
    'Exc_L5_6_IT_Car3',
    'Exc_L5_6_NP',
    'Exc_L5_6_RORB_LINC02196',
    'Exc_L5_ET',
    'Exc_L6_CT',
    'Exc_L6_THEMIS_NFIA',
    'Exc_L6b',
    'Exc_NRGN'
]  # List all your cell types here

# Parameters that don't change
static_config = {
    'train': {
        'epochs': 200,
        'learning_rate': 0.01,
        'weight_decay': 0.0001,
        'batch_size': 1024
    },
    'gene_expression': {
        'highly_expressed_threshold': 0.95,#from 0.95
        'lowly_expressed_threshold': 0.90,#from 0.90
        'normalization': True,
        'marker': True,
        'print_information': True
    },
    'pathways_network': {
        'species': 'human',
        'n_hidden_layer': 4,
        'pathway_relation': '/12tb_dsk1/usman/CellTICS/reactome/ReactomePathwaysRelation.txt',
        'pathway_names': '/12tb_dsk1/usman/CellTICS/reactome/ReactomePathways.txt',
        'ensemble_pathway_relation': '/12tb_dsk1/usman/CellTICS/reactome/Ensembl2Reactome_All_Levels.txt',
        'datatype': 'diagnosis'
    }
}

for cell_type in cell_types:
    config = {
        'dataset': {
            'train': f'{data_path}{cell_type}/train.csv',
            'test': f'{data_path}{cell_type}/test.csv',
            'val': f'{data_path}{cell_type}/val.csv',
            'y_train': f'{data_path}{cell_type}/y_train.csv',
            'y_test': f'{data_path}{cell_type}/y_test.csv',
            'y_val': f'{data_path}{cell_type}/y_val.csv'
        },
        'model_output': {
            'model_save_dir': f'{model_path}{cell_type}/'
        }
    }

    # Merge the static and dynamic parts of the configuration
    config.update(static_config)

    # Write the config to a new YAML file for each cell type
    with open(f'config_{cell_type}.yml', 'w') as file:
        yaml.dump(config, file)

    print(f'Config file created for {cell_type}')


