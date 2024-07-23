import os
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import argparse
from utils import *
from gene_expression import *
from pathway_hierarchy import *
import pandas as pd
import yaml
from custom_neural_network import *



class TabularDataset(Dataset):
    def __init__(self, count_matrix, label):
        # Read the CSV file
        self.data = count_matrix
        # Separate features and target
        self.features = self.data.values
        self.target = label.values
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get features and target for a given index
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.target[idx], dtype=torch.float32)
        return features, target

def evaluate(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    predicted_list = []
    labels_list = []
    
    with torch.no_grad():  # No need to compute gradients during evaluation
        for features, labels in dataloader:
            outputs = model(features)
            #print(outputs)
            predicted = torch.round(torch.sigmoid(outputs.data))
            #print(outputs)
            #print(predicted)
            #_, predicted = torch.sigmoid(outputs.data)
            predicted_list.append(predicted)
            labels_list.append(labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    #print(total)
    accuracy = 100 * correct / total
    return accuracy, predicted_list, labels_list




def model(train_dataloader , val_dataloader, layers_node, masking, output_layer, learning_rate=0.001, num_epochs=50, weight_decay = 0):

    model = CustomNetwork(layers_node, output_layer, masking)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,weight_decay = weight_decay )  # Using SGD with momentum
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    patience = 20
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    early_stop = False
    
    for epoch in tqdm(range(num_epochs)):
        if early_stop:
            print("Early stopping")
            break
        epoch_cost = 0.
        
        for batch_features,batch_targets in train_dataloader:
            outputs = model(batch_features)
            #print(outputs)
            #print(batch_targets)
            #print(outputs)
            loss = criterion(outputs, batch_targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        
        train_accuracy, predicted_list_train, labels_list_train = evaluate(model, train_dataloader)
        val_accuracy, predicted_list_val, labels_list_val = evaluate(model, val_dataloader)
        #scheduler.step(val_accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Train_accuracy: {train_accuracy}, Val_accuracy: {val_accuracy}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
        # Save the best model
            torch.save(model.state_dict(), f'best_model_{output_layer}.pth')
            print('Model saved.')
        else:
            epochs_no_improve += 1
    
        # Early stopping
        '''if epochs_no_improve >= patience:
            early_stop = True
            print("Early stopping triggered")'''
        
    
    train_accuracy, predicted_list_train, labels_list_train = evaluate(model, train_dataloader)
    val_accuracy, predicted_list_val, labels_list_val = evaluate(model, val_dataloader)
    
    output_train = (predicted_list_train, labels_list_train)
    output_val = (predicted_list_val, labels_list_val)
    
    return output_train, output_val,model

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)



def main():

    parser = argparse.ArgumentParser(description='Sample application with config and argparse')
    parser.add_argument('--config', type=str, default='config.yml', help='Path to the configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    train = pd.read_csv(config['dataset']['train'],index_col=0)
    test = pd.read_csv(config['dataset']['test'],index_col=0)
    val = pd.read_csv(config['dataset']['val'],index_col=0)

    y_train = pd.read_csv(config['dataset']['y_train'])
    y_test = pd.read_csv(config['dataset']['y_test'])
    y_val = pd.read_csv(config['dataset']['y_val'])



    r_data_tmp = train.T
    q_data_tmp = test.T
    v_data_tmp = val.T
    r_label_tmp = y_train
    train_x, test_x, val_x, train_y = get_expression(r_data_tmp,
                                                q_data_tmp,
                                                v_data_tmp,
                                                r_label_tmp,
                                                thrh=config['gene_expression']['highly_expressed_threshold'],
                                                thrl=config['gene_expression']['lowly_expressed_threshold'],
                                                normalization=config['gene_expression']['normalization'],
                                                marker=config['gene_expression']['marker'])
    

    pathway_genes = get_gene_pathways(config['pathways_network']['ensembl_pathway_relation'], species=config['pathways_network']['species'])

    masking, layers_node, train_x, test_x,val_x = get_masking(config['pathways_network']['pathway_names'],
                                                        pathway_genes,
                                                        config['pathways_network']['pathway_relation'],
                                                        train_x,
                                                        test_x,
                                                        val_x,
                                                        train_y,
                                                        config['pathways_network']['datatype'],
                                                        config['pathways_network']['species'],
                                                        config['pathways_network']['n_hidden'])


    try:
        masking = list(masking.values())
        layers_node = list(layers_node.values())
    except:
        print('already_done')


    '''train_dataset = TabularDataset(train_x.T,train_y)
    val_dataset = TabularDataset(val_x.T,y_val)
    test_dataset = TabularDataset(test_x.T,y_test)  


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle= True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle= True)
    # Example of iterating through the DataLoader


    pred_y_df = pd.DataFrame(data=0, index=test_x.columns, columns=list(range(2, len(masking) + 2)))
    train_y_df = pd.DataFrame(data=0, index=train_x.columns, columns=list(range(2, len(masking) + 2)))
    model_dict = dict()
    activation_output = {}
    for output_layer in range(2, len(masking) + 2):
        if print_information:
            print("Current sub-neural network has " + str(output_layer - 1) + " hidden layers.")
        output_train, output_test,model_dict[output_layer] = model(train_dataloader,
                                            val_dataloader,
                                            layers_node,
                                            masking,
                                            output_layer,
                                            learning_rate=learning_rate,num_epochs=num_epochs,weight_decay = l2_regularization
                                            )  '''

if __name__ == "__main__":
    main()


                                            


                            