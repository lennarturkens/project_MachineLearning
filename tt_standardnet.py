"""tt_standardnet.py 
train and test functions for standardnet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import model


def test_standardnet(device, model, target_test):
    """test_standardnet function to calculate the PCA accuracy of the targetset"""
    
    with torch.no_grad():
        class_count = torch.zeros(10).to(device)
        class_accurate_count = torch.zeros(10).to(device)
            
        for batch in target_test:
    
            X, y = batch
            X, y = X.to(device), y.to(device)

            [class_out, disc_out] = model(X)
            y_pred = torch.argmax(class_out, dim=1).to(device)
        
            for i in range(0, len(y)):
        
                if y_pred[i] == y[i]:
                    class_accurate_count[int(y[i])] += 1
                class_count[int(y[i])] += 1
        
        PCA_accuracy = torch.mean(torch.div(class_accurate_count, class_count))
    
    return PCA_accuracy

def train_standardnet(device, model, optimizer, lambda_loss, num_epochs, source_train, target_train, source_test, target_test):
    """train_standardnet function to train the standardnet"""

    
    optimal_PCA_accuracy = 0
    optimal_model = None
    
    for epoch in range(num_epochs):
        
        loss_avg = 0 
        num_batches = 0
    
        for batch_source, batch_target in zip(source_train, target_train):
                                    
            model.zero_grad() 

            X_source, y_source = batch_source
            X_source, y_source = X_source.to(device), y_source.to(device)
            
            X_target, y_target = batch_target
            X_target, y_target = X_target.to(device), y_target.to(device)
                
            [source_class_output, source_disc_output] = model(X_source)
            [target_class_output, target_disc_output] = model(X_target)

            loss_class = F.nll_loss(source_class_output, y_source)  
            loss_disc = - torch.mean(torch.log(source_disc_output + 1e-7)) - torch.mean(torch.log(1 - target_disc_output + 1e-7))
            
            total_loss = loss_class - lambda_loss * loss_disc
            total_loss.backward()
            
            optimizer.step()
        
            with torch.no_grad():
                loss_avg += loss_class
                num_batches += 1
    
                
        with torch.no_grad():
            loss_avg /= num_batches
            PCA_accuracy = test_standardnet(device, model, target_test)
            
            if PCA_accuracy > optimal_PCA_accuracy:
                optimal_PCA_accuracy = PCA_accuracy
                optimal_model = model
                
        print("Epoch : % 2d, Average loss : % 5.2f, PCA accuracy : % 5.2f" %(epoch, loss_avg, PCA_accuracy)) 

        
    return [optimal_model, optimal_PCA_accuracy]

