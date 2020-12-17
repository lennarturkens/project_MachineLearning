"""tt_simnet.py
(pre-)train and test functions for simnet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def pretrain(device, model, optimizer, num_epochs, trainset, regularization=False, gamma_reg=0):
    
    for epoch in range(num_epochs):
        
        loss_avg = 0 
        num_batches = 0
    
        for batch in trainset:
            
            X, y = batch
            X, y = X.to(device), y.to(device)
                
            model.zero_grad() 
            loss = 0
            
            if regularization == False:
                model_output = model(X)
                loss += F.nll_loss(model_output, y)   
                
            if regularization == True:
                prototypes = torch.zeros(10, 128).to(device)
                num_images = torch.zeros(10).to(device)
                
                h = model.encoder(X)
                
                for i in range(0, len(X)):
                    label = int(y[i])
                    prototypes[label] += h[i]
                    num_images[label] += 1
        
                for i in range(0, 10):
                    if num_images[i] != 0:
                        prototypes[i] /= num_images[i]
                
                regularizer = torch.norm(torch.matmul(prototypes.t(), prototypes) - torch.eye(128).to(device))
                model_output = model.classifier(h)
                loss += F.nll_loss(model_output, y) + gamma_reg * regularizer
                
            
            loss.backward()
            optimizer.step()
        
            with torch.no_grad():
                loss_avg += loss
                num_batches += 1
                
        print("Epoch : % 2d, Average loss : % 5.2f" %(epoch, loss_avg/num_batches)) 

    return model


def get_random_prototype_images(device, dataset):
        
    prototype_images = torch.zeros(10,1,28,28).to(device)
    proto_count = 0
        
    for batch in dataset:
    
        X_s, y_s = batch
        X_s = X_s.to(device)
    
        for j in range(0, len(y_s)):
        
            if torch.sum(prototype_images[int(y_s[j])]) == 0:
                prototype_images[int(y_s[j])] = X_s[j].view(1, 1, 28, 28)
                proto_count += 1
                
        
        if proto_count == 10:
            break
                 
    return prototype_images 


def calculate_test_prototypes(device, model, source_test):
    
    with torch.no_grad():

        prototypes = torch.zeros(128, 10).to(device)
        num_images = torch.zeros(10).to(device)

        for batch in source_test:
            X, y = batch
            X = X.to(device)

            batchsize = len(y)

            for i in range(0, batchsize):
        
                label = int(y[i])
                g_out = model.gnet(X[i].view(1,1,28,28))
                g_out = g_out.view(128)
        
                prototypes[:, label] += g_out
                
                num_images[label] += 1
            
        for j in range(0,10):
            prototypes[:, j] /= num_images[j]
        
    return prototypes


def test_simnet(device, model, source_test, target_test):
    
    with torch.no_grad():
        class_count = torch.zeros(10).to(device)
        class_accurate_count = torch.zeros(10).to(device)
        
        test_proto = calculate_test_prototypes(device, model, source_test)
    
        for batch in target_test:
    
            X, y = batch
            X, y = X.to(device), y.to(device)

            [class_out, disc_out, encoded_proto] = model(X, mode="test", encoded_proto=test_proto.t())
            y_pred = torch.argmax(class_out, dim=1).to(device)
        
            for i in range(0, len(y)):
        
                if y_pred[i] == y[i]:
                    class_accurate_count[int(y[i])] += 1
                class_count[int(y[i])] += 1
        
        PCA_accuracy = torch.mean(torch.div(class_accurate_count, class_count))
    
    return PCA_accuracy


def train_simnet(device, model, optimizer, num_epochs, source_train, target_train, source_test, target_test, lambda_loss, gamma_reg):

    target_PCA_accs = torch.zeros(num_epochs)
    optimal_model = None
    
    for epoch in range(num_epochs):
        loss_class_avg = 0 
        num_batches = 0
        
        for batch_source, batch_target in zip(source_train, target_train):
           
            optimizer.zero_grad()
            
            random_proto_images = get_random_prototype_images(device, source_train)
            
            X_s, y_s = batch_source
            X_s, y_s = X_s.to(device), y_s.to(device)
            
            X_t, y_t = batch_target
            X_t, y_t = X_t.to(device), y_t.to(device)
                        
            [source_class_out, source_disc_out, encoded_proto] = model(X_s, mode="train", proto_images=random_proto_images)
            [target_class_out, target_disc_out, encoded_proto] = model(X_t, mode="train", proto_images=random_proto_images)
            regularizer = torch.norm(torch.matmul(encoded_proto.t(), encoded_proto) - torch.eye(128).to(device))
            
            loss_class = F.nll_loss(source_class_out, y_s) + gamma_reg * regularizer
            loss_disc = - torch.mean(torch.log(source_disc_out + 1e-7)) - torch.mean(torch.log(1 - target_disc_out + 1e-7))
            total_loss = loss_class - lambda_loss * loss_disc
                
            total_loss.backward()
            optimizer.step()
        
    
            with torch.no_grad():
                loss_class_avg += loss_class
                num_batches += 1
         
        with torch.no_grad():
            loss_class_avg = loss_class_avg/num_batches
            target_PCA_accs[epoch] = test_simnet(device, model, source_test, target_test)
            if target_PCA_accs[epoch] == torch.max(target_PCA_accs):
                optimal_model = model

            
            
        print("Epoch : % 2d, Average class loss : % 5.2f, target_PCA_acc : % 5.2f" %(epoch, loss_class_avg, target_PCA_accs[epoch])) 

        
    return [optimal_model, target_PCA_accs]