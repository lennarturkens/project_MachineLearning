from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch



def tsne_domain(device, model_feature_encoder, source_test, target_test, num_test_batches, batchsize):

    features_all = torch.zeros(2*batchsize*num_test_batches, 128)
    y_all = torch.zeros(2*batchsize*num_test_batches)


    with torch.no_grad():
    
        batch_count = 0

        for source_batch in source_test:
        
            X, y = source_batch
            X = X.to(device)
        
            features = model_feature_encoder(X)
        
            features_all[batch_count*batchsize:(batch_count+1)*batchsize,:] = features
            y_all[batch_count*batchsize:(batch_count+1)*batchsize] = y
        
            batch_count += 1
        
            if batch_count == num_test_batches:
                break
        
        for target_batch in target_test:
        
            X, y = target_batch
            X = X.to(device)
        
            features = model_feature_encoder(X)
        
            features_all[batch_count*batchsize:(batch_count+1)*batchsize,:] = features
            y_all[batch_count*batchsize:(batch_count+1)*batchsize] = y
        
    
            batch_count += 1   
        
            if batch_count == 2*num_test_batches:
                break
        
    features_all = features_all.cpu().detach().numpy()
    tsne = TSNE(n_components=2, random_state=0)
    features_red = tsne.fit_transform(features_all)
    
    return features_red


def show_reduced_features(device, model_encoders, interesting_points, lambdas, source_test, target_test, save=False):
    
    num_test_batches = 62 

    fig = plt.figure()
    fig.set_size_inches(40, 10)

    for i in range(len(interesting_points)):
        
        idx = interesting_points[i]
        features_red = tsne_domain(device, model_encoders[idx], source_test, target_test, num_test_batches, 32)
        ax = fig.add_subplot(1,len(interesting_points),i+1)
        ax.set_title("$\lambda = $" + str(lambdas[i]), fontsize=40)
        ax = plt.scatter(features_red[:32*num_test_batches, 0], features_red[:32*num_test_batches, 1], color="blue")#, c=domain_all[:32*num_TSNE_batches], cmap='Paired')
        ax = plt.scatter(features_red[32*num_test_batches:, 0], features_red[32*num_test_batches:, 1], color="red")#, c=domain_all[32*num_TSNE_batches:], cmap='Paired')
    
    if save == True:
        plt.savefig('simnet_featurereduction.png')
        
    plt.show()
    
def show_PCA_lambda(lambda_loss, target_PCA_accs, save=False):

    plt.plot(lambda_loss, target_PCA_accs, color="black")
    plt.title("PCA accuracy as function of hyperparameter $\lambda$", fontsize=15)
    plt.xlabel("$\lambda$", fontsize=20)
    plt.ylabel("PCA accuracy", fontsize=20)
    plt.tight_layout()
    
    if save == True:
        plt.savefig("PCAA_simnet")
        
    plt.show()