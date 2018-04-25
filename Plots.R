## Keith Mertan ##
## March 30, 2018 ##
## CS 7641 - Machine Learning ##
## Create plots for Assignment 3 - Unsupervised Learning and Dimensionality Reduction ##

library(dplyr)
library(magrittr)

## CHANGE WORKING DIRECTORY to 'kmertan - assignment 3' once this is turned in 
## (since assignment 4 will also need to be in the 'kmertan' folder)

setwd('/Users/kmertan/Documents/CS7641/kmertan3')

## Part 1: Run the clustering algorithms on your datasets and describe what you see

## vv Global Variables vv ##
{  datasets <- c('BASE',
               'PCA',
               'ICA',
               'RP',
               'RF')
  
  datanames <- c('cancer',
                 'contra')
  
  algos <- c('gmm', 'km')
  
  optimals <- data.frame(datasets = rep(datasets, length(datanames) * 2),
                         datanames = c(rep(datanames[1], length(datasets) * 2), rep(datanames[2], length(datasets) * 2)),
                         clust_algo = rep(algos, length(datasets) * length(datanames)))
  
  optimals %<>% arrange(datanames, clust_algo, datasets)
  
  optimals$value <- c(5, 3, 8, 7, 6,
                       5, 4, 8, 4, 4,
                       7, 9, 8, 7, 9,
                       7, 7, 8, 5, 6)
}


clust_plots <- function(data){
  oldwd <- getwd()
  setwd(paste0(oldwd, '/output/', data, '/'))
  
  clusters = 2:10
  
  contra_gmm_line = optimals %>% filter(datasets == data, datanames == 'contra', clust_algo == 'gmm') %>% select(value) %>% as.numeric()
  contra_km_line = optimals %>% filter(datasets == data, datanames == 'contra', clust_algo == 'km') %>% select(value) %>% as.numeric()
  
  cancer_gmm_line = optimals %>% filter(datasets == data, datanames == 'cancer', clust_algo == 'gmm') %>% select(value) %>% as.numeric()
  cancer_km_line = optimals %>% filter(datasets == data, datanames == 'cancer', clust_algo == 'km') %>% select(value) %>% as.numeric()
  
  base_km_sse <- read.csv('SSE.csv')
  base_gmm_ll <- read.csv('logliklihood.csv')
  
  jpeg(paste0('../plots/', data, ' - Elbow Method - K Means.jpg'))
  
  plot(clusters, base_km_sse$cancer.SSE..left., type = 'o',
       ylim = c(min(base_km_sse[ , -1]), max(base_km_sse[ , -1])),
       xlab = 'Number of Clusters',
       ylab = 'SSE',
       col = 'blue',
       main = paste0('K Means', ifelse(data == 'BASE', '', paste0(' - ', data)), '\nElbow Method for Determining Optimal Number of Clusters'))
  lines(clusters, base_km_sse$contra.SSE..left., col = 'red', type = 'o')
  legend(x = 'topright', legend = c('Breast Cancer', 'Contraceptive Use'),
         lty = 1, col = c('blue', 'red'),
         cex = .6)
  abline(v = cancer_km_line, lty = 2, col = 'blue')
  abline(v = contra_km_line, lty = 2, col = 'red')
  
  dev.off()
  
  
  jpeg(paste0('../plots/', data, ' - Elbow Method - GMM.jpg'))
  
  plot(clusters, base_gmm_ll$cancer.log.likelihood, type = 'o',
       ylim = c(min(base_gmm_ll[ , -1]), max(base_gmm_ll[ , -1])),
       xlab = 'Number of Clusters',
       ylab = 'Log Liklihood',
       col = 'blue',
       main = paste0('GMM', ifelse(data == 'BASE', '', paste0(' - ', data)), '\nElbow Method for Determining Optimal Number of Clusters'))
  lines(clusters, base_gmm_ll$contra.log.likelihood, col = 'red', type = 'o')
  legend(x = 'topleft', legend = c('Breast Cancer', 'Contraceptive Use'),
         lty = 1, col = c('blue', 'red'),
         cex = .6)
  abline(v = cancer_gmm_line, lty = 2, col = 'blue')
  abline(v = contra_gmm_line, lty = 2, col = 'red')
  
  dev.off()
  
  jpeg(paste0('../plots/', data, ' - Cluster Validation - GMM.jpg'))
  
  cancer_MI <- read.csv('cancer adjMI.csv')
  contra_MI <- read.csv('contra adjMI.csv')
  
  cancer_MI %$% plot(clusters, t(.[1, -1]), type = 'o',
       ylim = c(min(.[ , -1]), max(.[ , -1])),
       xlab = 'Number of Clusters',
       ylab = 'Adjusted Mutual Information',
       col = 'blue',
       main = paste0('GMM', ifelse(data == 'BASE', '', paste0(' - ', data)), '\nValidating Clusters Against Labels using AMI'))
  contra_MI %$% lines(clusters, t(.[1, -1]), type = 'o', col = 'red')
  legend(x = 'topright', legend = c('Breast Cancer', 'Contraceptive Use'),
         lty = 1, col = c('blue', 'red'),
         cex = .6)
  
  dev.off()
  
  
  jpeg(paste0('../plots/', data, ' - Cluster Validation - K Means.jpg'))
  
  cancer_MI %$% plot(clusters, t(.[2, -1]), type = 'o',
                     ylim = c(min(.[ , -1]), max(.[ , -1])),
                     xlab = 'Number of Clusters',
                     ylab = 'Adjusted Mutual Information',
                     col = 'blue',
                     main = paste0('K Means', ifelse(data == 'BASE', '', paste0(' - ', data)), '\nValidating Clusters Against Labels using AMI'))
  contra_MI %$% lines(clusters, t(.[2, -1]), type = 'o', col = 'red')
  legend(x = 'topright', legend = c('Breast Cancer', 'Contraceptive Use'),
         lty = 1, col = c('blue', 'red'),
         cex = .6)
  
  dev.off()
  
  setwd(oldwd)
}

for(data in datasets){
  clust_plots(data)
}

clust_plots_2d <- function(data){
  stopifnot(data %in% datasets)
  
  for(name in datanames){
    
    title = NA
    if(name == 'cancer'){
      title <- 'Breast Cancer'
    }
    else{
      title <- 'Contraceptive Choice' 
    }
    
    twod <- read.csv(paste0('./output/', data, '/', name, '2D.csv'))
    
    jpeg(paste0('./output/plots/', data, name, ' GMM 2D.jpg'))
    
    twod %$% plot(x, y,
                  col = gmm_cluster + 1,
                  pch = target,
                  xaxt = 'n', yaxt = 'n',
                  xlab = '', ylab = '',
                  main = paste0(title, ifelse(data == 'BASE', '', data), '\n2D Projected GMM Clusters'))
    
    dev.off()
    
    jpeg(paste0('./output/plots/', data, name, ' KM 2D.jpg'))
    
    twod %$% plot(x, y,
                  col = km_cluster + 1,
                  pch = target,
                  xaxt = 'n', yaxt = 'n',
                  xlab = '', ylab = '',
                  main = paste0(title, ifelse(data == 'BASE', '', data), '\n2D Projected KM Clusters'))
    
    dev.off()
  }
}

for(data in datasets){
  clust_plots_2d(data)
}

## First we read in the dimensionality reduction data for each of our data sets

dim_red_plots <- function(data){
  stopifnot(data %in% datanames)
  
  pca_cutoff <- .9
  ica_cutoff <- .2
  rf_cutoff <- .9
  
  title = NA
  if(data == 'cancer'){
    title <- 'Breast Cancer'
  }
  else{
    title <- 'Contraceptive Choice' 
  }
  
  pca <- read.csv(paste0('./output/PCA/', data, ' scree.csv'), header = F, col.names = c('num_components', 'explained_var'))
  ica <- read.csv(paste0('./output/ICA/', data, ' scree.csv'), header = F, col.names = c('num_components', 'kurtosis'))
  rp_recon <- read.csv(paste0('./output/RP/', data, ' scree2.csv'), col.names = c('num_components', paste0('recon_error_', 1:10)))
  rf <- read.csv(paste0('./output/RF/', data, ' scree.csv'), header = F, col.names = c('num_components', 'feature_importance'))
  
  
  jpeg(paste0('./output/plots/PCA ', data,  '.jpg'))
  
  pca %<>% mutate(percent_var = cumsum(explained_var)/sum(explained_var))
  pca %$% plot(num_components, percent_var, type = 'o',
               ylim = c(0, 1),
               col = 'orange',
               xlab = 'Number of Components',
               ylab = 'Cumulative Explained Variance',
               main = paste0(title, '\nPCA Number of Components'))
  abline(h = pca_cutoff, col = 'black', lty = 2)
  
  dev.off()
  
  jpeg(paste0('./output/plots/ICA ', data, '.jpg'))
  
  ica_cutoff_ind <- which(ica$kurtosis == max(ica$kurtosis))
  ica %$% barplot(kurtosis,
                  xlab = 'Number of Components',
                  ylab = 'Kurtosis',
                  col = c(rep('orange', ica_cutoff_ind - 1), 'blue', rep('orange', nrow(rf) - ica_cutoff_ind)),
                  main = paste0(title, '\nICA Number of Components'),
                  names.arg = .$num_components)
  
  dev.off()
  
  ## Both datasets here happen to have their overall lowest reconstruction error on the seventh iteration of RP
  ## If this wasn't true, I would split this into two functions and specify the min column for each
  
  jpeg(paste0('./output/plots/RP ', data, '.jpg'))
  
  rp_recon %$% plot(num_components, recon_error_7, type = 'o',
                    ylim = c(0, 1),
                    col = 'orange',
                    xlab = 'Number of Components',
                    ylab = 'Reconstruction Error',
                    main = paste0(title, '\nRandom Projection Number of Components'))
  abline(h = ica_cutoff, col = 'black', lty = 2)
  
  dev.off()
  
  jpeg(paste0('./output/plots/RF ', data, '.jpg'))
  
  rf %<>% mutate(percent_imp = cumsum(feature_importance)/sum(feature_importance))
  rf_cutoff_ind <- which(rf$percent_imp > rf_cutoff)[1]
  
  rf %$% barplot(feature_importance,
                 xlab = 'Features',
                 ylab = 'Feature Importance',
                 col = c(rep('blue', rf_cutoff_ind), rep('orange', nrow(rf) - rf_cutoff_ind)),
                 main = paste0(title, '\nRandom Forest Feature Selection'))
  
  dev.off()
}

for(data in datanames){
  dim_red_plots(data)
}


## NN ##

# This gets maximum mean_test_scores for dim_red data
{
optimal_nn_arch <- c("{'NN__alpha': 0.01, 'NN__hidden_layer_sizes': (12, 12, 12)}", 
                     "{'NN__alpha': 1.0, 'NN__hidden_layer_sizes': (18, 18)}")
names(optimal_nn_arch) <- c('cancer', 'contra')

cancer_best_hl <- '(12, 12, 12)'
cancer_best_alpha <- .01
cancer_bests <- c()
cancer_overall_bests <- c()

contra_best_hl <- '(18, 18)'
contra_best_alpha <- 1
contra_bests <- c()
contra_overall_bests <- c()

cancer_pca_best <- 6
cancer_ica_best <- 2
cancer_rp_best <- 8
cancer_rf_best <- 6

contra_pca_best <- 9
contra_ica_best <- 7
contra_rp_best <- 10
contra_rf_best <- 7
  
  
  
cancer_nn <- read.csv('./BASE/cancer NN bmk.csv')
contra_nn <- read.csv('./BASE/contra NN bmk.csv')

cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                     param_NN__hidden_layer_sizes == cancer_best_hl) %>%
              select(mean_test_score) %>% as.numeric()
contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                     param_NN__hidden_layer_sizes == contra_best_hl) %>%
              select(mean_test_score) %>% as.numeric()

cancer_bests <- c(cancer_bests, cancer_best)
contra_bests <- c(contra_bests, contra_best)
cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))


cancer_nn <- read.csv('./output/PCA/cancer dim red.csv')
contra_nn <- read.csv('./output/PCA/contra dim red.csv')

cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                    param_NN__hidden_layer_sizes == cancer_best_hl,
                                    param_pca__n_components == cancer_pca_best) %>%
  select(mean_test_score) %>% as.numeric()
contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                    param_NN__hidden_layer_sizes == contra_best_hl,
                                    param_pca__n_components == contra_pca_best) %>%
  select(mean_test_score) %>% as.numeric()

cancer_bests <- c(cancer_bests, cancer_best)
contra_bests <- c(contra_bests, contra_best)
cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))


cancer_nn <- read.csv('./output/ICA/cancer dim red.csv')
contra_nn <- read.csv('./output/ICA/contra dim red.csv')

cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                    param_NN__hidden_layer_sizes == cancer_best_hl,
                                    param_ica__n_components == cancer_ica_best) %>%
  select(mean_test_score) %>% as.numeric()
contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                    param_NN__hidden_layer_sizes == contra_best_hl,
                                    param_ica__n_components == contra_ica_best) %>%
  select(mean_test_score) %>% as.numeric()

cancer_bests <- c(cancer_bests, cancer_best)
contra_bests <- c(contra_bests, contra_best)
cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))


cancer_nn <- read.csv('./output/RP/cancer dim red.csv')
contra_nn <- read.csv('./output/RP/contra dim red.csv')

cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                    param_NN__hidden_layer_sizes == cancer_best_hl,
                                    param_rp__n_components == cancer_rp_best) %>%
  select(mean_test_score) %>% as.numeric()
contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                    param_NN__hidden_layer_sizes == contra_best_hl,
                                    param_rp__n_components == contra_rp_best) %>%
  select(mean_test_score) %>% as.numeric()

cancer_bests <- c(cancer_bests, cancer_best)
contra_bests <- c(contra_bests, contra_best)
cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))


cancer_nn <- read.csv('./output/RF/cancer dim red.csv')
contra_nn <- read.csv('./output/RF/contra dim red.csv')

cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                    param_NN__hidden_layer_sizes == cancer_best_hl,
                                    param_filter__n == cancer_rf_best) %>%
  select(mean_test_score) %>% as.numeric()
contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                    param_NN__hidden_layer_sizes == contra_best_hl,
                                    param_filter__n == contra_rf_best) %>%
  select(mean_test_score) %>% as.numeric()

cancer_bests <- c(cancer_bests, cancer_best)
contra_bests <- c(contra_bests, contra_best)
cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
}
cancer_bests
contra_bests
cancer_overall_bests
contra_overall_bests
# Plots max overall and max with baseline params
{

jpeg('./output/plots/contra NN dim red and base - baseline.jpg', height = 240)  

barplot(contra_bests, names.arg = datasets, 
        col = c('gray', rep('pink', 4)),
        ylab = 'Average Validation Accuracy',
        main = 'Contraceptive Use\nNeural Network Performance with Baseline Parameters')

dev.off()

jpeg('./output/plots/cancer NN dim red and base - baseline.jpg', height = 240)  

barplot(cancer_bests, names.arg = datasets, 
        col = c('gray', rep('pink', 4)),
        ylab = 'Average Validation Accuracy',
        main = 'Breast Cancer\nNeural Network Performance with Baseline Parameters')

dev.off()

jpeg('./output/plots/contra NN dim red and base - optimized.jpg', height = 240)  

barplot(contra_overall_bests, names.arg = datasets, 
        col = c('gray', rep('pink', 4)),
        ylab = 'Average Validation Accuracy',
        main = 'Contraceptive Use\nNeural Network Performance with Optimized Parameters')

dev.off()

jpeg('./output/plots/cancer NN dim red and base - optimized.jpg', height = 240)  

barplot(cancer_overall_bests, names.arg = datasets, 
        col = c('gray', rep('pink', 4)),
        ylab = 'Average Validation Accuracy',
        main = 'Breast Cancer\nNeural Network Performance with Optimized Parameters')

dev.off()
}




# This gets maximum mean_test_scores for cluster centers and cluster probs
{
  optimal_nn_arch <- c("{'NN__alpha': 0.01, 'NN__hidden_layer_sizes': (12, 12, 12)}", 
                       "{'NN__alpha': 1.0, 'NN__hidden_layer_sizes': (18, 18)}")
  names(optimal_nn_arch) <- c('cancer', 'contra')
  
  cancer_best_hl <- '(12, 12, 12)'
  cancer_best_alpha <- .01
  cancer_bests <- c()
  cancer_overall_bests <- c()
  
  contra_best_hl <- '(18, 18)'
  contra_best_alpha <- 1
  contra_bests <- c()
  contra_overall_bests <- c()
  
  cancer_pca_best <- 6
  cancer_ica_best <- 2
  cancer_rp_best <- 8
  cancer_rf_best <- 6
  
  contra_pca_best <- 9
  contra_ica_best <- 7
  contra_rp_best <- 10
  contra_rf_best <- 7
  
  
  ## K Means
  
  cancer_nn <- read.csv('./output/BASE/cancer cluster KMeans.csv')
  contra_nn <- read.csv('./output/BASE/contra cluster KMeans.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl, 
                                      param_km__n_clusters == 5) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      param_km__n_clusters == 7) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/PCA/cancer cluster KMeans.csv')
  contra_nn <- read.csv('./output/PCA/contra cluster KMeans.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_pca__n_components == cancer_pca_best,
                                      param_km__n_clusters == 8) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      #param_pca__n_components == contra_pca_best,
                                      param_km__n_clusters == 8) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/ICA/cancer cluster KMeans.csv')
  contra_nn <- read.csv('./output/ICA/contra cluster KMeans.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_ica__n_components == cancer_ica_best,
                                      param_km__n_clusters == 4) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      #param_ica__n_components == contra_ica_best,
                                      param_km__n_clusters == 7) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/RP/cancer cluster KMeans.csv')
  contra_nn <- read.csv('./output/RP/contra cluster KMeans.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_rp__n_components == cancer_rp_best,
                                      param_km__n_clusters == 4) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      #param_rp__n_components == contra_rp_best,
                                      param_km__n_clusters == 6) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/RF/cancer cluster KMeans.csv')
  contra_nn <- read.csv('./output/RF/contra cluster KMeans.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_filter__n == cancer_rf_best,
                                      param_km__n_clusters == 4) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                     #param_filter__n == contra_rf_best,
                                      param_km__n_clusters == 5) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))

  
  
  ## GMM
  
  cancer_nn <- read.csv('./output/BASE/cancer cluster GMM.csv')
  contra_nn <- read.csv('./output/BASE/contra cluster GMM.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl, 
                                      param_gmm__n_components == 5) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      param_gmm__n_components == 7) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/PCA/cancer cluster GMM.csv')
  contra_nn <- read.csv('./output/PCA/contra cluster GMM.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_pca__n_components == cancer_pca_best,
                                      param_gmm__n_components == 8) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      #param_pca__n_components == contra_pca_best,
                                      param_gmm__n_components == 8) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/ICA/cancer cluster GMM.csv')
  contra_nn <- read.csv('./output/ICA/contra cluster GMM.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_ica__n_components == cancer_ica_best,
                                      param_gmm__n_components == 3) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      #param_ica__n_components == contra_ica_best,
                                      param_gmm__n_components == 9) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/RP/cancer cluster GMM.csv')
  contra_nn <- read.csv('./output/RP/contra cluster GMM.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_rp__n_components == cancer_rp_best,
                                      param_gmm__n_components == 6) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      #param_rp__n_components == contra_rp_best,
                                      param_gmm__n_components == 9) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/RF/cancer cluster GMM.csv')
  contra_nn <- read.csv('./output/RF/contra cluster GMM.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_filter__n == cancer_rf_best,
                                      param_gmm__n_components == 7) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      #param_filter__n == contra_rf_best,
                                      param_gmm__n_components == 7) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
}

{
  axis_names <- merge(datasets, c('KM', 'GMM')) %$% paste0(x, '\n', y)
  jpeg('./output/plots/contra NN clust dist and base - baseline.jpg', height = 240)  
  
  barplot(contra_bests, names.arg = axis_names, 
          cex.names = .8,
          col = c('gray', rep('pink', 4)),
          ylab = 'Average Validation Accuracy',
          main = 'Contraceptive Use - Cluster Distance/Probability Only\nNeural Network Performance with Baseline Parameters')
  
  dev.off()
  
  jpeg('./output/plots/cancer NN clust dist and base - baseline.jpg', height = 240)  
  
  barplot(cancer_bests, names.arg = axis_names, 
          cex.names = .8,
          col = c('gray', rep('pink', 4)),
          ylab = 'Average Validation Accuracy',
          main = 'Breast Cancer - Cluster Distance/Probability Only\nNeural Network Performance with Baseline Parameters')
  
  dev.off()
  
  jpeg('./output/plots/contra NN clust dist and base - optimized.jpg', height = 240)  
  
  barplot(contra_overall_bests, names.arg = axis_names, 
          cex.names = .8,
          col = c('gray', rep('pink', 4)),
          ylab = 'Average Validation Accuracy',
          main = 'Contraceptive Use - Cluster Distance/Probability Only\nNeural Network Performance with Optimized Parameters')
  
  dev.off()
  
  jpeg('./output/plots/cancer NN clust dist and base - optimized.jpg', height = 240)  
  
  barplot(cancer_overall_bests, names.arg = axis_names,
          cex.names = .8,
          col = c('gray', rep('pink', 4)),
          ylab = 'Average Validation Accuracy',
          main = 'Breast Cancer - Cluster Distance/Probability Only\nNeural Network Performance with Optimized Parameters')
  
  dev.off()
}

cancer_bests
contra_bests
cancer_overall_bests
contra_overall_bests





# This gets maximum mean_test_scores for data with cluster assignments
{
  optimal_nn_arch <- c("{'NN__alpha': 0.01, 'NN__hidden_layer_sizes': (12, 12, 12)}", 
                       "{'NN__alpha': 1.0, 'NN__hidden_layer_sizes': (18, 18)}")
  names(optimal_nn_arch) <- c('cancer', 'contra')
  
  cancer_best_hl <- '(12, 12, 12)'
  cancer_best_alpha <- .01
  cancer_bests <- c()
  cancer_overall_bests <- c()
  
  contra_best_hl <- '(18, 18)'
  contra_best_alpha <- 1
  contra_bests <- c()
  contra_overall_bests <- c()
  
  cancer_pca_best <- 6
  cancer_ica_best <- 2
  cancer_rp_best <- 8
  cancer_rf_best <- 6
  
  contra_pca_best <- 9
  contra_ica_best <- 7
  contra_rp_best <- 10
  contra_rf_best <- 7
  
  
  ## K Means
  
  cancer_nn <- read.csv('./output/BASE/cancer cluster KM CLUSTER AND DATA.csv')
  contra_nn <- read.csv('./output/BASE/contra cluster KM CLUSTERS AND DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl, 
                                      param_km__n_components == 5) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      param_km__n_components == 7) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/PCA/cancer cluster KM CLUSTER AND DATA.csv')
  contra_nn <- read.csv('./output/PCA/contra cluster KM CLUSTERS AND DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_pca__n_components == cancer_pca_best,
                                      param_km__n_components == 8) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      #param_pca__n_components == contra_pca_best,
                                      param_km__n_components == 8) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/ICA/cancer cluster KM CLUSTER AND DATA.csv')
  contra_nn <- read.csv('./output/ICA/contra cluster KM CLUSTERS AND DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_ica__n_components == cancer_ica_best,
                                      param_km__n_components == 4) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      #param_ica__n_components == contra_ica_best,
                                      param_km__n_components == 7) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/RP/cancer cluster KM CLUSTER AND DATA.csv')
  contra_nn <- read.csv('./output/RP/contra cluster KM CLUSTERS AND DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_rp__n_components == cancer_rp_best,
                                      param_km__n_components == 4) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      #param_rp__n_components == contra_rp_best,
                                      param_km__n_components == 6) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/RF/cancer cluster KM CLUSTER AND DATA.csv')
  contra_nn <- read.csv('./output/RF/contra cluster KM CLUSTERS AND DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_filter__n == cancer_rf_best,
                                      param_km__n_components == 4) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      #param_filter__n == contra_rf_best,
                                      param_km__n_components == 5) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
  
  
  ## GMM
  
  cancer_nn <- read.csv('./output/BASE/cancer cluster GMM CLUSTER AND DATA.csv')
  contra_nn <- read.csv('./output/BASE/contra cluster GMM CLUSTERS WITH DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl, 
                                      param_gmm__n_components == 5) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      param_gmm__n_components == 7) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/PCA/cancer cluster GMM CLUSTER AND DATA.csv')
  contra_nn <- read.csv('./output/PCA/contra cluster GMM CLUSTERS WITH DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_pca__n_components == cancer_pca_best,
                                      param_gmm__n_components == 8) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      #param_pca__n_components == contra_pca_best,
                                      param_gmm__n_components == 8) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/ICA/cancer cluster GMM CLUSTER AND DATA.csv')
  contra_nn <- read.csv('./output/ICA/contra cluster GMM CLUSTERS WITH DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_ica__n_components == cancer_ica_best,
                                      param_gmm__n_components == 3) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      #param_ica__n_components == contra_ica_best,
                                      param_gmm__n_components == 9) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/RP/cancer cluster GMM CLUSTER AND DATA.csv')
  contra_nn <- read.csv('./output/RP/contra cluster GMM CLUSTERS WITH DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_rp__n_components == cancer_rp_best,
                                      param_gmm__n_components == 6) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      #param_rp__n_components == contra_rp_best,
                                      param_gmm__n_components == 9) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
  
  cancer_nn <- read.csv('./output/RF/cancer cluster GMM CLUSTER AND DATA.csv')
  contra_nn <- read.csv('./output/RF/contra cluster GMM CLUSTERS WITH DATA.csv')
  
  cancer_best <- cancer_nn %>% filter(param_NN__alpha == cancer_best_alpha, 
                                      param_NN__hidden_layer_sizes == cancer_best_hl,
                                      #param_filter__n == cancer_rf_best,
                                      param_gmm__n_components == 7) %>%
    select(mean_test_score) %>% as.numeric()
  contra_best <- contra_nn %>% filter(param_NN__alpha == contra_best_alpha, 
                                      param_NN__hidden_layer_sizes == contra_best_hl,
                                      #param_filter__n == contra_rf_best,
                                      param_gmm__n_components == 7) %>%
    select(mean_test_score) %>% as.numeric()
  
  cancer_bests <- c(cancer_bests, cancer_best)
  contra_bests <- c(contra_bests, contra_best)
  cancer_overall_bests <- c(cancer_overall_bests, max(cancer_nn$mean_test_score))
  contra_overall_bests <- c(contra_overall_bests, max(contra_nn$mean_test_score))
  
}


{
  axis_names <- merge(datasets, c('KM', 'GMM')) %$% paste0(x, '\n', y)
  jpeg('./output/plots/contra NN data and clust and base - baseline.jpg', height = 240)  
  
  barplot(contra_bests, names.arg = axis_names, 
          cex.names = .8,
          col = c('gray', rep('pink', 4)),
          ylab = 'Average Validation Accuracy',
          main = 'Contraceptive Use - Data + Clusters\nNeural Network Performance with Baseline Parameters')
  
  dev.off()
  
  jpeg('./output/plots/cancer NN data and clust and base - baseline.jpg', height = 240)  
  
  barplot(cancer_bests, names.arg = axis_names, 
          cex.names = .8,
          col = c('gray', rep('pink', 4)),
          ylab = 'Average Validation Accuracy',
          main = 'Breast Cancer - Data + Clusters\nNeural Network Performance with Baseline Parameters')
  
  dev.off()
  
  jpeg('./output/plots/contra NN data and clust and base - optimized.jpg', height = 240)  
  
  barplot(contra_overall_bests, names.arg = axis_names, 
          cex.names = .8,
          col = c('gray', rep('pink', 4)),
          ylab = 'Average Validation Accuracy',
          main = 'Contraceptive Use - Data + Clusters\nNeural Network Performance with Optimized Parameters')
  
  dev.off()
  
  jpeg('./output/plots/cancer NN data and clust and base - optimized.jpg', height = 240)  
  
  barplot(cancer_overall_bests, names.arg = axis_names,
          cex.names = .8,
          col = c('gray', rep('pink', 4)),
          ylab = 'Average Validation Accuracy',
          main = 'Breast Cancer - Data + Clusters\nNeural Network Performance with Optimized Parameters')
  
  dev.off()
}

cancer_bests
contra_bests
cancer_overall_bests
contra_overall_bests