# Imports
import time
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Literal, Callable

# Move function
def moveTo(data, device):
    """Move data to specified device."""
    if isinstance(data, tuple):
        return tuple(d.to(device) if hasattr(d, 'to') else d for d in data)
    return data.to(device)

def trainLoop(model: nn.Module, trainLoader: DataLoader, lossFunc, optimizer, scoreFuncs: Dict, epoch: int, device: Literal['cpu', 'cuda'] = 'cpu'):
    '''
    Training loop! 
    
    Args:
        model: PyTorch model to train
        trainLoader: DataLoader for training data
        lossFunc: Loss function
        optimizer: Optimizer
        scoreFuncs: Dictionary of scoring functions
        epoch: Current epoch number
        device: Device to run training on
    
    Returns:
        pd.DataFrame: Results of training metrics
    '''
    model.train()
    start = time.time()
    
    # Preallocate lists for efficiency
    runningLoss = []
    yTrue = []
    yPred = []
    
    for inputs, labels in tqdm(trainLoader, desc = 'Train Batch', leave = False):
        # Move data to device
        inputs, labels = moveTo((inputs, labels), device)
        
        # Zero gradients before forward pass
        optimizer.zero_grad()
        
        # Forward pass
        yHat = model(inputs)
        
        # Compute loss
        loss = lossFunc(yHat, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Collect predictions and ground truth
        yTrue.extend(labels.cpu().numpy().tolist())
        yPred.extend(yHat.detach().cpu().numpy().tolist())
        runningLoss.append(loss.item())
    
    # Process predictions
    yPred = np.asarray(yPred)
    
    # Compute metrics
    metrics = {'Train Loss': np.mean(runningLoss)}
    for name, scoreFunc in scoreFuncs.items():
        metrics['Train ' + name] = scoreFunc(yTrue, yPred)
    
    end = time.time()
    print(f'Train Epoch {epoch + 1}: {end - start:.2f}s')
    
    return pd.DataFrame([metrics])

def testLoop(model: nn.Module, testLoader: DataLoader, lossFunc, scoreFuncs: Dict, epoch: int, device: Literal['cpu', 'cuda'] = 'cpu'):
    '''
    Validation/Test! 
    
    Args:
        model: PyTorch model to evaluate
        testLoader: DataLoader for test/validation data
        lossFunc: Loss function
        scoreFuncs: Dictionary of scoring functions
        epoch: Current epoch number
        device: Device to run evaluation on
    
    Returns:
        pd.DataFrame: Results of test metrics
    '''
    model.eval()
    start = time.time()
    
    # Preallocate lists for efficiency
    runningLoss = []
    yTrue = []
    yPred = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(testLoader, desc = 'Test Batch', leave = False):
            # Move data to device
            inputs, labels = moveTo((inputs, labels), device)
            
            # Forward pass
            yHat = model(inputs)
            
            # Compute loss
            loss = lossFunc(yHat, labels)
            
            # Collect predictions and ground truth
            yTrue.extend(labels.cpu().numpy().tolist())
            yPred.extend(yHat.cpu().numpy().tolist())
            runningLoss.append(loss.item())
    
    # Process predictions
    yPred = np.asarray(yPred)
    
    # Compute metrics
    metrics = {'Test Loss': np.mean(runningLoss)}
    for name, scoreFunc in scoreFuncs.items():
        metrics['Test ' + name] = scoreFunc(yTrue, yPred)
    
    end = time.time()
    print(f'Test Epoch {epoch + 1}: {end - start:.2f}s')
    
    return pd.DataFrame([metrics])

def runEpoch(model: torch.nn.Module, trainLoader: torch.utils.data.DataLoader, testLoader: torch.utils.data.DataLoader, 
              lossFunc: Callable, optimizer: torch.optim.Optimizer, scoreFuncs: Dict[str, Callable], 
              numEpochs: int, device: str, resultsDir: str = 'results'):
    '''
    Run training and testing for specified number of epochs.
    
    Args:
        model: PyTorch model to train and test
        trainLoader: DataLoader for training data
        testLoader: DataLoader for test data
        lossFunc: Loss function
        optimizer: Optimizer
        scoreFuncs: Dictionary of scoring functions
        numEpochs: Number of epochs to train
        device: Device to run training on
        resultsDir: Directory to save results
    
    Returns:
        Tuple of DataFrames containing train and test results
    '''
    # Ensure results directory exists
    os.makedirs(resultsDir, exist_ok = True)
    
    # Initialize lists to collect results across epochs
    trainResultsList = []
    testResultsList = []
    
    # Move model to specified device
    model.to(device)
    
    # Training loop
    for epoch in tqdm(range(numEpochs), desc = "Training Epochs"):
        # Perform train and test loops
        trainResult = trainLoop(
            model = model, 
            trainLoader = trainLoader, 
            lossFunc = lossFunc, 
            optimizer = optimizer, 
            scoreFuncs = scoreFuncs,
            epoch = epoch, 
            device = device
        )
        
        testResult = testLoop(
            model = model, 
            testLoader = testLoader, 
            lossFunc = lossFunc, 
            scoreFuncs = scoreFuncs,
            epoch = epoch, 
            device = device
        )
        
        trainResultsList.append(trainResult)
        testResultsList.append(testResult)
    
    # Concatenate results across epochs
    trainResultsDf = pd.concat(trainResultsList, ignore_index = True)
    testResultsDf = pd.concat(testResultsList, ignore_index = True)
    
    # Save results to CSV
    trainResultsPath = os.path.join(resultsDir, 'train_results.csv')
    testResultsPath = os.path.join(resultsDir, 'test_results.csv')
    
    trainResultsDf.to_csv(trainResultsPath, index = False)
    testResultsDf.to_csv(testResultsPath, index = False)
    
    # Create a combined results DataFrame
    combinedResultsDf = pd.DataFrame({
        'Epoch': range(numEpochs),
        **{f'Train {col}': trainResultsDf[col] for col in trainResultsDf.columns},
        **{f'Test {col}': testResultsDf[col] for col in testResultsDf.columns}
    })
    combinedResultsPath = os.path.join(resultsDir, 'combined_results.csv')
    combinedResultsDf.to_csv(combinedResultsPath, index = False)
    
    # Plotting 
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize = (12, 5))
        
        # Plot train and test losses
        plt.subplot(1, 2, 1)
        plt.plot(trainResultsDf['Train Loss'], label = 'Train Loss')
        plt.plot(testResultsDf['Test Loss'], label = 'Test Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot other metrics (assuming first metric after loss)
        otherMetrics = [col for col in trainResultsDf.columns if col != 'Train Loss']
        if otherMetrics:
            plt.subplot(1, 2, 2)
            for metric in otherMetrics:
                plt.plot(trainResultsDf[metric], label = f'Train {metric}')
                plt.plot(testResultsDf[metric.replace('Train', 'Test')], label = f'Test {metric}')
            plt.title('Metrics Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Metric Value')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(resultsDir, 'training_metrics.png'))
        plt.close()
    except ImportError:
        print('Matplotlib not available. Skipping plotting.')
    
    return trainResultsDf, testResultsDf, combinedResultsDf
