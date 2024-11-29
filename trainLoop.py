# Imports
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator
from torch.utils.data import DataLoader, TensorDataset

def ganLossFunction(generator, discriminator, realData, fakeData, device):
    '''
    Custom loss function for GAN training
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        realData: Real data samples
        fakeData: Generated fake data samples
        device: Training device
    
    Returns:
        Tuple of generator and discriminator losses
    '''
    # Generate fake data
    batchSize = realData.size(0)
    noise = torch.randn(batchSize, generator.latentZDim, device = device)
    fakeSamples = generator(noise)
    
    # Discriminator forward pass
    realLabels = torch.ones(batchSize, 1, device = device)
    fakeLabels = torch.zeros(batchSize, 1, device = device)
    
    realOutput = discriminator(realData)
    fakeOutput = discriminator(fakeSamples)
    
    # Discriminator loss
    realLoss = nn.BCEWithLogitsLoss()(realOutput, realLabels)
    fakeLoss = nn.BCEWithLogitsLoss()(fakeOutput, fakeLabels)
    discLoss = (realLoss + fakeLoss) / 2
    
    # Generator loss
    genLabels = torch.ones(batchSize, 1, device = device)
    genOutput = discriminator(fakeSamples)
    genLoss = nn.BCEWithLogitsLoss()(genOutput, genLabels)
    
    return genLoss, discLoss

def trainGan(generator, discriminator, trainData, numEpochs = 100, batchSize = 64, lr = 0.0002):
    '''
    Main training function for GAN
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        trainData: Training data tensor
        numEpochs: Number of training epochs
        batchSize: Batch size for training
        lr: Learning rate
    '''
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create DataLoader
    dataset = TensorDataset(trainData)
    trainLoader = DataLoader(dataset, batch_size = batchSize, shuffle = True)
    
    # Optimizers
    genOptimizer = optim.Adam(generator.parameters(), lr = lr, betas = (0.5, 0.999))
    discOptimizer = optim.Adam(discriminator.parameters(), lr = lr, betas = (0.5, 0.999))
    
    # Move models to device
    generator.to(device)
    discriminator.to(device)
    
    # Training loop
    for epoch in range(numEpochs):
        generator.train()
        discriminator.train()
        
        totalGenLoss = 0
        totalDiscLoss = 0
        
        for batch in trainLoader:
            realData = batch[0].to(device)
            
            # Update Discriminator
            discOptimizer.zero_grad()
            
            # Generate noise
            noise = torch.randn(realData.size(0), generator.latentZDim, device = device)
            
            # Compute GAN losses
            genLoss, discLoss = ganLossFunction(generator, discriminator, realData, noise, device)
            
            # Backpropagate Discriminator loss
            discLoss.backward()
            discOptimizer.step()
            
            # Update Generator
            genOptimizer.zero_grad()
            
            # Regenerate noise (to ensure fresh samples)
            noise = torch.randn(realData.size(0), generator.latentZDim, device = device)
            
            # Generate fake samples
            fakeSamples = generator(noise)
            
            # Compute generator loss
            genLoss, _ = ganLossFunction(generator, discriminator, realData, noise, device)
            genLoss.backward()
            genOptimizer.step()
            
            totalGenLoss += genLoss.item()
            totalDiscLoss += discLoss.item()
        
        # Print epoch statistics
        print(f'Epoch [{epoch+1}/{numEpochs}]')
        print(f'Discriminator Loss: {totalDiscLoss / len(trainLoader):.4f}')
        print(f'Generator Loss: {totalGenLoss / len(trainLoader):.4f}')
        
        # Optional: Save models periodically
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f'generator_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch+1}.pth')

def train():
    # Initialize models
    generator = Generator()
    discriminator = Discriminator()
    
    trainData = torch.randn(1000, 3, 150, 150)  # Example placeholder data
    
    # Train the GAN
    trainGan(generator, discriminator, trainData)

train()
