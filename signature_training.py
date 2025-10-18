from utilities import Utilities
from signature_feature_extraction import SignatureFeatureExtraction
from siamese_network import SiameseNetwork, SignatureDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import config


class SignatureTraining:
    
    def training_genuine_with_siamese(location, trainingSize):
        print("Training on Genuine Signatures...")
        featureList = []  # list to hold features 
        for i in range(1, trainingSize + 1):
            feature = SignatureFeatureExtraction.preprocess_and_feature_extraction_radon_transform_features(f'{location}/signature{i}.png')
            featureList.append(feature)

        config.global_features = featureList  # Save for later use

        feature_length = featureList[0].shape[0]
        dataset = SignatureDataset(featureList)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        # ==== Train Siamese Model ====
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SiameseNetwork(input_size=feature_length).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()  # or BCEWithLogitsLoss() if you modify the model

        for epoch in range(10):
            total_loss = 0.0
            model.train()
            for x1, x2, label in dataloader:
                x1, x2, label = x1.to(device), x2.to(device), label.to(device).unsqueeze(1)

                output = model(x1, x2)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

            # ✅ Return the trained model instead of dataloader
        return model
    
    




