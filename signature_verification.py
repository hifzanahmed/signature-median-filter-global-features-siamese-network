from utilities import Utilities
from signature_feature_extraction import SignatureFeatureExtraction
import config
import numpy as np
import torch

class SignatureVerificationTraining:
        
    def verify_test_signature_with_siamese_network(test_image_path, model):
        # Step 1: Extract features from test image
        test_feature = SignatureFeatureExtraction.preprocess_and_feature_extraction_radon_transform_features(test_image_path)

        # Step 2: Load known genuine signature features
        genuine_signatures = config.global_features  # should be a list of feature vectors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Step 3: Move model to device
        model.to(device)
        model.eval()

        # Step 4: Convert test feature to tensor
        test = torch.tensor(test_feature, dtype=torch.float32).unsqueeze(0).to(device)

        # Step 5: Compare with all genuine references
        scores = []
        with torch.no_grad():
            for ref_feature in genuine_signatures:
                ref = torch.tensor(ref_feature, dtype=torch.float32).unsqueeze(0).to(device)
                output = model(ref, test)  # If using BCEWithLogitsLoss, wrap with sigmoid
                prob = torch.sigmoid(output).item()  # if output is a logit
                scores.append(prob)

        # Step 6: Average the scores
        avg_similarity = sum(scores) / len(scores)
        return avg_similarity