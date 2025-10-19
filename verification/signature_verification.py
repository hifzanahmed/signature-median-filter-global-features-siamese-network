from utils.utilities import Utilities
from features.signature_feature_extraction import SignatureFeatureExtraction
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

        # Step 3: Move model to device and set to eval mode
        model.to(device)
        model.eval()  # ✅ Must be in eval mode for inference

        # Step 4: Convert test feature to tensor
        test_tensor = torch.tensor(test_feature, dtype=torch.float32).unsqueeze(0).to(device)

        # Step 5: Compare with all genuine references
        scores = []
        with torch.no_grad():  # ✅ No gradient tracking during inference
            for i, ref_feature in enumerate(genuine_signatures):
                ref_tensor = torch.tensor(ref_feature, dtype=torch.float32).unsqueeze(0).to(device)
                output = model(ref_tensor, test_tensor)  # Output is a logit
                print(f"Raw model output: {output.item()}")

                # Apply sigmoid since you used BCEWithLogitsLoss during training
                prob = torch.sigmoid(output).item()
                print(f"Sigmoid probability: {prob}")
                scores.append(prob)

        print("Similarity scores:", scores)

        # Step 6: Average the scores
        similarity_score = np.mean(scores)
        print(f"Similarity Score: {similarity_score:.4f}")

        return similarity_score
