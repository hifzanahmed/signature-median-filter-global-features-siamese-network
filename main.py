from training.signature_training import SignatureTraining
from verification.signature_verification import SignatureVerificationTraining 

def main():
    location_of_training_signature = 'data/saba/'
    size_of_training_signature = 6
    location_of_test_signature = 'data/saba/signature7.png'
    # Training Phase
    s1 = SignatureTraining.training_genuine_with_siamese(location_of_training_signature, size_of_training_signature) 
    # Verification Phase of input test signature 
    s2 = SignatureVerificationTraining.verify_with_siamese_network(location_of_test_signature, s1 )
    print("S1 (Training Score):", s1)
    print("S2 (Verification Score):", s2)   
    print(f"Similarity Score: {s2:.4f}")
    
    if s2 > 0.5:
        print("✅ Likely Genuine")
    else:
        print("❌ Likely Forged")



if __name__ == "__main__":
    main()