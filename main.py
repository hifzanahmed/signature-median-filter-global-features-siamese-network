from training.signature_training import SignatureTraining
from verification.signature_verification import SignatureVerificationTraining 

def main():
    location_of_training_signature = 'data/SA/'
    size_of_training_signature = 6
    location_of_test_signature = 'data/SA/signature7.png'
    # Training Phase
    print("🚀 Training Started...")
    model = SignatureTraining.training_genuine_with_siamese(location_of_training_signature, size_of_training_signature) 
    # Verification Phase of input test signature 
    print("\n🔍 Verifying Test Signature...")
    similarity  = SignatureVerificationTraining.verify_test_signature_with_siamese_network(location_of_test_signature, model)
    print(f"Similarity Score: {similarity :.4f}")
     
    if similarity > 0.5:
        print("✅ Likely Genuine")
    else:
        print("❌ Likely Forged")



if __name__ == "__main__":
    main()