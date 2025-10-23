from training.signature_training import SignatureTraining
from verification.signature_verification import SignatureVerificationTraining 

def main():
    location_genuine_training_signature = 'data/SA/'
    size_of_location_genuine_training_signature = 6
    location_forged_training_signature = 'data/SH/'
    size_of_location_forged_training_signature = 5
    location_of_test_signature = 'data/SA/Forged/signature1.png'
    # Training Phase
    print("🚀 Training Started...")
    model = SignatureTraining.training_with_genuine_and_forged_with_drt_siamese(
        location_genuine_training_signature, location_forged_training_signature, size_of_location_genuine_training_signature,
        size_of_location_forged_training_signature) 
    #model = SignatureTraining.training_with_genuine_and_forged_with_global_siamese(
    #    location_genuine_training_signature, location_forged_training_signature, size_of_location_genuine_training_signature,
    #    size_of_location_forged_training_signature) 
    # Verification Phase of input test signature 
    print("\n🔍 Verifying Test Signature...")
    similarity  = SignatureVerificationTraining.verify_test_signature_with_drt_siamese_network(location_of_test_signature, model)
    #similarity  = SignatureVerificationTraining.verify_test_signature_with_global_siamese_network(location_of_test_signature, model)
     
    THRESHOLD = 0.65  # 🔧 Tune based on validation results
    if similarity > THRESHOLD:
        print("✅ Likely Genuine")
    else:
        print("❌ Likely Forged")




if __name__ == "__main__":
    main()