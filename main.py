from signature_training import SignatureTraining
from signature_verification import SignatureVerificationTraining 

def main():
    location_of_training_signature = 'data/saba/'
    size_of_training_signature = 6
    location_of_test_signature = 'data/saba/signature7.png'
    # Training Phase
    #s1 = SignatureTraining.training_genuine(location_of_training_signature, size_of_training_signature)
    s1 = SignatureTraining.training_genuine_with_siamese(location_of_training_signature, size_of_training_signature) 
    # Verification Phase of input test signature 
    #s2 = SignatureVerificationTraining.verifiy_test_signature(location_of_test_signature)
    s2 = SignatureVerificationTraining.verify_with_siamese_network(location_of_test_signature, s1 )
    print("S1 (Training Score):", s1)
    print("S2 (Verification Score):", s2)   
    # Decision Making: calculating the score and comparing it with a threshold value
    #score = 0
    #if s1 != 0 and s2 != 0:
    #    score = s1/s2       
    #print("Score (S1/S2):", score)
    #if score==0 or score > 0.73:
    #    print("Signature is Genuine") 
    #else:
    #    print("Signature is Forged") 
    # ==== 5. Test with a New Signature ====

    print("\nTesting on a new signature...") 

    # Simulate a new test signature (replace with real one)

    print(f"Similarity Score: {s2:.4f}")
    if s2 > 0.5:
        print("✅ Likely Genuine")
    else:
        print("❌ Likely Forged")

if __name__ == "__main__":
    main()