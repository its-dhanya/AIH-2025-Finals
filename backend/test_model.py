import torch
from sentence_transformers import SentenceTransformer
import sys

print(f"--- Starting Model Test ---")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

try:
    # --- Step 1: Force CPU ---
    print("\n[Step 1] Forcing PyTorch to use CPU...")
    torch.set_default_device('cpu')
    print("✅ PyTorch default device set to 'cpu'.")

    # --- Step 2: Load the Model ---
    model_name = 'sentence-transformers/msmarco-MiniLM-L-6-v3'
    print(f"\n[Step 2] Loading model: '{model_name}'...")
    model = SentenceTransformer(model_name, device='cpu')
    print("✅ Model loaded successfully.")

    # --- Step 3: Run the Encode Function ---
    test_sentence = ['This is a test sentence.']
    print(f"\n[Step 3] Encoding test sentence: {test_sentence}...")
    embedding = model.encode(test_sentence)
    print("✅ SUCCESS! The model's encode() function worked correctly.")
    print(f"Output embedding shape: {embedding.shape}")

except Exception as e:
    print(f"\n🔥🔥🔥 TEST FAILED 🔥🔥🔥")
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Model Test Finished ---")