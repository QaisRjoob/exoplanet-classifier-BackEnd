"""
Test script for GET /planets/predict endpoint
This endpoint retrieves a saved planet's prediction by name only
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_get_prediction():
    """Test retrieving prediction by planet name"""
    print("=" * 70)
    print("Testing GET /planets/predict endpoint")
    print("=" * 70)
    
    # First, save a planet with prediction
    print("\n1️⃣  First, saving a test planet...")
    save_data = {
        "planet_name": "Test-Kepler-442b",
        "koi_period": 112.3,
        "koi_time0bk": 131.5,
        "koi_impact": 0.2,
        "koi_duration": 5.5,
        "koi_depth": 500.0,
        "koi_prad": 1.34,
        "koi_teq": 233.0,
        "koi_insol": 0.7,
        "koi_steff": 5636.0,
        "koi_slogg": 4.4,
        "koi_srad": 1.1
    }
    
    save_response = requests.post(
        f"{BASE_URL}/planets/predict-and-save",
        json=save_data
    )
    
    if save_response.status_code == 200:
        save_result = save_response.json()
        print(f"✅ Planet saved successfully!")
        print(f"   Planet ID: {save_result['planet_id']}")
        print(f"   Prediction: {save_result['prediction_label']}")
        print(f"   Confidence: {save_result['confidence']:.2%}")
    else:
        print(f"❌ Failed to save planet: {save_response.text}")
        return
    
    # Now retrieve the prediction using GET with planet name
    print("\n2️⃣  Retrieving prediction by planet name...")
    planet_name = "Test-Kepler-442b"
    
    get_response = requests.get(
        f"{BASE_URL}/planets/predict",
        params={"planet_name": planet_name}
    )
    
    print(f"\n📡 Request: GET {BASE_URL}/planets/predict?planet_name={planet_name}")
    print(f"📊 Status Code: {get_response.status_code}")
    
    if get_response.status_code == 200:
        result = get_response.json()
        print(f"\n✅ Prediction Retrieved Successfully!")
        print(f"\n{'=' * 70}")
        print("PREDICTION DETAILS:")
        print(f"{'=' * 70}")
        print(f"Planet Name:     {result['planet_data']['planet_name']}")
        print(f"Planet ID:       {result['planet_id']}")
        print(f"Prediction:      {result['prediction_label']}")
        print(f"Confidence:      {result['confidence']:.2%}")
        print(f"\nClass Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  • {class_name:15s}: {prob:.2%}")
        print(f"\nSaved At:        {result['saved_at']}")
        print(f"{'=' * 70}")
        
        # Show full response in JSON
        print(f"\n📄 Full JSON Response:")
        print(json.dumps(result, indent=2))
    else:
        print(f"\n❌ Failed to retrieve prediction")
        print(f"Error: {get_response.text}")
    
    # Test with non-existent planet
    print("\n\n3️⃣  Testing with non-existent planet...")
    non_existent = "NonExistentPlanet-999"
    
    error_response = requests.get(
        f"{BASE_URL}/planets/predict",
        params={"planet_name": non_existent}
    )
    
    print(f"📡 Request: GET {BASE_URL}/planets/predict?planet_name={non_existent}")
    print(f"📊 Status Code: {error_response.status_code}")
    
    if error_response.status_code == 404:
        print(f"✅ Correctly returned 404 for non-existent planet")
        print(f"   Message: {error_response.json()['detail']}")
    else:
        print(f"❌ Unexpected status code")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║         GET PREDICTION ENDPOINT TEST                                 ║
║         Retrieve saved planet predictions by name only               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

This test demonstrates:
  1. Saving a planet with POST /planets/predict-and-save
  2. Retrieving its prediction with GET /planets/predict?planet_name=...
  3. Error handling for non-existent planets

Make sure the API server is running on http://localhost:8000
""")
    
    try:
        test_get_prediction()
        print("\n\n" + "=" * 70)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nYou can now use this endpoint from your frontend:")
        print("  URL: GET http://localhost:8000/planets/predict")
        print("  Parameter: planet_name (query parameter)")
        print("  Example: /planets/predict?planet_name=Kepler-442b")
        print("=" * 70)
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the API server")
        print("Please make sure the server is running:")
        print("  cd backend")
        print("  uvicorn app:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
