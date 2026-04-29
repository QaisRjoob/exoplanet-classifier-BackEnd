"""
Test the new Predict-and-Save Planet Endpoint
==============================================
This endpoint allows you to:
1. Enter planet data in a simple form
2. Get a prediction
3. Save it to "My Planets" automatically
"""

import requests
import json

API_URL = "http://localhost:8000"

print("=" * 70)
print("TESTING: Predict-and-Save Planet Endpoint")
print("=" * 70)

# Example 1: Minimal planet data
print("\n[Test 1] Minimal planet data (just key features)")
print("-" * 70)

minimal_planet = {
    "planet_name": "Quick Test Planet",
    "koi_period": 10.5,
    "koi_depth": 500.0,
    "koi_prad": 2.5,
    "koi_teq": 400.0,
    "koi_steff": 5800.0,
    "koi_srad": 1.0,
    "notes": "Testing minimal input"
}

try:
    response = requests.post(
        f"{API_URL}/planets/predict-and-save",
        json=minimal_planet
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success!")
        print(f"\n📝 Planet saved:")
        print(f"   ID: {result['planet_id']}")
        print(f"   Name: {result['planet_data']['planet_name']}")
        print(f"\n🔮 Prediction:")
        print(f"   Classification: {result['prediction_label']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"\n📊 Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"   {label}: {prob:.2%}")
        
        saved_planet_id = result['planet_id']
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        saved_planet_id = None

except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to API!")
    print("Make sure the server is running:")
    print("  cd backend")
    print("  python app.py")
    saved_planet_id = None


# Example 2: Comprehensive planet data
print("\n\n[Test 2] Comprehensive planet data")
print("-" * 70)

comprehensive_planet = {
    "planet_name": "Super Earth Candidate",
    "koi_period": 25.3,
    "koi_depth": 800.0,
    "koi_prad": 3.2,
    "koi_teq": 350.0,
    "koi_insol": 5.5,
    "koi_model_snr": 18.5,
    "koi_steff": 5500.0,
    "koi_srad": 0.95,
    "koi_smass": 0.9,
    "koi_impact": 0.3,
    "koi_duration": 4.5,
    "koi_kepmag": 13.8,
    "notes": "Interesting super-Earth candidate in habitable zone"
}

try:
    response = requests.post(
        f"{API_URL}/planets/predict-and-save",
        json=comprehensive_planet
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success!")
        print(f"\n📝 Planet saved:")
        print(f"   ID: {result['planet_id']}")
        print(f"   Name: {result['planet_data']['planet_name']}")
        print(f"\n🔮 Prediction:")
        print(f"   Classification: {result['prediction_label']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        
        if result['prediction_label'] == "CONFIRMED":
            print(f"   🎉 This looks like a confirmed exoplanet!")
        elif result['prediction_label'] == "CANDIDATE":
            print(f"   🔍 This is a promising candidate for follow-up")
        else:
            print(f"   ⚠️  This might be a false positive")
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Error: {e}")


# Example 3: Retrieve saved planet from "My Planets"
if saved_planet_id:
    print("\n\n[Test 3] Retrieve saved planet from 'My Planets'")
    print("-" * 70)
    
    try:
        response = requests.get(f"{API_URL}/planets/{saved_planet_id}")
        
        if response.status_code == 200:
            result = response.json()
            planet = result['data']
            print(f"✅ Retrieved planet:")
            print(f"   Name: {planet.get('planet_name')}")
            print(f"   Classification: {planet.get('disposition')}")
            print(f"   Saved at: {planet.get('submitted_at')}")
            print(f"\n✅ Planet is now in your 'My Planets' collection!")
        else:
            print(f"❌ Error: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


# Example 4: List all saved planets
print("\n\n[Test 4] List all planets in 'My Planets'")
print("-" * 70)

try:
    response = requests.get(f"{API_URL}/planets/list")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Total planets in collection: {result['total_count']}")
        
        if result['total_count'] > 0:
            print(f"\n📋 Recent planets:")
            for i, planet in enumerate(result['planets'][:5], 1):
                print(f"   {i}. {planet.get('planet_name', 'Unnamed')} - {planet.get('disposition', 'Unknown')}")
        
    else:
        print(f"❌ Error: {response.status_code}")

except Exception as e:
    print(f"❌ Error: {e}")


print("\n" + "=" * 70)
print("TESTING COMPLETE!")
print("=" * 70)

print("\n💡 How to use this in your app:")
print("""
1. Create a form with input fields for planet features
2. Send POST request to: /planets/predict-and-save
3. Display the prediction result to the user
4. The planet is automatically saved to "My Planets"
5. User can view it in their planet collection

Example JavaScript/Fetch:
--------------------------
const planetData = {
    planet_name: "My Discovery",
    koi_period: 10.5,
    koi_depth: 500,
    koi_prad: 2.5,
    koi_steff: 5800
};

fetch('http://localhost:8000/planets/predict-and-save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(planetData)
})
.then(res => res.json())
.then(data => {
    console.log('Prediction:', data.prediction_label);
    console.log('Saved as:', data.planet_id);
});
""")
