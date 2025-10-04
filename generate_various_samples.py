"""
Generate various sample datasets for testing
Creates diverse sample files with different characteristics
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("🎨 GENERATING VARIOUS SAMPLE DATASETS")
print("=" * 80)

# Load the full dataset
print("\n📂 Loading main dataset...")
df = pd.read_csv('cumulative_2025.10.03_00.50.03.csv')
print(f"✅ Loaded {len(df):,} total samples\n")

# Create sample_data directory if it doesn't exist
sample_dir = Path('sample_data')
sample_dir.mkdir(exist_ok=True)

# Counter for generated files
files_created = 0

# ============================================================
# 1. HIGH CONFIDENCE SAMPLES (predictions with high probability)
# ============================================================
print("1️⃣  Creating HIGH CONFIDENCE samples...")
if 'koi_score' in df.columns:
    high_conf = df[df['koi_score'] >= 0.95].sample(
        n=min(30, len(df[df['koi_score'] >= 0.95])), 
        random_state=42
    )
    high_conf.to_csv('sample_data/sample_high_confidence.csv', index=False)
    print(f"   ✅ sample_high_confidence.csv - {len(high_conf)} samples (score >= 0.95)")
    files_created += 1
else:
    print("   ⚠️  Skipped - no koi_score column")
print()

# ============================================================
# 2. EARTH-LIKE PLANETS (similar to Earth size and temperature)
# ============================================================
print("2️⃣  Creating EARTH-LIKE PLANETS...")
earth_like_mask = (
    (df['koi_prad'] >= 0.8) & (df['koi_prad'] <= 1.5) &  # 0.8-1.5 Earth radii
    (df['koi_teq'] >= 200) & (df['koi_teq'] <= 350)       # Habitable zone
)
earth_like = df[earth_like_mask].sample(
    n=min(30, len(df[earth_like_mask])), 
    random_state=42
)
earth_like.to_csv('sample_data/sample_earth_like.csv', index=False)
print(f"   ✅ sample_earth_like.csv - {len(earth_like)} Earth-like exoplanets")
files_created += 1
print()

# ============================================================
# 3. HOT JUPITERS (large planets, short periods, very hot)
# ============================================================
print("3️⃣  Creating HOT JUPITERS...")
hot_jupiter_mask = (
    (df['koi_prad'] >= 8) &        # Jupiter-sized or larger
    (df['koi_period'] <= 10) &     # Very short orbital period
    (df['koi_teq'] >= 1000)        # Very hot
)
hot_jupiters = df[hot_jupiter_mask].sample(
    n=min(25, len(df[hot_jupiter_mask])), 
    random_state=42
)
hot_jupiters.to_csv('sample_data/sample_hot_jupiters.csv', index=False)
print(f"   ✅ sample_hot_jupiters.csv - {len(hot_jupiters)} hot Jupiter planets")
files_created += 1
print()

# ============================================================
# 4. SUPER-EARTHS (1.5-2.5 Earth radii)
# ============================================================
print("4️⃣  Creating SUPER-EARTHS...")
super_earth_mask = (
    (df['koi_prad'] >= 1.5) & (df['koi_prad'] <= 2.5)
)
super_earths = df[super_earth_mask].sample(
    n=min(30, len(df[super_earth_mask])), 
    random_state=42
)
super_earths.to_csv('sample_data/sample_super_earths.csv', index=False)
print(f"   ✅ sample_super_earths.csv - {len(super_earths)} super-Earth planets")
files_created += 1
print()

# ============================================================
# 5. LONG PERIOD PLANETS (outer solar system analogs)
# ============================================================
print("5️⃣  Creating LONG PERIOD planets...")
long_period_mask = df['koi_period'] >= 200  # > 200 days
long_period = df[long_period_mask].sample(
    n=min(30, len(df[long_period_mask])), 
    random_state=42
)
long_period.to_csv('sample_data/sample_long_period.csv', index=False)
print(f"   ✅ sample_long_period.csv - {len(long_period)} long period planets (>200 days)")
files_created += 1
print()

# ============================================================
# 6. SHORT PERIOD PLANETS (very close orbits)
# ============================================================
print("6️⃣  Creating SHORT PERIOD planets...")
short_period_mask = df['koi_period'] <= 2  # < 2 days
short_period = df[short_period_mask].sample(
    n=min(30, len(df[short_period_mask])), 
    random_state=42
)
short_period.to_csv('sample_data/sample_short_period.csv', index=False)
print(f"   ✅ sample_short_period.csv - {len(short_period)} short period planets (<2 days)")
files_created += 1
print()

# ============================================================
# 7. MULTI-CLASS BALANCED (equal representation)
# ============================================================
print("7️⃣  Creating MULTI-CLASS BALANCED dataset...")
samples_per_class = 20
balanced_samples = []

for disposition in ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']:
    class_samples = df[df['koi_disposition'] == disposition].sample(
        n=min(samples_per_class, len(df[df['koi_disposition'] == disposition])),
        random_state=42
    )
    balanced_samples.append(class_samples)

multi_balanced = pd.concat(balanced_samples, ignore_index=True)
multi_balanced = multi_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
multi_balanced.to_csv('sample_data/sample_multi_balanced.csv', index=False)
print(f"   ✅ sample_multi_balanced.csv - {len(multi_balanced)} samples (balanced classes)")
files_created += 1
print()

# ============================================================
# 8. COOL PLANETS (potentially habitable zone)
# ============================================================
print("8️⃣  Creating COOL PLANETS (habitable zone)...")
cool_mask = (df['koi_teq'] >= 175) & (df['koi_teq'] <= 400)  # Habitable temperature range
cool_planets = df[cool_mask].sample(
    n=min(30, len(df[cool_mask])), 
    random_state=42
)
cool_planets.to_csv('sample_data/sample_cool_planets.csv', index=False)
print(f"   ✅ sample_cool_planets.csv - {len(cool_planets)} potentially habitable planets")
files_created += 1
print()

# ============================================================
# 9. MINI-NEPTUNES (2.5-4 Earth radii)
# ============================================================
print("9️⃣  Creating MINI-NEPTUNES...")
mini_neptune_mask = (df['koi_prad'] >= 2.5) & (df['koi_prad'] <= 4)
mini_neptunes = df[mini_neptune_mask].sample(
    n=min(30, len(df[mini_neptune_mask])), 
    random_state=42
)
mini_neptunes.to_csv('sample_data/sample_mini_neptunes.csv', index=False)
print(f"   ✅ sample_mini_neptunes.csv - {len(mini_neptunes)} mini-Neptune planets")
files_created += 1
print()

# ============================================================
# 10. EXTREME CASES (outliers and unusual values)
# ============================================================
print("🔟 Creating EXTREME CASES...")
extreme_samples = []

# Very large planets
large = df[df['koi_prad'] >= 15].sample(n=min(5, len(df[df['koi_prad'] >= 15])), random_state=42)
extreme_samples.append(large)

# Very small planets
small = df[df['koi_prad'] <= 0.5].sample(n=min(5, len(df[df['koi_prad'] <= 0.5])), random_state=42)
extreme_samples.append(small)

# Very hot
hot = df[df['koi_teq'] >= 2000].sample(n=min(5, len(df[df['koi_teq'] >= 2000])), random_state=42)
extreme_samples.append(hot)

# Very long period
very_long = df[df['koi_period'] >= 500].sample(n=min(5, len(df[df['koi_period'] >= 500])), random_state=42)
extreme_samples.append(very_long)

extreme_cases = pd.concat(extreme_samples, ignore_index=True)
extreme_cases.to_csv('sample_data/sample_extreme_cases.csv', index=False)
print(f"   ✅ sample_extreme_cases.csv - {len(extreme_cases)} extreme/outlier cases")
files_created += 1
print()

# ============================================================
# 11. SUN-LIKE STARS (similar stellar parameters to our Sun)
# ============================================================
print("1️⃣1️⃣  Creating SUN-LIKE STAR systems...")
sun_like_mask = (
    (df['koi_steff'] >= 5500) & (df['koi_steff'] <= 6000) &  # Sun's temp ~5778K
    (df['koi_srad'] >= 0.9) & (df['koi_srad'] <= 1.1)        # Sun-like radius
)
sun_like = df[sun_like_mask].sample(
    n=min(30, len(df[sun_like_mask])), 
    random_state=42
)
sun_like.to_csv('sample_data/sample_sun_like_stars.csv', index=False)
print(f"   ✅ sample_sun_like_stars.csv - {len(sun_like)} planets around Sun-like stars")
files_created += 1
print()

# ============================================================
# 12. TRANSIT DEPTH VARIETY (different transit depths)
# ============================================================
print("1️⃣2️⃣  Creating TRANSIT DEPTH variety...")
transit_samples = []

# Shallow transits
shallow = df[df['koi_depth'] <= 100].sample(n=min(10, len(df[df['koi_depth'] <= 100])), random_state=42)
transit_samples.append(shallow)

# Medium transits
medium = df[(df['koi_depth'] > 100) & (df['koi_depth'] <= 1000)].sample(
    n=min(10, len(df[(df['koi_depth'] > 100) & (df['koi_depth'] <= 1000)])), 
    random_state=42
)
transit_samples.append(medium)

# Deep transits
deep = df[df['koi_depth'] > 1000].sample(n=min(10, len(df[df['koi_depth'] > 1000])), random_state=42)
transit_samples.append(deep)

transit_variety = pd.concat(transit_samples, ignore_index=True)
transit_variety.to_csv('sample_data/sample_transit_variety.csv', index=False)
print(f"   ✅ sample_transit_variety.csv - {len(transit_variety)} samples with varied transit depths")
files_created += 1
print()

# ============================================================
# 13. RANDOM DIVERSE MIX (stratified sampling)
# ============================================================
print("1️⃣3️⃣  Creating RANDOM DIVERSE mix...")
random_diverse = df.sample(n=min(50, len(df)), random_state=42)
random_diverse.to_csv('sample_data/sample_random_diverse.csv', index=False)
print(f"   ✅ sample_random_diverse.csv - {len(random_diverse)} randomly sampled planets")
files_created += 1
print()

# ============================================================
# SUMMARY
# ============================================================
print("=" * 80)
print("✅ GENERATION COMPLETE!")
print("=" * 80)
print(f"\n📊 Total files created: {files_created}")
print(f"📁 Location: sample_data/")
print("\n📋 Files created:")
print("   1. sample_high_confidence.csv - High prediction confidence")
print("   2. sample_earth_like.csv - Earth-like planets")
print("   3. sample_hot_jupiters.csv - Hot Jupiter planets")
print("   4. sample_super_earths.csv - Super-Earth planets")
print("   5. sample_long_period.csv - Long orbital periods")
print("   6. sample_short_period.csv - Short orbital periods")
print("   7. sample_multi_balanced.csv - Balanced classes")
print("   8. sample_cool_planets.csv - Habitable zone temperatures")
print("   9. sample_mini_neptunes.csv - Mini-Neptune sized")
print("   10. sample_extreme_cases.csv - Outliers and extreme values")
print("   11. sample_sun_like_stars.csv - Around Sun-like stars")
print("   12. sample_transit_variety.csv - Varied transit depths")
print("   13. sample_random_diverse.csv - Random diverse mix")
print("\n🎉 All sample datasets ready for testing!")
print("=" * 80)
