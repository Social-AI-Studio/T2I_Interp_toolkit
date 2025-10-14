# Quick check: Is analyze_ethnicity_distribution using correct aggregation?
# Run this in notebook

import inspect

print("="*80)
print("CHECKING ANALYZE_ETHNICITY_DISTRIBUTION")
print("="*80)

# Get source code
source = inspect.getsource(analyze_ethnicity_distribution)

# Check for aggregation method
if "aggregate_spatial='flatten'" in source:
    print("✓ CORRECT: Uses aggregate_spatial='flatten'")
elif "aggregate_spatial='mean'" in source:
    print("❌ BUG: Uses aggregate_spatial='mean' (should be 'flatten')")
elif "aggregate_spatial='all_positions'" in source:
    print("⚠️  PARTIAL: Uses 'all_positions' (should be 'flatten' for this approach)")
else:
    print("? Could not determine aggregation method")

print("\nRelevant lines:")
for i, line in enumerate(source.split('\n')):
    if 'aggregate_spatial' in line:
        print(f"  Line {i}: {line.strip()}")

print("\n" + "="*80)

