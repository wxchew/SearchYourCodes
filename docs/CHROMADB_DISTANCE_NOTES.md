# ChromaDB Distance Metrics - Important Notes

## Key Finding: ChromaDB Default Distance Function

**ChromaDB uses SQUARED L2 distance by default when no distance function is specified.**

### Empirical Test Results (August 5, 2025)

Tested with orthogonal unit vectors: `vec1 = [1,0,0]` and `vec2 = [0,1,0]`

| Distance Function | ChromaDB Result | Expected Value | Notes |
|-------------------|----------------|----------------|-------|
| **Default** (no specification) | **2.0** | **2.0** | ✅ **Squared L2 distance** |
| Cosine (`hnsw:space: cosine`) | 1.0 | 1.0 | ✅ Cosine distance |
| L2 (`hnsw:space: l2`) | 2.0 | 2.0 | ✅ Squared L2 distance |
| Inner Product (`hnsw:space: ip`) | 1.0 | 1.0 | ✅ Inner product distance |

### Mathematical Formula for Squared L2 Distance

For normalized vectors (||a|| = ||b|| = 1):
- Squared L2 distance: `d = ||a - b||² = ||a||² + ||b||² - 2⋅dot(a,b) = 2 - 2⋅dot(a,b)`
- Therefore: `dot(a,b) = 1 - d/2`
- Similarity conversion: `similarity = 1 - distance/2` ✅

### Our Collection Settings

Our collections in this project were created using:
```python
client.create_collection(
    name="unixcoder_snippets",
    metadata={"description": "...", "model": "..."}
    # No hnsw:space specified = defaults to Squared L2
)
```

This means they use **Squared L2 distance by default**.

### Common Confusion

Many people assume ChromaDB uses cosine distance by default because:
1. Cosine distance is common in embedding applications
2. The conversion formula looks similar
3. Documentation doesn't always make this clear

**Always specify the distance function explicitly** to avoid confusion:
```python
# For cosine distance (recommended for normalized embeddings)
client.create_collection(name="my_collection", metadata={"hnsw:space": "cosine"})

# For L2 distance  
client.create_collection(name="my_collection", metadata={"hnsw:space": "l2"})
```

### Testing Method

To verify distance function empirically:
1. Create a test collection
2. Add two orthogonal unit vectors: `[1,0,0]` and `[0,1,0]`
3. Query with first vector
4. Check distance to second vector:
   - **2.0** = Squared L2 distance
   - **1.0** = Cosine distance  
   - **√2 ≈ 1.414** = Regular L2 distance

### ChromaDB Version Tested
- ChromaDB version: 1.0.15
- Date: August 5, 2025
- Python environment: 3.13

---

**⚠️ Important:** Always verify distance function behavior when upgrading ChromaDB versions, as defaults might change.
