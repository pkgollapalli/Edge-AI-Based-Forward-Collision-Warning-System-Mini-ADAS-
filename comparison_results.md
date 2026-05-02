# Domain Shift Fix via Fine-tuning on DriveIndia

| Class | COCO baseline (recall) | Fine-tuned (recall) | Improvement |
|---|---|---|---|
| autorickshaw | 0% | 85.0% | +85.0% |
| bicycle | 2.7% | 97.5% | +94.8% |
| police_vehicle | 0% | 91.3% | +91.3% |
| commercial_vehicle | 0% | 47.4% | +47.4% |
| ambulance | 0% | 62.5% | +62.5% |
| zebra_crossing | 0% | 90.3% | +90.3% |
| pothole | 4.8% | 33.3% | +28.5% |
| person | 94.3% | 94.2% | ≈ same |
| car | 80.6% | 97.3% | +16.7% |
| motorcycle | 14.1% | 83.5% | +69.4% |
| truck | 81.6% | 69.5% | -12.1% |
| bus | 70.6% | 62.8% | -7.8% |

Validated on: 2500 DriveIndia val images
Latency on laptop CPU: 50 ms (fine-tuned) vs 172 ms (baseline)
