## Cross-Modal Approaches to Beat Tracking: A Case Study on Chopin Mazurkas

### Contents of This Repository
#### Introduction
1. Overview of the system
2. F2E conversion
3. Event-based late fusion
#### Confouding Effects of Conventional Post-Processors
#### Preliminary Results for Training-Based Fusion
#### Code:
1. dataset (ASAP) as example
2. audio beat tracker
        --training
        --inference
3. symbolic beat tracker
        --training
        --inference
4. frame-based fusion
        --addition-based
        --multiplication-based
6. unified pipeline
        --Gaussian smoothing and max-normalization
        --Peak Picking with Local Threshold
8. evaluation (F1, P, R, L-correct)
