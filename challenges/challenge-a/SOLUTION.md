# Solution — Challenge A: Physical AI & Robotics

## Your Name
Sanidhya Sharma

 ## Email
 sanidhya.sharma@mitwpu.edu.in

## Approach

- Pipeline: load CSV → bandpass filter + normalize → segment into overlapping 200ms windows → extract 16 features per window (4 channels × 4 features) → train Random Forest classifier
- Dataset: 50,000 rows @ ~1000Hz (confirmed via mean timestamp delta ≈ 0.001s), 5 equal 10,000-row blocks — one per grip (`rest`, `power_grip`, `pinch`, `lateral`, `point`), 10 seconds each

### Signal Preprocessing

- **Filter:** 4th-order Butterworth bandpass, 20–450Hz (`scipy.signal.butter` + `filtfilt`)
  - 20–450Hz range removes motion artifact (below) and high-freq noise (above); real EMG signal lives in between
  - `filtfilt` chosen over `lfilter` for zero-phase filtering — avoids shifting the signal in time, important since window boundaries depend on exact sample positions
  - Order 4 balances sharp cutoff vs. ringing/instability risk of higher orders — standard convention in EMG literature
- **Normalization:** z-score (mean=0, std=1), computed per channel, globally across the signal
  - Rejected min-max scaling — too sensitive to single outliers stretching the whole scale
  - Verified outliers (>3 std) cluster almost entirely in the `power_grip` block, not randomly scattered → confirms they're genuine strong contractions, not noise, validating z-score over `RobustScaler` (see `extras/outlier_detection_ch1.png`)
  - **Known limitation:** global normalization isn't realistic for true real-time deployment (would need per-window/rolling stats instead)

### Feature Extraction

- 4 features per channel × 4 channels = 16 features per window:
  - **RMS** — overall signal magnitude, primary contraction-strength indicator
  - **MAV** — simpler magnitude measure
  - **ZCR** — zero-crossing frequency, captures jitter/complexity
  - **WL** — cumulative sample-to-sample change, captures amplitude + rate of change
- Chosen to jointly capture "how strong" (RMS, MAV) and "how jittery" (ZCR, WL)
- Sanity check: avg ch1 RMS per grip — `rest=0.10 → point=0.58 → lateral=0.96 → pinch=1.16 → power_grip=1.53` — clean monotonic separation, confirms features carry real signal

### Segmentation

- Fixed sliding windows: 200 samples (200ms) size, 50-sample (50ms) overlap → 150-sample step
- Standard EMG convention: stable feature estimates + near-real-time responsiveness
- Overlap yields more windows than non-overlapping (333 vs. 250) and smoother predictions
- Labels assigned via majority vote across samples in each window (handles transition-boundary windows)

### Classification

- **Model:** Random Forest (`sklearn.ensemble.RandomForestClassifier`, 100 trees) — handles tabular sensor features well with minimal tuning, common EMG-classification default
- **Split:** 80/20 train/test, stratified (5 balanced classes)
- **Result: 98.51% test accuracy** (target was ≥70%)

### Challenges & Trade-offs

- **Boundary-window ambiguity:** exactly one window per label transition (4 total, out of 333 windows) lands almost exactly 50/50 across two grip labels (verified: window 66 spans samples 9900–10099, straddling the `rest`/`power_grip` boundary at row 10,000 with an exact 100/100 split). These are inherently ambiguous, real-world-realistic edge cases rather than errors — an onset/activity-detection based segmentation approach could avoid this, at the cost of significantly more implementation complexity, and would be a natural next improvement.
- **Simulated vs. real data:** this dataset shows very clean separation between grips (e.g. the 15x RMS difference between `rest` and `power_grip`). Real EMG from an actual arm would likely show more overlap between similar grips (e.g. `pinch` vs `lateral`, both involving thumb+index), so I'd expect accuracy to be meaningfully lower on real sensor data than on this simulated set.
- **Global vs. real-time-realistic normalization**, discussed above.

## How to Run

```bash
cd challenge-a
pip install -r requirements.txt
python src/placeholder.py
```

This loads `data/emg_signals.csv`, trains the classifier, prints training accuracy, then loads `data/test_signals.csv` and prints a predicted grip label for every window.

## Results

- **Test accuracy: 98.51%** (80/20 stratified train/test split on the labeled data)
- 333 total windows extracted from the 50,000-row training signal (~66-67 per grip type)
- Predictions on `test_signals.csv` (10,000 rows → 66 windows) show clean, consistent blocks matching the same grip-sequence pattern as training data, with no flickering between adjacent windows

## Bonus (if applicable)

- **Outlier investigation:** checked for outliers (values beyond 3 standard deviations) per channel to validate the normalization choice
  - Found outliers cluster almost entirely within the `power_grip` block rather than being randomly scattered — see `extras/outlier_detection_ch1.png`
  - Confirmed they represent genuine strong contractions rather than sensor artifacts, supporting z-score over a more conservative scaler

- **Classifier comparison:** trained three different classifiers on the exact same train/test split (80/20, stratified, `random_state=42`) to check whether model choice actually mattered
  - Random Forest (100 trees): **98.51%**
  - SVM (RBF kernel): **98.51%**
  - Logistic Regression: **98.51%**
  - All three scored identically — a meaningful result, not a coincidence
  - Average channel-1 RMS across grips: `rest=0.10 → point=0.58 → lateral=0.96 → pinch=1.16 → power_grip=1.53` — clean, non-overlapping progression, ~15x gap between weakest and strongest grip
  - When classes are this well-separated in feature space, even a simple linear model (Logistic Regression) draws an accurate boundary just as well as an ensemble (Random Forest) or kernel method (SVM)
  - **Takeaway:** the real driver of accuracy was Task 1/2 (filtering + feature extraction), not classifier sophistication in Task 3
  - Kept Random Forest as the submitted model — fast, constant-time prediction (relevant to the low-latency, embedded-systems context this challenge emphasizes), no tuning required
  - Logistic Regression is a reasonable lighter-weight alternative on more constrained hardware, given it performs identically here