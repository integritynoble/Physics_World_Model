# Calibration Sprint Service

## What is a Calibration Sprint?

A Calibration Sprint is a focused engagement where we help you calibrate your imaging system using the PWM toolkit. In 1-2 weeks, we work with your data to:

1. **Characterize** your system's mismatch profile
2. **Calibrate** operator parameters using Algorithm 1 (beam search) and Algorithm 2 (GPU differentiable)
3. **Validate** calibration quality with bootstrap confidence intervals
4. **Deliver** a calibrated operator + RunBundle with full provenance

## How It Works

### Phase 1: Intake (Day 1-2)
- Fill out the intake form below
- Share sample measurements and system specs
- We assess feasibility and create a calibration plan

### Phase 2: Calibration (Day 3-8)
- Run PWM calibration pipeline on your data
- Iterate on mismatch families and parameter spaces
- Generate identifiability reports
- Optimize with capture advisor recommendations

### Phase 3: Delivery (Day 9-10)
- Deliver calibrated operator configuration
- Provide RunBundle with SHA256-verified results
- Document calibration procedure for reproducibility
- Handoff meeting with recommendations

## Intake Form

To request a Calibration Sprint, provide:

| Field | Description |
|-------|-------------|
| **Organization** | Your institution or company |
| **Imaging modality** | e.g., CASSI, CACTI, SPC, custom |
| **System description** | Hardware specs, detector, optics |
| **Sample data** | 3-5 representative measurements (.npy/.npz) |
| **Known parameters** | Any calibrated parameters you already have |
| **Goal** | What reconstruction quality are you targeting? |
| **Timeline** | When do you need results? |

Submit via: [Create a GitHub Issue](../../issues/new?template=calibration_sprint.md)

## Pricing

Contact us for pricing. Academic discount available.

## Past Sprints

| Client | Modality | Outcome |
|--------|----------|---------|
| *Launching soon* | — | — |

## FAQ

**Q: Do I need to share my data publicly?**
A: No. All data is handled under NDA if requested.

**Q: Can I run the calibration myself?**
A: Yes! All tools are open source. The Sprint service provides expert guidance and GPU compute.

**Q: What if calibration doesn't converge?**
A: We provide a detailed report explaining why and recommendations for improving data collection.
