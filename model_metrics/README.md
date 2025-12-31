# ASL Recognition Model - Metrics and Visualizations

This directory contains comprehensive metrics, visualizations, and analysis for the ASL Recognition CNN model.

## Generated Files

### Core Metrics Visualizations

1. **01_training_history.png** - Training and validation accuracy/loss curves
2. **02_confusion_matrix.png** - Confusion matrix (raw counts and normalized)
3. **03_class_metrics.png** - Per-class Precision, Recall, and F1-Score bar chart
4. **04_f1_score_heatmap.png** - F1-Score visualization for each class
5. **05_per_class_accuracy.png** - Accuracy for each class with color coding
6. **06_error_analysis.png** - Top 20 most confused class pairs
7. **07_overall_metrics.png** - Overall model performance summary
8. **08_class_distribution.png** - Class distribution in train/val sets
9. **09_sample_predictions.png** - 16 sample predictions with confidence scores

### Additional Analysis

10. **10_confidence_distribution.png** - Distribution of prediction confidence scores
11. **11_top_k_accuracy.png** - Top-K accuracy metrics (Top-1, Top-2, Top-3, Top-5)
12. **12_model_complexity.png** - Model architecture and parameter distribution
13. **13_performance_matrix.png** - Heatmap of Precision, Recall, F1-Score for all classes
14. **14_class_balance.png** - Detailed class balance analysis

### Reports

- **model_report.txt** - Comprehensive text report with all metrics

## Model Performance Summary

### Overall Metrics
- **Accuracy**: 96.52%
- **Precision**: 96.65%
- **Recall**: 96.52%
- **F1-Score**: 96.52%

### Class Performance
- **28/36 classes** have F1-Score ≥ 0.95 (Excellent)
- **32/36 classes** have F1-Score ≥ 0.90 (Very Good)
- **33/36 classes** have F1-Score ≥ 0.85 (Good)
- **3/36 classes** have F1-Score < 0.85 (Need Improvement)

### Best Performing Classes
- y, x, q, p, 7: Perfect 1.0000 F1-Score
- Many other classes also achieve perfect scores

### Classes Needing Improvement
- o: 0.7973 F1-Score
- 0: 0.8196 F1-Score
- w: 0.8377 F1-Score

## Model Architecture

- **Type**: CNN with 4 Convolutional Blocks
- **Total Parameters**: 1,057,764
- **Input Size**: 64x64x3
- **Number of Classes**: 36 (0-9, a-z)
- **Regularization**: Dropout (0.25-0.5), Batch Normalization
- **Data Augmentation**: Rotation, shifts, zoom

## Key Insights

1. **No Overfitting**: Training and validation accuracy are nearly identical (96.55% vs 96.52%)
2. **Excellent Generalization**: Model performs well on unseen data
3. **High Confidence**: Most predictions have high confidence scores
4. **Class Balance**: Some classes have more samples than others, but model handles this well
5. **Robust Performance**: Top-3 accuracy is very high, showing model is confident in predictions

## Usage

All visualizations are saved as high-resolution PNG files (300 DPI) suitable for:
- Presentations
- Reports
- Documentation
- Publications

View the images to understand:
- Which classes perform best/worst
- Common confusion patterns
- Model confidence levels
- Class distribution and balance


