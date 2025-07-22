translations = {
    # Main elements
    'app_title': 'Intelligent Skin Cancer Diagnostic System',
    'app_description': 'This system uses **models specifically trained for skin cancer** with the ISIC 2019 dataset.',
    'settings': '⚙️ Settings',
    'settings_description': 'Analysis parameters',
    'image_upload': '📸 Image Upload',
    
    # PDF specific
    'pdf_report_title': 'Intelligent Medical Report',
    'report_date_time': 'Analysis date and time',
    'model_used': 'Model used',
    'threshold_value': 'Confidence threshold',
    'analyzed_image': 'ANALYZED IMAGE',
    'diagnosis_results': 'DIAGNOSIS RESULTS',
    'raw_confidence_value': 'Raw model value',
    'raw_value': 'Raw Value',
    
    # Configuration
    'debug_mode': '🐛 Debug Mode',
    'debug_help': 'Enables detailed info',
    'select_model': '🤖 Select model',
    'select_model_help': 'Different performance characteristics',
    'model_info': '📊 **Model Information:**',
    'parameters': '**Parameters:**',
    'layers': '**Layers:**',
    'confidence_threshold': '🎯 Confidence threshold',
    'confidence_help': 'Higher values = more confidence',
    'decision_threshold': '⚖️ Malignant/Benign Threshold',
    'decision_help': 'Lower values = more sensitive',
    'threshold_note': '💡 **Note**: Lower threshold increases sensitivity.',
    
    # Images
    'upload_prompt': 'Upload skin lesion image (JPG, JPEG, PNG)',
    'upload_help': 'Image should be clear',
    'original_image': 'Original Image',
    'processed_image': 'Processed Image (300x300)',
    
    # Results
    'processing_image': 'Processing...',
    'benign': 'Benign',
    'malignant': 'Malignant',
    'confidence': 'Confidence',
    'prediction': 'Diagnosis',
    'diagnosis_results': 'Diagnosis Results',
    'advanced_analysis': 'Advanced Analysis',
    'metrics_title': 'Metrics',
    'model_comparison_desc': 'Results of analysis of the same image with different models',
    
    # Basic metrics
    'accuracy': 'Accuracy',
    'sensitivity': 'Sensitivity',
    'specificity': 'Specificity',
    
    # Languages
    'language': 'Language',
    
    # Additional sections
    'results_interpretation': 'Results Interpretation',
    'model_comparison': 'Models Comparison',
    'consistency_analysis': 'Consistency Analysis',
    'pdf_success': 'PDF Report successfully generated',
    
    # Warnings and messages
    'low_confidence_warning': '⚠️ **Low confidence**: Confidence in diagnosis is below threshold. Consult a specialist.',
    'favorable_result': '✅ **Favorable result**: Lesion appears benign. Medical follow-up recommended.',
    'attention_required': '🚨 **Attention required**: Malignant features detected. Consult specialist urgently.',
    
    # Technical info and legal notices
    'technical_info': 'TECHNICAL INFORMATION',
    'technical_dataset': 'Dataset: ISIC 2019 (25,331 real images)',
    'technical_type': 'Type: Binary Classification (Benign/Malignant)',
    'technical_accuracy': 'Accuracy: ~69% (optimized for skin cancer)',
    'technical_input': 'Input: 300x300 pixels',
    'technical_architecture': 'Architecture: Transfer Learning with fine-tuning',
    'medical_disclaimer_title': 'MEDICAL DISCLAIMER',
    'medical_disclaimer_1': 'This system is for educational and research purposes only.',
    'medical_disclaimer_2': 'Results DO NOT constitute medical diagnosis.',
    'medical_disclaimer_3': 'ALWAYS consult with a dermatologist for professional diagnosis.',
    
    # PDF and reports
    'pdf_section_title': 'Generate PDF Report',
    'generate_pdf_button': 'Generate Complete PDF Report',
    'pdf_includes': 'The PDF report includes',
    'pdf_content_diagnosis': 'Diagnosis and image analysis',
    'pdf_content_comparison': 'Comparison between all models',
    'pdf_content_matrix': 'Confusion matrix and advanced metrics',
    'pdf_content_charts': 'MCC charts and statistical analysis',
    'pdf_content_mcnemar': 'McNemar tests',
    'pdf_content_recommendations': 'Medical recommendations',
    
    # Labels for charts and statistical analysis
    'confusion_matrix_title': 'Confusion Matrix',
    'metrics_dashboard_title': 'Metrics Dashboard',
    'statistical_analysis_title': 'Advanced Statistical Analysis',
    'model_analysis_description': 'Detailed analysis of the selected model performance',
    'statistical_analysis_description': 'Including Matthews Coefficient and McNemar Tests',
    'mcc_comparison_title': 'Comparative Summary of Matthews Coefficients (MCC)',
    'mcc_comparison_description': 'Comparison of all models based on Matthews Coefficient',
    'mcnemar_tests_title': 'McNemar Statistical Tests',
    'mcnemar_description': 'Statistical comparison between models',
    'activation_maps_title': 'Activation Maps Visualization',
    'activation_maps_description': 'Visualization of the regions that most influenced the diagnosis',
    'generating_activation_map': 'Generating activation map...',
    'activation_map_caption': 'Activation Map (Grad-CAM)',
    'heatmap_description': 'The heatmap shows the regions that most influenced the model\'s diagnosis. Red and yellow areas are the most relevant.',
    'activation_error': 'The activation map could not be generated for this model.',
    
    # Metrics interpretation
    'metrics_interpretation': '📋 Interpretation:',
    'accuracy_explanation': 'of predictions are correct',
    'sensitivity_explanation': 'of malignant cases are detected',
    'specificity_explanation': 'of benign cases are correctly identified',
    'precision_explanation': 'of cases classified as malignant are actually malignant',
    'f1_explanation': 'is the balance between precision and sensitivity',
    'mcc_explanation': '(Matthews Correlation Coefficient - balanced for uneven classes)',
    
    # Metrics table and other labels
    'confusion_matrix_chart': '🎯 Confusion Matrix',
    'advanced_metrics': '📈 Advanced Performance Metrics',
    'mcc_table_title': '📋 Summary Table - Matthews Coefficients',
    'generating_pdf': 'Generating PDF report...',
    'real_data_metrics': '✅ **Real Training Data**: Showing real metrics for {model} model on the ISIC 2019 dataset',
    'simulated_data_metrics': '⚠️ **Simulated Data**: Using example data for demonstration',
    'mcnemar_test_results': 'McNemar Test Results',
    
    # PDF chart labels
    'confidence_comparison_plot': 'Confidence Comparison',
    'inference_speed_plot': 'Inference Speed',
    'mcc_comparative_plot': 'MCC Comparative',
    'mcnemar_pvalues_plot': 'McNemar P-values',
    
    # Consistency analysis
    'perfect_consistency': '✅ **Perfect consistency**: All models agree on the diagnosis:',
    'inconsistency_detected': '⚠️ **Inconsistency detected**: Models do not agree on the diagnosis',
    'diagnoses_obtained': '**Diagnoses obtained**:',
    'recommendation_title': '💡 **Recommendation**:',
    'inconsistency_recommendation': 'When inconsistencies are detected, consultation with a specialist is recommended for confirmation.',
    
    # Confusion matrix interpretation
    'confusion_matrix_interpretation': '🔍 Confusion Matrix Interpretation',
    'matrix_elements': '**📊 Matrix Elements:**',
    'true_positives': '**True Positives (TP)**: Correctly identified malignant cases',
    'true_negatives': '**True Negatives (TN)**: Correctly identified benign cases',
    'false_positives': '**False Positives (FP)**: Benign cases classified as malignant',
    'false_negatives': '**False Negatives (FN)**: Malignant cases classified as benign',
    'medical_importance': '**🎯 Medical Importance:**',
    'fn_critical': '**False Negatives** are critical (missed cancer)',
    'fp_anxiety': '**False Positives** cause unnecessary anxiety',
    'recall_importance': '**High Recall** is crucial for early detection',
    'precision_importance': '**High Precision** reduces false alarms',
    
    # MCC interpretation
    'efficientnet_title': '**🥇 EfficientNetB4:**',
    'efficientnet_metrics': '- MCC: 0.7845 (**Excellent**)\n- Best overall balance\n- Recommended for clinical use\n- Superior diagnostic reliability',
    'resnet_title': '**🥈 ResNet152:**',
    'resnet_metrics': '- MCC: 0.6234 (**Good**)\n- Moderate performance\n- Viable alternative\n- Acceptable balance',
    'custom_cnn_title': '**🥉 Custom CNN:**',
    'custom_cnn_metrics': '- MCC: 0.5789 (**Good**)\n- Standard performance\n- Complementary option\n- Possible improvements',
    
    # Model evaluation and statistics
    'best_balance': 'Best overall balance',
    'recommended_clinical': 'Recommended for clinical use',
    'superior_reliability': 'Superior diagnostic reliability',
    'moderate_performance': 'Moderate performance',
    'viable_alternative': 'Viable alternative',
    'acceptable_balance': 'Acceptable balance',
    'standard_performance': 'Standard performance',
    'complementary_option': 'Complementary option',
    'possible_improvements': 'Possible improvements',
    'statistical_conclusions': 'Statistical Conclusions',
    'statistical_superiority': 'demonstrates significant statistical superiority',
    'superior_comparisons': '**Comparisons where {model} is superior:**',
    'no_statistical_diff': 'No statistically significant differences between models',
    'medical_interpretation': 'Medical interpretation',
    'for_model': 'for',
    'mcnemar_confirm': 'McNemar test results confirm that',
    'stat_diff': 'Shows statistically significant differences compared to other models',
    'diagnostic_superiority': 'Demonstrates superiority in diagnostic accuracy',
    'clinical_reliability': 'Provides greater reliability for clinical decisions',
    'robust_option': 'Is the most robust option for medical implementation',
    'justified_selection': 'Justifies its selection as the main model for diagnosis',
}
